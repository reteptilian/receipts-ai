from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import cast

from pydantic import ValidationError
from receipts_ai.firestore_transactions import (
    set_receipt_item_user_overrides,
    set_transaction_user_overrides,
    transactions_from_firestore,
)
from receipts_ai.models.transaction import (
    LineType,
    ReceiptItem,
    ReceiptItemUserOverrides,
    Transaction,
    TransactionUserOverrides,
)
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Input, Label, Static
from textual.widgets.data_table import RowKey

TransactionLoader = Callable[[], Sequence[Transaction]]

ReceiptItemParser = Callable[[str], object | None]
ReceiptItemFormatter = Callable[[ReceiptItem], str]


@dataclass(frozen=True)
class ReceiptItemColumn:
    label: str
    field_name: str
    parser: ReceiptItemParser
    formatter: ReceiptItemFormatter
    direct_edit: bool = False


RECEIPT_ITEM_COLUMNS = (
    ReceiptItemColumn(
        "Description",
        "description",
        lambda value: _parse_required_text(value, "Description"),
        lambda item: cast(str, _effective_receipt_item_value(item, "description")),
    ),
    ReceiptItemColumn(
        "Quantity",
        "quantity",
        lambda value: _parse_optional_float(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "quantity")),
    ),
    ReceiptItemColumn(
        "Unit price",
        "unit_price",
        lambda value: _parse_optional_text(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "unit_price")),
    ),
    ReceiptItemColumn(
        "Amount",
        "amount",
        lambda value: _parse_required_text(value, "Amount"),
        lambda item: str(_effective_receipt_item_value(item, "amount")),
    ),
    ReceiptItemColumn(
        "Discount amount",
        "discount_amount",
        lambda value: _parse_optional_text(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "discount_amount")),
    ),
    ReceiptItemColumn(
        "Discount description",
        "discount_description",
        lambda value: _parse_optional_text(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "discount_description")),
    ),
    ReceiptItemColumn(
        "Net amount",
        "net_amount",
        lambda value: _parse_required_text(value, "Net amount"),
        lambda item: str(_effective_receipt_item_value(item, "net_amount")),
    ),
    ReceiptItemColumn(
        "Line type",
        "line_type",
        lambda value: _parse_optional_line_type(value),
        lambda item: _format_optional(
            _format_line_type(_effective_receipt_item_value(item, "line_type"))
        ),
    ),
    ReceiptItemColumn(
        "Category ID",
        "category_id",
        lambda value: _parse_optional_text(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "category_id")),
    ),
    *(
        ReceiptItemColumn(
            f"Taxonomy {index}",
            f"taxonomy{index}",
            lambda value: _parse_optional_text(value),
            lambda item, field_name=f"taxonomy{index}": _format_optional(
                _effective_receipt_item_value(item, field_name)
            ),
        )
        for index in range(1, 10)
    ),
)


def _open_file_in_external_viewer(filepath: str) -> None:
    """Opens a file in the system's default external viewer."""
    system_platform = platform.system()
    try:
        if system_platform == "Darwin":  # macOS
            subprocess.run(["open", filepath], check=True)
        elif system_platform == "Windows":  # Windows
            os.startfile(filepath)
        else:  # Linux (and other Unix-like systems)
            subprocess.run(["xdg-open", filepath], check=True)
    except Exception:
        # Ignore errors when opening external viewer to avoid crashing the app
        pass


class CellEditScreen(ModalScreen[str]):
    """Screen for editing a single cell's value."""

    def __init__(self, label: str, value: str) -> None:
        super().__init__()
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="cell-edit-dialog"):
                yield Label(f"Edit {self._label}")
                yield Input(self._value, id="cell-edit-input")
                with Horizontal(id="cell-edit-buttons"):
                    yield Static("Enter to Save / Esc to Cancel", id="cell-edit-help")

    def on_mount(self) -> None:
        self.query_one("#cell-edit-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss()

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]


class ReceiptItemsScreen(Screen[None]):
    """Screen showing the receipt items for one transaction."""

    app: ReceiptsAIApp

    BINDINGS = [
        ("e", "edit_cell", "Edit cell"),
        ("escape", "back", "Back"),
        ("q", "back", "Back"),
    ]

    def __init__(self, transaction: Transaction) -> None:
        super().__init__()
        self._transaction = transaction

    def action_back(self) -> None:
        self.dismiss()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="receipt-header"):
            yield Label("Date", classes="receipt-header-label")
            yield Input(
                _effective_transaction_date(self._transaction).isoformat(),
                id="receipt-date",
                compact=True,
            )
            yield Label("Payee", classes="receipt-header-label")
            yield Input(
                _effective_transaction_payee(self._transaction), id="receipt-payee", compact=True
            )
            yield Label("Amount", classes="receipt-header-label")
            yield Input(
                _effective_transaction_amount(self._transaction), id="receipt-amount", compact=True
            )
            yield Static(self._transaction.currency, id="receipt-currency")
        yield DataTable(id="receipt-items")
        yield Static("", id="receipt-edit-status")
        yield Footer()

    def on_mount(self) -> None:
        table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        table.cursor_type = "cell"
        table.zebra_stripes = True
        table.add_columns(*(column.label for column in RECEIPT_ITEM_COLUMNS))

        if (
            getattr(self._transaction, "ingestion_type", None) == "receipt_img"
            and (ingestion_file_url := getattr(self._transaction, "ingestion_file_url", None))
        ):
            self.run_worker(
                lambda: _open_file_in_external_viewer(ingestion_file_url),
                thread=True,
                name=f"open-viewer-{self._transaction.id}",
            )

        receipt = self._transaction.receipt
        if receipt is None:
            return

        for item in receipt.items:
            table.add_row(*_receipt_item_row(item))

        table.focus()

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        data_table = cast(DataTable[str], event.data_table)
        if data_table.id == "receipt-items":
            self.action_edit_cell()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._commit_header_input(event.input)

    def on_input_blurred(self, event: Input.Blurred) -> None:
        if event.input.id in {"receipt-date", "receipt-payee", "receipt-amount"}:
            self._commit_header_input(event.input)

    def action_edit_cell(self) -> None:
        table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        coordinate = table.cursor_coordinate
        if not table.is_valid_coordinate(coordinate):
            return

        column = RECEIPT_ITEM_COLUMNS[coordinate.column]
        value = table.get_cell_at(coordinate)

        def check_edit(new_value: str | None) -> None:
            if new_value is not None:
                self._commit_cell_edit(coordinate, new_value)

        self.app.push_screen(CellEditScreen(column.label, str(value)), check_edit)

    def _commit_header_input(self, input_widget: Input) -> None:
        value = input_widget.value.strip()
        try:
            match input_widget.id:
                case "receipt-date":
                    self._set_transaction_override("transaction_date", _parse_optional_date(value))
                case "receipt-payee":
                    self._set_transaction_override("payee", _parse_optional_text(value))
                case "receipt-amount":
                    self._set_transaction_override("amount", _parse_optional_text(value))
                case _:
                    return
        except (ValueError, ValidationError) as exc:
            self._show_edit_error(str(exc))
            return

        self._show_edit_status("Saving transaction override...")
        self.run_worker(
            self._persist_transaction_overrides,
            thread=True,
            name=f"persist-tx-{self._transaction.id}",
        )

    def _commit_cell_edit(self, coordinate: Coordinate, raw_value: str) -> None:
        receipt = self._transaction.receipt
        if receipt is None:
            return

        item = receipt.items[coordinate.row]
        column = RECEIPT_ITEM_COLUMNS[coordinate.column]
        try:
            parsed_value = column.parser(raw_value.strip())
            self._set_receipt_item_override(item, column.field_name, parsed_value)
        except (ValueError, ValidationError) as exc:
            self._show_edit_error(str(exc))
            return

        table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        table.update_cell_at(
            coordinate,
            column.formatter(item),
            update_width=True,
        )
        self._show_edit_status(f"Saving {column.label}...")
        self.run_worker(
            lambda: self._persist_receipt_item_overrides(item, coordinate.row),
            thread=True,
            name=f"persist-item-{item.id or coordinate.row}",
        )

    def _persist_transaction_overrides(self) -> None:
        try:
            set_transaction_user_overrides(
                self._transaction.id, cast(TransactionUserOverrides, self._transaction.user_overrides)
            )
            self.app.call_from_thread(self._show_edit_status, "Transaction override persisted")
        except Exception as exc:
            self.app.call_from_thread(self._show_edit_error, f"Failed to persist: {exc}")

    def _persist_receipt_item_overrides(self, item: ReceiptItem, index: int) -> None:
        try:
            set_receipt_item_user_overrides(
                self._transaction.id,
                cast(ReceiptItemUserOverrides, item.user_overrides),
                receipt_item_id=item.id,
                item_index=None if item.id else index,
            )
            self.app.call_from_thread(self._show_edit_status, "Item override persisted")
        except Exception as exc:
            self.app.call_from_thread(self._show_edit_error, f"Failed to persist item: {exc}")

    def _set_transaction_override(self, field_name: str, value: object | None) -> None:
        overrides = self._transaction.user_overrides or TransactionUserOverrides()
        update = overrides.model_dump()
        update[field_name] = value
        self._transaction.user_overrides = TransactionUserOverrides.model_validate(update)

    def _set_receipt_item_override(
        self,
        item: ReceiptItem,
        field_name: str,
        value: object | None,
    ) -> None:
        overrides = item.user_overrides or ReceiptItemUserOverrides()
        update = overrides.model_dump()
        update[field_name] = value
        item.user_overrides = ReceiptItemUserOverrides.model_validate(update)

    def _show_edit_status(self, message: str) -> None:
        status = self.query_one("#receipt-edit-status", Static)
        status.remove_class("error")
        status.update(escape(message))

    def _show_edit_error(self, message: str) -> None:
        status = self.query_one("#receipt-edit-status", Static)
        status.add_class("error")
        status.update(escape(message))


class ReceiptsAIApp(App[None]):
    """Textual app for browsing Receipts AI transactions."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        dock: top;
        height: 1;
        padding: 0 1;
    }

    #status.error {
        color: $error;
    }

    DataTable {
        height: 1fr;
    }

    #receipt-header {
        dock: top;
        height: auto;
        padding: 0 1;
        align-vertical: middle;
    }

    .receipt-header-label {
        width: auto;
        padding: 0 1;
    }

    #receipt-date {
        width: 14;
    }

    #receipt-payee {
        width: 1fr;
    }

    #receipt-amount {
        width: 16;
    }

    #receipt-currency {
        width: 5;
        padding: 0 1;
    }

    #cell-edit-dialog {
        padding: 1 2;
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
    }

    #cell-edit-input {
        margin: 1 0;
    }

    #cell-edit-help {
        color: $text-muted;
        text-align: center;
        width: 1fr;
    }

    #receipt-edit-status {
        dock: bottom;
        height: 1;
        padding: 0 1;
    }

    #receipt-edit-status.error {
        color: $error;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        transaction_loader: TransactionLoader = transactions_from_firestore,
    ) -> None:
        super().__init__()
        self._transaction_loader = transaction_loader
        self._transactions_by_id: dict[str, Transaction] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Loading transactions...", id="status")
        yield DataTable(id="transactions")
        yield Footer()

    def on_mount(self) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(
            "Date",
            "Payee",
            "Description",
            "Ingestion file",
            "Receipt?",
            "Amount",
        )
        table.focus()
        self.run_worker(self._load_transactions, thread=True, name="load-transactions")

    def _load_transactions(self) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        cursor_coordinate = table.cursor_coordinate
        cursor_row_key = None
        if table.is_valid_coordinate(cursor_coordinate):
            cursor_row_key = table.coordinate_to_cell_key(cursor_coordinate).row_key

        try:
            transactions = list(self._transaction_loader())
        except Exception as exc:  # pragma: no cover - exercised through app runtime.
            self.call_from_thread(self._show_error, exc)
            return

        self.call_from_thread(self._show_transactions, transactions, cursor_row_key)

    def _show_transactions(
        self, transactions: Sequence[Transaction], cursor_row_key: RowKey | None = None
    ) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        table.clear()
        sorted_transactions = sorted(transactions, key=_transaction_sort_key)
        self._transactions_by_id = {
            transaction.id: transaction for transaction in sorted_transactions
        }
        for transaction in sorted_transactions:
            table.add_row(
                _effective_transaction_date(transaction).isoformat(),
                _effective_transaction_payee(transaction),
                transaction.description or "",
                transaction.ingestion_filename or "",
                _format_receipt_indicator(transaction),
                _format_amount(_effective_transaction_amount(transaction), transaction.currency),
                key=transaction.id,
            )

        if cursor_row_key is not None:
            try:
                table.move_cursor(row=table.get_row_index(cursor_row_key))
            except Exception:
                pass

        status = self.query_one("#status", Static)
        status.remove_class("error")
        transaction_count = len(transactions)
        status.update(
            f"{transaction_count} transaction{'s' if transaction_count != 1 else ''}"
            if transaction_count
            else "No transactions found"
        )

    def _show_error(self, exc: Exception) -> None:
        status = self.query_one("#status", Static)
        status.add_class("error")
        status.update(f"Unable to load transactions: {exc}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        data_table = cast(DataTable[str], event.data_table)
        if data_table.id != "transactions":
            return

        transaction_id = event.row_key.value
        if transaction_id is None:
            return

        transaction = self._transactions_by_id.get(transaction_id)
        if transaction is None or not _has_receipt_items(transaction):
            return

        def on_return(_: None) -> None:
            self.run_worker(self._load_transactions, thread=True, name="load-transactions")

        self.push_screen(ReceiptItemsScreen(transaction), on_return)


def _transaction_sort_key(transaction: Transaction) -> tuple[object, ...]:
    try:
        amount = Decimal(_effective_transaction_amount(transaction))
    except InvalidOperation:
        return (
            _effective_transaction_date(transaction),
            1,
            _effective_transaction_amount(transaction),
        )

    return (_effective_transaction_date(transaction), 0, amount)


def _format_amount(amount: str, currency: str) -> str:
    try:
        decimal_amount = Decimal(amount)
    except InvalidOperation:
        formatted_amount = amount
    else:
        formatted_amount = f"{decimal_amount:,.2f}"

    return f"{formatted_amount} {currency}"


def _format_receipt_indicator(transaction: Transaction) -> str:
    return "Y" if _has_receipt_items(transaction) else ""


def _has_receipt_items(transaction: Transaction) -> bool:
    receipt = transaction.receipt
    return receipt is not None and bool(receipt.items)


def _receipt_item_row(item: ReceiptItem) -> list[str]:
    return [column.formatter(item) for column in RECEIPT_ITEM_COLUMNS]


def _format_optional(value: object | None) -> str:
    return "" if value is None else str(value)


def _format_line_type(value: object | None) -> str | None:
    if isinstance(value, LineType):
        return value.value
    return None if value is None else str(value)


def _effective_transaction_date(transaction: Transaction) -> date:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.transaction_date is not None:
        return overrides.transaction_date

    return transaction.transaction_date


def _effective_transaction_payee(transaction: Transaction) -> str:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.payee is not None:
        return overrides.payee

    return transaction.payee or ""


def _effective_transaction_amount(transaction: Transaction) -> str:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.amount is not None:
        return overrides.amount

    return transaction.amount


def _effective_receipt_item_value(item: ReceiptItem, field_name: str) -> object | None:
    overrides = item.user_overrides
    if overrides is not None:
        override_value = getattr(overrides, field_name)
        if override_value is not None:
            return override_value

    return getattr(item, field_name)


def _parse_required_text(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} must not be empty")

    return value


def _parse_optional_text(value: str) -> str | None:
    return value or None


def _parse_optional_float(value: str) -> float | None:
    if not value:
        return None

    return float(value)


def _parse_optional_date(value: str) -> date | None:
    if not value:
        return None

    return date.fromisoformat(value)


def _parse_optional_line_type(value: str) -> LineType | None:
    if not value:
        return None

    return LineType(value)


def main() -> None:
    ReceiptsAIApp().run()

