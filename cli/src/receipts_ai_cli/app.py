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
    link_bank_statement_transaction_to_receipt,
    save_transaction_review_edits,
    transactions_from_firestore,
    unlink_bank_statement_transaction_from_receipt,
)
from receipts_ai.models.transaction import (
    CategoryAllocation,
    LineType,
    ReceiptItem,
    ReceiptItemUserOverrides,
    Transaction,
    TransactionUserOverrides,
    UserCategoryAllocation,
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


class TransactionReviewScreen(Screen[None]):
    """Screen for drafting and saving transaction review edits."""

    app: ReceiptsAIApp

    BINDINGS = [
        ("e", "edit_cell", "Edit cell"),
        ("a", "add_category_allocation", "Add allocation"),
        ("d", "delete_category_allocation", "Delete allocation"),
        ("s", "save_and_exit", "Save and Exit"),
        ("escape", "exit_without_saving", "Exit Without Saving"),
        ("q", "exit_without_saving", "Exit Without Saving"),
    ]

    def __init__(
        self,
        transaction: Transaction,
        *,
        receipt_transaction: Transaction | None = None,
    ) -> None:
        super().__init__()
        self._source_transaction = transaction
        self._source_receipt_transaction = receipt_transaction
        self._transaction = transaction.model_copy(deep=True)
        self._receipt_transaction = (
            receipt_transaction.model_copy(deep=True)
            if receipt_transaction is not None
            else self._transaction
            if _has_receipt_items(transaction)
            else None
        )
        self._dirty = False

    def action_exit_without_saving(self) -> None:
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
            yield Label("Amount", classes="receipt-header-label")
            yield Input(
                _effective_transaction_amount(self._transaction), id="receipt-amount", compact=True
            )
            yield Static(self._transaction.currency, id="receipt-currency")
            yield Label("Payee", classes="receipt-header-label")
            yield Input(
                _effective_transaction_payee(self._transaction), id="receipt-payee", compact=True
            )
            yield Label("Description", classes="receipt-header-label")
            yield Input(
                _effective_transaction_description(self._transaction),
                id="receipt-description",
                compact=True,
            )
        yield Static("", id="receipt-edit-status")
        yield Static("Category Allocations", id="category-allocations-title")
        yield DataTable(id="category-allocations")
        yield Static("Receipt Items", id="receipt-items-title")
        yield DataTable(id="receipt-items")
        yield Static(
            "s Save and Exit | a Add allocation | d Delete allocation | "
            "e Edit cell | Esc/q Exit Without Saving",
            id="review-controls",
        )

    def on_mount(self) -> None:
        allocations_table = cast(DataTable[str], self.query_one("#category-allocations", DataTable))
        allocations_table.cursor_type = "cell"
        allocations_table.zebra_stripes = True
        allocations_table.add_columns("Category ID", "Amount")
        self._refresh_category_allocations_table()

        table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        table.cursor_type = "cell"
        table.zebra_stripes = True
        table.add_columns(*(column.label for column in RECEIPT_ITEM_COLUMNS))

        receipt_transaction = self._receipt_transaction
        if (
            receipt_transaction is not None
            and getattr(receipt_transaction, "ingestion_type", None) == "receipt_img"
            and (ingestion_file_url := getattr(receipt_transaction, "ingestion_file_url", None))
        ):
            self.run_worker(
                lambda: _open_file_in_external_viewer(ingestion_file_url),
                thread=True,
                name=f"open-viewer-{receipt_transaction.id}",
            )

        receipt = receipt_transaction.receipt if receipt_transaction is not None else None
        if receipt is None:
            allocations_table.focus()
            return

        for item in receipt.items:
            table.add_row(*_receipt_item_row(item))

        allocations_table.focus()

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        data_table = cast(DataTable[str], event.data_table)
        if data_table.id in {"category-allocations", "receipt-items"}:
            self.action_edit_cell()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._commit_header_input(event.input)

    def on_input_blurred(self, event: Input.Blurred) -> None:
        if event.input.id in {
            "receipt-date",
            "receipt-payee",
            "receipt-description",
            "receipt-amount",
        }:
            self._commit_header_input(event.input)

    def action_edit_cell(self) -> None:
        table = self._focused_edit_table()
        if table is None:
            return
        coordinate = table.cursor_coordinate
        if not table.is_valid_coordinate(coordinate):
            return

        column_label = (
            ("Category ID", "Amount")[coordinate.column]
            if table.id == "category-allocations"
            else RECEIPT_ITEM_COLUMNS[coordinate.column].label
        )
        value = table.get_cell_at(coordinate)

        def check_edit(new_value: str | None) -> None:
            if new_value is not None:
                if table.id == "category-allocations":
                    self._commit_category_allocation_cell_edit(coordinate, new_value)
                else:
                    self._commit_receipt_item_cell_edit(coordinate, new_value)

        self.app.push_screen(CellEditScreen(column_label, str(value)), check_edit)

    def action_add_category_allocation(self) -> None:
        allocations = _effective_category_allocations(self._transaction)
        allocations.append(UserCategoryAllocation(category_id="New Category", amount="0.00"))
        self._set_category_allocation_overrides(allocations)
        self._refresh_category_allocations_table()
        self._mark_dirty("Added category allocation")

    def action_delete_category_allocation(self) -> None:
        table = cast(DataTable[str], self.query_one("#category-allocations", DataTable))
        coordinate = table.cursor_coordinate
        if not table.is_valid_coordinate(coordinate):
            return
        allocations = _effective_category_allocations(self._transaction)
        if coordinate.row >= len(allocations):
            return
        del allocations[coordinate.row]
        self._set_category_allocation_overrides(allocations)
        self._refresh_category_allocations_table()
        self._mark_dirty("Deleted category allocation")

    def action_save_and_exit(self) -> None:
        try:
            self._validate_save()
        except ValueError as exc:
            self._show_edit_error(str(exc))
            return

        self._show_edit_status("Saving review edits...")
        self.run_worker(
            self._persist_review_edits,
            thread=True,
            name=f"persist-review-{self._source_transaction.id}",
        )

    def _commit_header_input(self, input_widget: Input) -> None:
        value = input_widget.value.strip()
        try:
            match input_widget.id:
                case "receipt-date":
                    self._set_transaction_override("transaction_date", _parse_optional_date(value))
                case "receipt-payee":
                    self._set_transaction_override("payee", _parse_optional_text(value))
                case "receipt-description":
                    self._set_transaction_override("description", _parse_optional_text(value))
                case "receipt-amount":
                    self._set_transaction_override("amount", _parse_optional_text(value))
                case _:
                    return
        except (ValueError, ValidationError) as exc:
            self._show_edit_error(_field_edit_error_message(_header_input_label(input_widget), exc))
            return

        self._mark_dirty("Draft transaction edit")

    def _commit_category_allocation_cell_edit(self, coordinate: Coordinate, raw_value: str) -> None:
        allocations = _effective_category_allocations(self._transaction)
        if coordinate.row >= len(allocations):
            return
        allocation = allocations[coordinate.row]
        update = allocation.model_dump()
        try:
            match coordinate.column:
                case 0:
                    update["category_id"] = _parse_required_text(raw_value.strip(), "Category ID")
                case 1:
                    update["amount"] = _parse_required_decimal_text(raw_value.strip(), "Amount")
                case _:
                    return
            allocations[coordinate.row] = UserCategoryAllocation.model_validate(update)
        except (ValueError, ValidationError) as exc:
            self._show_edit_error(
                _field_edit_error_message(
                    ("Category ID", "Category allocation amount")[coordinate.column], exc
                )
            )
            return

        self._set_category_allocation_overrides(allocations)
        self._refresh_category_allocations_table()
        self._mark_dirty("Draft category allocation edit")

    def _commit_receipt_item_cell_edit(self, coordinate: Coordinate, raw_value: str) -> None:
        if self._receipt_transaction is None:
            return
        receipt = self._receipt_transaction.receipt
        if receipt is None:
            return

        item = receipt.items[coordinate.row]
        column = RECEIPT_ITEM_COLUMNS[coordinate.column]
        try:
            parsed_value = column.parser(raw_value.strip())
            self._set_receipt_item_override(item, column.field_name, parsed_value)
        except (ValueError, ValidationError) as exc:
            self._show_edit_error(_field_edit_error_message(column.label, exc))
            return

        table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        table.update_cell_at(
            coordinate,
            column.formatter(item),
            update_width=True,
        )
        self._mark_dirty(f"Draft {column.label} edit")

    def _persist_review_edits(self) -> None:
        try:
            receipt_items = None
            receipt_transaction_id = None
            if self._receipt_transaction is not None and self._receipt_transaction.receipt:
                receipt_items = self._receipt_transaction.receipt.items
                receipt_transaction_id = (
                    self._source_receipt_transaction.id
                    if (self._source_receipt_transaction is not None)
                    else self._source_transaction.id
                )

            save_transaction_review_edits(
                self._source_transaction.id,
                self._transaction.user_overrides or TransactionUserOverrides(),
                receipt_transaction_id=receipt_transaction_id,
                receipt_items=receipt_items,
            )
            self._source_transaction.user_overrides = self._transaction.user_overrides
            if self._source_receipt_transaction is not None and self._receipt_transaction:
                self._source_receipt_transaction.receipt = self._receipt_transaction.receipt
            elif self._receipt_transaction is not None:
                self._source_transaction.receipt = self._receipt_transaction.receipt
            self._dirty = False
            self.app.call_from_thread(self.dismiss)
        except Exception as exc:
            self.app.call_from_thread(self._show_edit_error, f"Failed to persist: {exc}")

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

    def _set_category_allocation_overrides(
        self, allocations: Sequence[CategoryAllocation | UserCategoryAllocation]
    ) -> None:
        overrides = self._transaction.user_overrides or TransactionUserOverrides()
        update = overrides.model_dump()
        update["category_allocations"] = [
            UserCategoryAllocation(
                category_id=allocation.category_id,
                amount=allocation.amount,
            )
            for allocation in allocations
        ]
        self._transaction.user_overrides = TransactionUserOverrides.model_validate(update)

    def _refresh_category_allocations_table(self) -> None:
        table = cast(DataTable[str], self.query_one("#category-allocations", DataTable))
        table.clear(columns=False)
        for allocation in _effective_category_allocations(self._transaction):
            table.add_row(allocation.category_id, allocation.amount)

    def _validate_save(self) -> None:
        transaction_amount = _decimal_amount(
            _effective_transaction_amount(self._transaction), "Transaction amount"
        )
        allocations = _effective_category_allocations(self._transaction)
        allocation_total = sum(
            (
                _decimal_amount(allocation.amount, "Category allocation amount")
                for allocation in allocations
            ),
            Decimal("0"),
        )
        if allocation_total != transaction_amount:
            raise ValueError(
                "Save blocked: category allocations must add up to the transaction amount. "
                f"The allocations total {allocation_total}, but the transaction amount is "
                f"{transaction_amount}."
            )

        if self._receipt_transaction is None or self._receipt_transaction.receipt is None:
            return

        receipt_item_total = sum(
            (
                _decimal_amount(
                    str(_effective_receipt_item_value(item, "amount")), "Receipt item amount"
                )
                for item in self._receipt_transaction.receipt.items
            ),
            Decimal("0"),
        )
        expected_receipt_total = -transaction_amount
        if receipt_item_total != expected_receipt_total:
            raise ValueError(
                "Save blocked: receipt item amounts must add up to the sign-flipped "
                f"transaction amount. The receipt items total {receipt_item_total}, but the "
                f"expected total is {expected_receipt_total}."
            )

    def _focused_edit_table(self) -> DataTable[str] | None:
        allocations_table = cast(DataTable[str], self.query_one("#category-allocations", DataTable))
        receipt_items_table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        focused = self.focused
        if focused is allocations_table:
            return allocations_table
        if focused is receipt_items_table:
            return receipt_items_table
        return receipt_items_table if receipt_items_table.has_focus else allocations_table

    def _mark_dirty(self, message: str) -> None:
        self._dirty = True
        self._show_edit_status(f"{message} (unsaved)")

    def _show_edit_status(self, message: str) -> None:
        status = self.query_one("#receipt-edit-status", Static)
        status.remove_class("error")
        status.update(escape(message))

    def _show_edit_error(self, message: str) -> None:
        status = self.query_one("#receipt-edit-status", Static)
        status.add_class("error")
        status.update(escape(message))


ReceiptItemsScreen = TransactionReviewScreen


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

    #category-allocations {
        height: 6;
    }

    #category-allocations-title,
    #receipt-items-title {
        height: 1;
        padding: 0 1;
        text-style: bold;
    }

    #receipt-header {
        height: 1;
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

    #receipt-description {
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
        height: 1;
        padding: 0 1;
    }

    #receipt-edit-status.error {
        color: $error;
    }

    #review-controls {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("space", "toggle_transaction_selection", "Select"),
        ("l", "link_selected_transactions", "Link"),
        ("u", "unlink_transaction", "Unlink"),
    ]

    def __init__(
        self,
        *,
        transaction_loader: TransactionLoader = transactions_from_firestore,
    ) -> None:
        super().__init__()
        self._transaction_loader = transaction_loader
        self._transactions_by_id: dict[str, Transaction] = {}
        self._receipt_transactions_by_display_id: dict[str, Transaction] = {}
        self._selected_transaction_ids: set[str] = set()

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
        display_transactions = _display_transactions(sorted_transactions)
        self._transactions_by_id = {
            transaction.id: transaction for transaction in sorted_transactions
        }
        self._receipt_transactions_by_display_id = _receipt_transactions_by_display_id(
            display_transactions, self._transactions_by_id
        )
        self._selected_transaction_ids.intersection_update(self._transactions_by_id)
        for transaction in display_transactions:
            table.add_row(
                _effective_transaction_date(transaction).isoformat(),
                _effective_transaction_payee(transaction),
                transaction.description or "",
                transaction.ingestion_filename or "",
                _format_receipt_indicator(
                    self._receipt_transactions_by_display_id.get(transaction.id, transaction),
                    selected=transaction.id in self._selected_transaction_ids,
                ),
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
        transaction_count = len(display_transactions)
        record_count = len(transactions)
        selected_count = len(self._selected_transaction_ids)
        selected_status = f" | {selected_count} selected" if selected_count else ""
        record_status = f" ({record_count} records)" if record_count != transaction_count else ""
        status.update(
            f"{transaction_count} transaction{'s' if transaction_count != 1 else ''}"
            f"{record_status}{selected_status}"
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
        if transaction is None:
            return
        receipt_transaction = self._receipt_transactions_by_display_id.get(transaction_id)
        if receipt_transaction is transaction:
            receipt_transaction = None

        def on_return(_: None) -> None:
            self.run_worker(self._load_transactions, thread=True, name="load-transactions")

        self.push_screen(
            TransactionReviewScreen(transaction, receipt_transaction=receipt_transaction),
            on_return,
        )

    def action_toggle_transaction_selection(self) -> None:
        transaction_id = self._current_transaction_id()
        if transaction_id is None:
            return
        if transaction_id in self._selected_transaction_ids:
            self._selected_transaction_ids.remove(transaction_id)
        else:
            self._selected_transaction_ids.add(transaction_id)
        self._refresh_visible_transactions()

    def action_link_selected_transactions(self) -> None:
        if len(self._selected_transaction_ids) != 2:
            self._show_status("Select exactly one BST and one RBT, then press l")
            return

        selected = [
            self._transactions_by_id[transaction_id]
            for transaction_id in self._selected_transaction_ids
            if transaction_id in self._transactions_by_id
        ]
        bank_statement = next(
            (
                transaction
                for transaction in selected
                if _is_bank_statement_transaction(transaction)
            ),
            None,
        )
        receipt_based = next(
            (transaction for transaction in selected if _is_receipt_based_transaction(transaction)),
            None,
        )
        if bank_statement is None or receipt_based is None:
            self._show_status("Selection must contain one BST and one RBT", error=True)
            return

        self._show_status("Linking transactions...")
        self.run_worker(
            lambda: self._persist_transaction_link(bank_statement.id, receipt_based.id),
            thread=True,
            name="link-transactions",
        )

    def action_unlink_transaction(self) -> None:
        transaction_id = self._current_transaction_id()
        if transaction_id is None:
            return
        transaction = self._transactions_by_id.get(transaction_id)
        if transaction is None:
            return
        if not _is_bank_statement_transaction(transaction):
            self._show_status("Move to the linked BST row and press u", error=True)
            return
        if transaction.linked_receipt_based_transaction_id is None:
            self._show_status("Current BST is not linked", error=True)
            return

        self._show_status("Unlinking transaction...")
        self.run_worker(
            lambda: self._persist_transaction_unlink(transaction.id),
            thread=True,
            name="unlink-transaction",
        )

    def _persist_transaction_link(self, bank_statement_id: str, receipt_based_id: str) -> None:
        try:
            link_bank_statement_transaction_to_receipt(bank_statement_id, receipt_based_id)
            self._selected_transaction_ids.clear()
            self.call_from_thread(self._show_status, "Transactions linked")
            self.call_from_thread(
                lambda: self.run_worker(
                    self._load_transactions, thread=True, name="load-transactions"
                )
            )
        except Exception as exc:
            self.call_from_thread(self._show_status, f"Failed to link: {exc}", True)

    def _persist_transaction_unlink(self, bank_statement_id: str) -> None:
        try:
            unlink_bank_statement_transaction_from_receipt(bank_statement_id)
            self._selected_transaction_ids.clear()
            self.call_from_thread(self._show_status, "Transaction unlinked")
            self.call_from_thread(
                lambda: self.run_worker(
                    self._load_transactions, thread=True, name="load-transactions"
                )
            )
        except Exception as exc:
            self.call_from_thread(self._show_status, f"Failed to unlink: {exc}", True)

    def _current_transaction_id(self) -> str | None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        coordinate = table.cursor_coordinate
        if not table.is_valid_coordinate(coordinate):
            return None
        row_key = table.coordinate_to_cell_key(coordinate).row_key
        value = row_key.value
        return value if isinstance(value, str) else None

    def _refresh_visible_transactions(self) -> None:
        self._show_transactions(list(self._transactions_by_id.values()))

    def _show_status(self, message: str, error: bool = False) -> None:
        status = self.query_one("#status", Static)
        if error:
            status.add_class("error")
        else:
            status.remove_class("error")
        status.update(escape(message))


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


def _display_transactions(transactions: Sequence[Transaction]) -> list[Transaction]:
    linked_receipt_ids = {
        transaction.linked_receipt_based_transaction_id
        for transaction in transactions
        if _is_bank_statement_transaction(transaction)
        and transaction.linked_receipt_based_transaction_id is not None
    }
    return [transaction for transaction in transactions if transaction.id not in linked_receipt_ids]


def _receipt_transactions_by_display_id(
    display_transactions: Sequence[Transaction],
    transactions_by_id: dict[str, Transaction],
) -> dict[str, Transaction]:
    receipt_transactions: dict[str, Transaction] = {}
    for transaction in display_transactions:
        linked_receipt_id = transaction.linked_receipt_based_transaction_id
        if linked_receipt_id is not None and linked_receipt_id in transactions_by_id:
            receipt_transactions[transaction.id] = transactions_by_id[linked_receipt_id]
        elif _has_receipt_items(transaction):
            receipt_transactions[transaction.id] = transaction
    return receipt_transactions


def _format_receipt_indicator(transaction: Transaction, *, selected: bool = False) -> str:
    indicator = "Y" if _has_receipt_items(transaction) else ""
    if selected:
        return f"{indicator}*" if indicator else "*"
    return indicator


def _has_receipt_items(transaction: Transaction) -> bool:
    receipt = transaction.receipt
    return receipt is not None and bool(receipt.items)


def _is_bank_statement_transaction(transaction: Transaction) -> bool:
    record_type = getattr(transaction, "record_type", None)
    return record_type == "bank_statement" or transaction.source == "bank_statement"


def _is_receipt_based_transaction(transaction: Transaction) -> bool:
    record_type = getattr(transaction, "record_type", None)
    return record_type == "receipt_based" or transaction.source == "receipt"


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


def _effective_transaction_description(transaction: Transaction) -> str:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.description is not None:
        return overrides.description

    return transaction.description or ""


def _effective_transaction_amount(transaction: Transaction) -> str:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.amount is not None:
        return overrides.amount

    return transaction.amount


def _effective_category_allocations(
    transaction: Transaction,
) -> list[CategoryAllocation | UserCategoryAllocation]:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.category_allocations is not None:
        return list(overrides.category_allocations)

    return list(transaction.category_allocations or [])


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


def _parse_required_decimal_text(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} must not be empty")
    _decimal_amount(value, label)
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


def _decimal_amount(value: str, label: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a decimal amount") from exc


def _header_input_label(input_widget: Input) -> str:
    match input_widget.id:
        case "receipt-date":
            return "transaction date"
        case "receipt-amount":
            return "transaction amount"
        case "receipt-payee":
            return "payee"
        case "receipt-description":
            return "description"
        case _:
            return "field"


def _field_edit_error_message(label: str, exc: ValueError | ValidationError) -> str:
    return f"Could not update {label}: {_validation_error_explanation(exc)}"


def _validation_error_explanation(exc: ValueError | ValidationError) -> str:
    if isinstance(exc, ValidationError):
        for error in exc.errors():
            error_type = str(error.get("type", ""))
            location = ".".join(str(part) for part in error.get("loc", ()))
            if error_type == "string_pattern_mismatch" and (
                location.endswith("amount")
                or location.endswith("unit_price")
                or location.endswith("discount_amount")
                or location.endswith("net_amount")
            ):
                return "enter a decimal amount like -12.34, with up to 4 decimal places."
            if error_type == "string_too_short":
                return "this value cannot be blank."
            if error_type == "greater_than":
                return "enter a number greater than zero."
            if error_type == "enum":
                return "enter one of the supported values for this field."
        return "the value does not match the expected format."

    message = str(exc)
    if "Invalid isoformat string" in message:
        return "enter a date as YYYY-MM-DD, or leave it blank to clear the override."
    if "must be a decimal amount" in message:
        return "enter a decimal amount like -12.34, with up to 4 decimal places."
    if message:
        return message
    return "the value does not match the expected format."


def main() -> None:
    ReceiptsAIApp().run()
