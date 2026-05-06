from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from typing import cast

from receipts_ai.firestore_transactions import transactions_from_firestore
from receipts_ai.models.transaction import ReceiptItem, Transaction
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

TransactionLoader = Callable[[], Sequence[Transaction]]

RECEIPT_ITEM_COLUMNS = (
    "Description",
    "Quantity",
    "Unit price",
    "Amount",
    "Discount amount",
    "Discount description",
    "Net amount",
    "Line type",
    "Category ID",
    "Taxonomy 1",
    "Taxonomy 2",
    "Taxonomy 3",
    "Taxonomy 4",
    "Taxonomy 5",
    "Taxonomy 6",
    "Taxonomy 7",
    "Taxonomy 8",
    "Taxonomy 9",
)


class ReceiptItemsScreen(Screen[None]):
    """Screen showing the receipt items for one transaction."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
    ]

    def __init__(self, transaction: Transaction) -> None:
        super().__init__()
        self._transaction = transaction

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(_format_transaction_header(self._transaction), id="receipt-header")
        yield DataTable(id="receipt-items")
        yield Footer()

    def on_mount(self) -> None:
        table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(*RECEIPT_ITEM_COLUMNS)

        receipt = self._transaction.receipt
        if receipt is None:
            return

        for item in receipt.items:
            table.add_row(*_receipt_item_row(item))


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
        height: 1;
        padding: 0 1;
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
        try:
            transactions = list(self._transaction_loader())
        except Exception as exc:  # pragma: no cover - exercised through app runtime.
            self.call_from_thread(self._show_error, exc)
            return

        self.call_from_thread(self._show_transactions, transactions)

    def _show_transactions(self, transactions: Sequence[Transaction]) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        table.clear()
        sorted_transactions = sorted(transactions, key=_transaction_sort_key)
        self._transactions_by_id = {
            transaction.id: transaction for transaction in sorted_transactions
        }
        for transaction in sorted_transactions:
            table.add_row(
                transaction.transaction_date.isoformat(),
                transaction.payee or "",
                transaction.description or "",
                transaction.ingestion_filename or "",
                _format_receipt_indicator(transaction),
                _format_amount(transaction.amount, transaction.currency),
                key=transaction.id,
            )

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

        self.push_screen(ReceiptItemsScreen(transaction))


def _transaction_sort_key(transaction: Transaction) -> tuple[object, ...]:
    try:
        amount = Decimal(transaction.amount)
    except InvalidOperation:
        return (transaction.transaction_date, 1, transaction.amount)

    return (transaction.transaction_date, 0, amount)


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


def _format_transaction_header(transaction: Transaction) -> str:
    return " | ".join(
        (
            transaction.transaction_date.isoformat(),
            transaction.payee or "",
            _format_amount(transaction.amount, transaction.currency),
        )
    )


def _receipt_item_row(item: ReceiptItem) -> list[str]:
    return [
        item.description,
        _format_optional(item.quantity),
        _format_optional(item.unit_price),
        item.amount,
        _format_optional(item.discount_amount),
        _format_optional(item.discount_description),
        item.net_amount,
        _format_optional(item.line_type.value if item.line_type is not None else None),
        _format_optional(item.category_id),
        _format_optional(item.taxonomy1),
        _format_optional(item.taxonomy2),
        _format_optional(item.taxonomy3),
        _format_optional(item.taxonomy4),
        _format_optional(item.taxonomy5),
        _format_optional(item.taxonomy6),
        _format_optional(item.taxonomy7),
        _format_optional(item.taxonomy8),
        _format_optional(item.taxonomy9),
    ]


def _format_optional(value: object | None) -> str:
    return "" if value is None else str(value)


def main() -> None:
    ReceiptsAIApp().run()
