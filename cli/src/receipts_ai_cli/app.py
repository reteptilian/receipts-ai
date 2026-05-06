from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from typing import cast

from receipts_ai.firestore_transactions import transactions_from_firestore
from receipts_ai.models.transaction import Transaction
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

TransactionLoader = Callable[[], Sequence[Transaction]]


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

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Loading transactions...", id="status")
        yield DataTable(id="transactions")
        yield Footer()

    def on_mount(self) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Date", "Payee", "Description", "Ingestion file", "Amount")
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
        for transaction in transactions:
            table.add_row(
                transaction.transaction_date.isoformat(),
                transaction.payee or "",
                transaction.description or "",
                transaction.ingestion_filename or "",
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


def _format_amount(amount: str, currency: str) -> str:
    try:
        decimal_amount = Decimal(amount)
    except InvalidOperation:
        formatted_amount = amount
    else:
        formatted_amount = f"{decimal_amount:,.2f}"

    return f"{formatted_amount} {currency}"


def main() -> None:
    ReceiptsAIApp().run()
