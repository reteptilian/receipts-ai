from __future__ import annotations

# pyright: reportPrivateUsage=false
import importlib
from collections.abc import Sequence
from typing import Any, cast

from receipts_ai.firestore_transactions import transactions_from_firestore
from receipts_ai.models.transaction import Transaction, TransactionUserOverrides
from rich.markup import escape
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static
from textual.widgets.data_table import RowKey

from receipts_ai_cli.screens.rules import RulesScreen
from receipts_ai_cli.screens.transaction_review import TransactionReviewScreen
from receipts_ai_cli.transaction_helpers import (
    TRANSACTION_TABLE_COLUMNS,
    TransactionLoader,
    _display_transactions,
    _effective_transaction_amount,
    _effective_transaction_date,
    _effective_transaction_payee,
    _effective_transaction_reviewed,
    _effective_transaction_taxonomy,
    _format_amount,
    _format_receipt_indicator,
    _format_transaction_category,
    _format_transaction_status,
    _is_bank_statement_transaction,
    _is_receipt_based_transaction,
    _receipt_transactions_by_display_id,
    _transaction_sort_key,
    _transaction_table_column_widths,
)


def _app_module() -> Any:
    return importlib.import_module("receipts_ai_cli.app")


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

    #transaction-taxonomy {
        height: 1;
        padding: 0 1;
    }

    #transaction-taxonomy:focus {
        background: $accent;
        color: $text;
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

    #category-choice-dialog {
        padding: 1 2;
        width: 72;
        height: 24;
        border: thick $primary;
        background: $surface;
    }

    #category-choice-list {
        height: 1fr;
        margin: 1 0;
    }

    #category-choice-help {
        color: $text-muted;
        text-align: center;
        width: 1fr;
    }

    #taxonomy-choice-dialog {
        padding: 1 2;
        width: 96;
        height: 28;
        border: thick $primary;
        background: $surface;
    }

    #taxonomy-choice-input {
        margin: 1 0 0 0;
    }

    #taxonomy-choice-status {
        height: 1;
        color: $text-muted;
    }

    #taxonomy-choice-list {
        height: 1fr;
        margin: 1 0;
    }

    #taxonomy-choice-help {
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

    #rule-suggestion-dialog {
        padding: 1 2;
        width: 76;
        height: auto;
        border: thick $primary;
        background: $surface;
    }

    #rule-suggestion-prompt {
        margin: 1 0;
    }

    #rule-suggestion-buttons {
        height: 3;
    }

    #rules-status {
        dock: top;
        height: 1;
        padding: 0 1;
    }

    #rules-status.error {
        color: $error;
    }

    #rules-controls {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
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
        ("r", "toggle_transaction_reviewed", "Toggle reviewed"),
        ("R", "open_rules", "Rules"),
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
        self._category_display_by_id: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Loading transactions...", id="status")
        yield DataTable(id="transactions")
        yield Footer()

    def on_mount(self) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        table.cursor_type = "row"
        table.zebra_stripes = True
        column_widths = _transaction_table_column_widths(self.size.width)
        for column in TRANSACTION_TABLE_COLUMNS:
            table.add_column(
                column.label,
                key=column.key,
                width=column_widths[column.key],
            )
        table.focus()
        self.run_worker(self._load_transactions, thread=True, name="load-transactions")

    def on_resize(self, event: events.Resize) -> None:
        self._resize_transaction_columns(event.size.width)

    def _load_transactions(self) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        cursor_coordinate = table.cursor_coordinate
        cursor_row_key = None
        if table.is_valid_coordinate(cursor_coordinate):
            cursor_row_key = table.coordinate_to_cell_key(cursor_coordinate).row_key

        try:
            transactions = list(self._transaction_loader())
            transactions = self._transactions_with_rules_applied(transactions)
            category_options = tuple(_app_module().load_budget_category_options())
        except Exception as exc:  # pragma: no cover - exercised through app runtime.
            self.call_from_thread(self._show_error, exc)
            return

        self.call_from_thread(
            self._show_transactions, transactions, cursor_row_key, category_options
        )

    def _transactions_with_rules_applied(
        self, transactions: Sequence[Transaction]
    ) -> list[Transaction]:
        try:
            rules = tuple(_app_module().automation_rules_from_firestore())
        except Exception:
            return list(transactions)
        if not rules:
            return list(transactions)

        updated_transactions: list[Transaction] = []
        for transaction in transactions:
            updated_transaction = transaction.model_copy(deep=True)
            if updated_transaction.reviewed is not True:
                _app_module().apply_automation_rules(updated_transaction, rules)
            updated_transactions.append(updated_transaction)
        return updated_transactions

    def _show_transactions(
        self,
        transactions: Sequence[Transaction],
        cursor_row_key: RowKey | None = None,
        category_options: Sequence[Any] | None = None,
    ) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        table.clear()
        if category_options is not None:
            self._category_display_by_id = {
                option.category_id: option.path_text for option in category_options
            }
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
                _effective_transaction_taxonomy(transaction),
                _format_transaction_status(transaction),
                _format_transaction_category(transaction, self._category_display),
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
            TransactionReviewScreen(
                transaction,
                receipt_transaction=receipt_transaction,
                category_options=tuple(_app_module().load_budget_category_options()),
            ),
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

    def action_open_rules(self) -> None:
        self.push_screen(RulesScreen())

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

    def action_toggle_transaction_reviewed(self) -> None:
        transaction_id = self._current_transaction_id()
        if transaction_id is None:
            return
        transaction = self._transactions_by_id.get(transaction_id)
        if transaction is None:
            return

        reviewed = not _effective_transaction_reviewed(transaction)
        receipt_transaction = self._receipt_transactions_by_display_id.get(transaction_id)
        receipt_transaction_id = (
            receipt_transaction.id
            if receipt_transaction is not None and receipt_transaction is not transaction
            else None
        )
        transaction.reviewed = reviewed
        if receipt_transaction_id is not None and receipt_transaction is not None:
            receipt_transaction.reviewed = reviewed
        self._refresh_visible_transactions()
        status = "Reviewed" if reviewed else "Unreviewed"
        self._show_status(f"Marked transaction as {status}. Saving...")
        self.run_worker(
            lambda: self._persist_transaction_reviewed(
                transaction.id,
                transaction.user_overrides or TransactionUserOverrides(),
                reviewed,
                receipt_transaction_id,
            ),
            thread=True,
            name=f"toggle-review-{transaction.id}",
        )

    def _persist_transaction_link(self, bank_statement_id: str, receipt_based_id: str) -> None:
        try:
            _app_module().link_bank_statement_transaction_to_receipt(
                bank_statement_id, receipt_based_id
            )
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
            _app_module().unlink_bank_statement_transaction_from_receipt(bank_statement_id)
            self._selected_transaction_ids.clear()
            self.call_from_thread(self._show_status, "Transaction unlinked")
            self.call_from_thread(
                lambda: self.run_worker(
                    self._load_transactions, thread=True, name="load-transactions"
                )
            )
        except Exception as exc:
            self.call_from_thread(self._show_status, f"Failed to unlink: {exc}", True)

    def _persist_transaction_reviewed(
        self,
        transaction_id: str,
        transaction_user_overrides: TransactionUserOverrides,
        reviewed: bool,
        receipt_transaction_id: str | None,
    ) -> None:
        try:
            _app_module().save_transaction_review_edits(
                transaction_id,
                transaction_user_overrides,
                reviewed=reviewed,
                receipt_transaction_id=receipt_transaction_id,
            )
            status = "Reviewed" if reviewed else "Unreviewed"
            self.call_from_thread(self._show_status, f"Transaction marked {status}")
            self.call_from_thread(
                lambda: self.run_worker(
                    self._load_transactions, thread=True, name="load-transactions"
                )
            )
        except Exception as exc:
            self.call_from_thread(self._show_status, f"Failed to toggle reviewed: {exc}", True)
            self.call_from_thread(
                lambda: self.run_worker(
                    self._load_transactions, thread=True, name="load-transactions"
                )
            )

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

    def _resize_transaction_columns(self, terminal_width: int | None = None) -> None:
        table = cast(DataTable[str], self.query_one("#transactions", DataTable))
        if len(table.ordered_columns) != len(TRANSACTION_TABLE_COLUMNS):
            return
        width = terminal_width or table.size.width or self.size.width
        column_widths = _transaction_table_column_widths(width)
        for table_column, column_config in zip(
            table.ordered_columns, TRANSACTION_TABLE_COLUMNS, strict=True
        ):
            table_column.auto_width = False
            table_column.width = column_widths[column_config.key]
        table.refresh(layout=True)

    def _show_status(self, message: str, error: bool = False) -> None:
        status = self.query_one("#status", Static)
        if error:
            status.add_class("error")
        else:
            status.remove_class("error")
        status.update(escape(message))

    def _category_display(self, category_id: str | None) -> str:
        if category_id is None:
            return ""
        return self._category_display_by_id.get(category_id, category_id)
