from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import importlib
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, cast

from pydantic import ValidationError
from receipts_ai.models.transaction import (
    CategoryAllocation,
    ReceiptItem,
    ReceiptItemUserOverrides,
    Transaction,
    TransactionUserOverrides,
    UserCategoryAllocation,
)
from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.coordinate import Coordinate
from textual.screen import Screen
from textual.widgets import DataTable, Header, Input, Label, Static

from receipts_ai_cli.screens.modals import CategoryChoiceScreen, CellEditScreen
from receipts_ai_cli.transaction_helpers import (
    RECEIPT_ITEM_COLUMNS,
    _decimal_amount,
    _effective_category_allocations,
    _effective_receipt_item_value,
    _effective_transaction_amount,
    _effective_transaction_date,
    _effective_transaction_description,
    _effective_transaction_payee,
    _field_edit_error_message,
    _has_receipt_items,
    _header_input_label,
    _parse_optional_date,
    _parse_optional_text,
    _parse_required_decimal_text,
    _parse_required_text,
    _receipt_item_row,
)


def _app_module() -> Any:
    return importlib.import_module("receipts_ai_cli.app")


class TransactionReviewScreen(Screen[None]):
    """Screen for drafting and saving transaction review edits."""

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
        category_choices: Sequence[str] | None = None,
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
        self._category_choices = (
            tuple(category_choices)
            if category_choices is not None
            else _app_module().load_budget_category_choices()
        )

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
                lambda: _app_module()._open_file_in_external_viewer(ingestion_file_url),
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

        if table.id == "category-allocations" and coordinate.column == 0:

            def check_category_choice(new_value: str | None) -> None:
                if new_value is not None:
                    self._commit_category_allocation_cell_edit(coordinate, new_value)

            self.app.push_screen(
                CategoryChoiceScreen(str(value), self._category_choices),
                check_category_choice,
            )
            return

        def check_edit(new_value: str | None) -> None:
            if new_value is not None:
                if table.id == "category-allocations":
                    self._commit_category_allocation_cell_edit(coordinate, new_value)
                else:
                    self._commit_receipt_item_cell_edit(coordinate, new_value)

        self.app.push_screen(CellEditScreen(column_label, str(value)), check_edit)

    def action_add_category_allocation(self) -> None:
        allocations = _effective_category_allocations(self._transaction)
        category_id = (
            "Miscellaneous > Uncategorized"
            if "Miscellaneous > Uncategorized" in self._category_choices
            else self._category_choices[0]
        )
        allocations.append(UserCategoryAllocation(category_id=category_id, amount="0.00"))
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
        update = {
            "category_id": allocation.category_id,
            "amount": allocation.amount,
        }
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

            _app_module().save_transaction_review_edits(
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
