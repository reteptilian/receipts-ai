from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import importlib
import logging
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, cast

from pydantic import ValidationError
from receipts_ai.budget_categories import BudgetCategoryChoice
from receipts_ai.models.transaction import (
    CategoryAllocation,
    ReceiptItem,
    ReceiptItemUserOverrides,
    Transaction,
    TransactionUserOverrides,
    UserCategoryAllocation,
)
from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.coordinate import Coordinate
from textual.screen import Screen
from textual.widgets import DataTable, Header, Input, Label, Static

from receipts_ai_cli.screens.modals import (
    CategoryChoiceScreen,
    CellEditScreen,
    RuleSuggestionScreen,
    TaxonomyChoiceScreen,
)
from receipts_ai_cli.transaction_helpers import (
    RECEIPT_ITEM_COLUMNS,
    _decimal_amount,
    _effective_category_allocations,
    _effective_receipt_item_value,
    _effective_transaction_amount,
    _effective_transaction_date,
    _effective_transaction_description,
    _effective_transaction_payee,
    _effective_transaction_reviewed,
    _effective_transaction_taxonomy,
    _field_edit_error_message,
    _has_receipt_items,
    _header_input_label,
    _is_transaction_pair,
    _parse_optional_date,
    _parse_optional_text,
    _parse_required_decimal_text,
    _parse_required_text,
    _receipt_item_row,
)


class TransactionTaxonomyStatic(Static, can_focus=True):
    """Focusable transaction taxonomy field."""


_HEADER_INPUT_IDS = frozenset(
    {
        "receipt-date",
        "receipt-payee",
        "receipt-description",
        "receipt-amount",
    }
)

LOGGER = logging.getLogger(__name__)


def _app_module() -> Any:
    return importlib.import_module("receipts_ai_cli.app")


class TransactionReviewScreen(Screen[None]):
    """Screen for drafting and saving transaction review edits."""

    BINDINGS = [
        ("e", "edit_cell", "Edit cell"),
        ("t", "edit_transaction_taxonomy", "Edit taxonomy"),
        ("a", "add_category_allocation", "Add allocation"),
        ("d", "delete_category_allocation", "Delete allocation"),
        ("r", "toggle_reviewed", "Toggle reviewed"),
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
        category_options: Sequence[BudgetCategoryChoice] | None = None,
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
        self._pending_rule_suggestions: tuple[Any, ...] = ()
        self._canceling_header_input_ids: set[str] = set()
        active_category_options = (
            _app_module().load_budget_category_options()
            if category_options is None and category_choices is None
            else category_options or ()
        )
        self._category_options = tuple(active_category_options)
        self._category_display_by_id = {
            option.category_id: option.path_text for option in self._category_options
        }
        self._category_choices = (
            tuple(category_choices)
            if category_choices is not None
            else tuple(option.category_id for option in self._category_options)
        )
        self._category_choice_labels = {
            choice: self._category_display_by_id.get(choice, choice)
            for choice in self._category_choices
        }

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
        yield Static("", id="review-state")
        yield TransactionTaxonomyStatic("", id="transaction-taxonomy")
        yield Static("", id="transaction-mcc-description")
        yield Static("", id="receipt-edit-status")
        yield Static("Category Allocations", id="category-allocations-title")
        yield DataTable(id="category-allocations")
        yield Static("Receipt Items", id="receipt-items-title")
        yield DataTable(id="receipt-items")
        controls = (
            "r Toggle reviewed | t Edit taxonomy | s Save and Exit | "
            "e Edit cell | Esc/q Exit Without Saving"
        )
        if not self._has_receipt_items():
            controls = (
                "r Toggle reviewed | t Edit taxonomy | s Save and Exit | "
                "a Add allocation | d Delete allocation | e Edit cell | Esc/q Exit Without Saving"
            )
        yield Static(controls, id="review-controls")

    def on_mount(self) -> None:
        allocations_table = cast(DataTable[str], self.query_one("#category-allocations", DataTable))
        allocations_table.cursor_type = "cell"
        allocations_table.zebra_stripes = True
        allocations_table.add_columns("Category", "Amount")
        self._refresh_category_allocations_table()
        self._refresh_review_state()
        self._refresh_transaction_taxonomy()
        self._refresh_transaction_mcc_description()
        if self._has_receipt_items():
            self.query_one("#category-allocations-title", Static).add_class("hidden")
            allocations_table.add_class("hidden")

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
            table.add_row(*_receipt_item_row(item, self._category_display))

        table.focus()

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        data_table = cast(DataTable[str], event.data_table)
        if data_table.id in {"category-allocations", "receipt-items"}:
            self.action_edit_cell()

    def on_click(self, event: events.Click) -> None:
        if event.widget is self.query_one("#transaction-taxonomy", TransactionTaxonomyStatic):
            self.action_edit_transaction_taxonomy()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape" and isinstance(self.focused, Input):
            input_widget = self.focused
            if input_widget.id in _HEADER_INPUT_IDS:
                self._cancel_header_input_edit(input_widget)
                event.stop()
                event.prevent_default()
        if event.key in {"enter", "space"} and self.focused is self.query_one(
            "#transaction-taxonomy", TransactionTaxonomyStatic
        ):
            self.action_edit_transaction_taxonomy()
            event.stop()
            event.prevent_default()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._commit_header_input(event.input)

    def on_input_blurred(self, event: Input.Blurred) -> None:
        if event.input.id in _HEADER_INPUT_IDS:
            if event.input.id in self._canceling_header_input_ids:
                self._canceling_header_input_ids.discard(event.input.id)
                return
            self._commit_header_input(event.input)

    def action_edit_cell(self) -> None:
        if self.focused is self.query_one("#transaction-taxonomy", TransactionTaxonomyStatic):
            self.action_edit_transaction_taxonomy()
            return

        table = self._focused_edit_table()
        if table is None:
            return
        coordinate = table.cursor_coordinate
        if not table.is_valid_coordinate(coordinate):
            return

        column_label = (
            ("Category", "Amount")[coordinate.column]
            if table.id == "category-allocations"
            else RECEIPT_ITEM_COLUMNS[coordinate.column].label
        )
        value = table.get_cell_at(coordinate)

        if (
            table.id == "receipt-items"
            and RECEIPT_ITEM_COLUMNS[coordinate.column].field_name == "taxonomy"
        ):

            def check_taxonomy_choice(new_value: str | None) -> None:
                self._commit_receipt_item_cell_edit(coordinate, new_value or "")

            self.app.push_screen(TaxonomyChoiceScreen(str(value)), check_taxonomy_choice)
            return

        if (table.id == "category-allocations" and coordinate.column == 0) or (
            table.id == "receipt-items"
            and RECEIPT_ITEM_COLUMNS[coordinate.column].field_name == "category_id"
        ):

            def check_category_choice(new_value: str | None) -> None:
                if new_value is not None:
                    if table.id == "category-allocations":
                        self._commit_category_allocation_cell_edit(coordinate, new_value)
                    else:
                        self._commit_receipt_item_cell_edit(coordinate, new_value)

            self.app.push_screen(
                CategoryChoiceScreen(
                    self._category_id_from_display(str(value)),
                    self._category_choices,
                    self._category_choice_labels,
                ),
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

    def action_edit_transaction_taxonomy(self) -> None:
        def check_taxonomy_choice(new_value: str | None) -> None:
            try:
                self._set_transaction_override(
                    "taxonomy",
                    "" if new_value is None else _parse_optional_text(new_value),
                )
            except (ValueError, ValidationError) as exc:
                self._show_edit_error(_field_edit_error_message("Taxonomy", exc))
                return

            self._refresh_transaction_taxonomy()
            self._mark_dirty("Draft transaction taxonomy edit")

        self.app.push_screen(
            TaxonomyChoiceScreen(_effective_transaction_taxonomy(self._transaction)),
            check_taxonomy_choice,
        )

    def action_add_category_allocation(self) -> None:
        if self._has_receipt_items():
            return
        allocations = _effective_category_allocations(self._transaction)
        category_id = (
            "miscellaneous.uncategorized"
            if "miscellaneous.uncategorized" in self._category_choices
            else self._category_choices[0]
        )
        allocations.append(UserCategoryAllocation(category_id=category_id, amount="0.00"))
        self._set_category_allocation_overrides(allocations)
        self._refresh_category_allocations_table()
        self._mark_dirty("Added category allocation")

    def action_delete_category_allocation(self) -> None:
        if self._has_receipt_items():
            return
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
        LOGGER.debug(
            "Save and Exit requested for transaction %s: dirty=%s reviewed=%s "
            "effective_reviewed=%s has_receipt_items=%s source_receipt_transaction_id=%s",
            self._source_transaction.id,
            self._dirty,
            self._transaction.reviewed,
            _effective_transaction_reviewed(self._transaction),
            self._has_receipt_items(),
            (
                self._source_receipt_transaction.id
                if self._source_receipt_transaction is not None
                else None
            ),
        )
        try:
            self._validate_save()
        except ValueError as exc:
            LOGGER.debug(
                "Save validation blocked rule suggestions for transaction %s: %s",
                self._source_transaction.id,
                exc,
            )
            self._show_edit_error(str(exc))
            return

        self._pending_rule_suggestions = self._rule_suggestions_for_save()
        LOGGER.debug(
            "Prepared %d pending rule suggestion(s) for transaction %s before persist",
            len(self._pending_rule_suggestions),
            self._source_transaction.id,
        )
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
                    parsed_value = _parse_optional_date(value)
                    if parsed_value == _effective_transaction_date(self._transaction):
                        return
                    self._set_transaction_override("transaction_date", parsed_value)
                case "receipt-payee":
                    parsed_value = _parse_optional_text(value)
                    if parsed_value == _effective_transaction_payee(self._transaction):
                        return
                    self._set_transaction_override("payee", parsed_value)
                case "receipt-description":
                    parsed_value = _parse_optional_text(value)
                    if parsed_value == _effective_transaction_description(self._transaction):
                        return
                    self._set_transaction_override("description", parsed_value)
                case "receipt-amount":
                    parsed_value = _parse_optional_text(value)
                    if parsed_value == _effective_transaction_amount(self._transaction):
                        return
                    self._set_transaction_override("amount", parsed_value)
                case _:
                    return
        except (ValueError, ValidationError) as exc:
            self._show_edit_error(_field_edit_error_message(_header_input_label(input_widget), exc))
            return

        self._mark_dirty("Draft transaction edit")

    def _cancel_header_input_edit(self, input_widget: Input) -> None:
        if input_widget.id is not None:
            self._canceling_header_input_ids.add(input_widget.id)
        input_widget.value = self._effective_header_input_value(input_widget)
        table = self._focused_edit_table()
        if table is not None:
            table.focus()

    def _effective_header_input_value(self, input_widget: Input) -> str:
        match input_widget.id:
            case "receipt-date":
                return _effective_transaction_date(self._transaction).isoformat()
            case "receipt-payee":
                return _effective_transaction_payee(self._transaction)
            case "receipt-description":
                return _effective_transaction_description(self._transaction)
            case "receipt-amount":
                return _effective_transaction_amount(self._transaction)
            case _:
                return input_widget.value

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
                    update["category_id"] = _parse_required_text(raw_value.strip(), "Category")
                case 1:
                    update["amount"] = _parse_required_decimal_text(raw_value.strip(), "Amount")
                case _:
                    return
            allocations[coordinate.row] = UserCategoryAllocation.model_validate(update)
        except (ValueError, ValidationError) as exc:
            self._show_edit_error(
                _field_edit_error_message(
                    ("Category", "Category allocation amount")[coordinate.column], exc
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
            if column.field_name == "taxonomy" and not raw_value.strip():
                parsed_value = ""
            else:
                parsed_value = column.parser(raw_value.strip())
            self._set_receipt_item_override(item, column.field_name, parsed_value)
        except (ValueError, ValidationError) as exc:
            self._show_edit_error(_field_edit_error_message(column.label, exc))
            return

        table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        table.update_cell_at(
            coordinate,
            _receipt_item_row(item, self._category_display)[coordinate.column],
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
                reviewed=self._transaction.reviewed,
                receipt_transaction_id=receipt_transaction_id,
                receipt_items=receipt_items,
            )
            self._source_transaction.user_overrides = self._transaction.user_overrides
            self._source_transaction.reviewed = self._transaction.reviewed
            if self._source_receipt_transaction is not None and self._receipt_transaction:
                self._source_receipt_transaction.receipt = self._receipt_transaction.receipt
                self._source_receipt_transaction.reviewed = self._transaction.reviewed
            elif self._receipt_transaction is not None:
                self._source_transaction.receipt = self._receipt_transaction.receipt
            self._dirty = False
            self.app.call_from_thread(self._finish_persisted_review)
        except Exception as exc:
            self.app.call_from_thread(self._show_edit_error, f"Failed to persist: {exc}")

    def _finish_persisted_review(self) -> None:
        if not self._pending_rule_suggestions:
            LOGGER.debug(
                "No pending rule suggestions after saving transaction %s; dismissing review screen",
                self._source_transaction.id,
            )
            self.dismiss()
            return
        LOGGER.debug(
            "Prompting %d rule suggestion(s) after saving transaction %s",
            len(self._pending_rule_suggestions),
            self._source_transaction.id,
        )
        self._show_edit_status("Review saved. Checking automation rule suggestions...")
        self._prompt_rule_suggestion(0)

    def _prompt_rule_suggestion(self, index: int) -> None:
        if index >= len(self._pending_rule_suggestions):
            LOGGER.debug(
                "Finished prompting rule suggestions for transaction %s",
                self._source_transaction.id,
            )
            self.dismiss()
            return
        suggestion = self._pending_rule_suggestions[index]
        LOGGER.debug(
            "Prompting rule suggestion %d/%d for transaction %s: key=%s prompt=%r",
            index + 1,
            len(self._pending_rule_suggestions),
            self._source_transaction.id,
            suggestion.key,
            suggestion.prompt,
        )

        def on_decision(create_rule: bool | None) -> None:
            LOGGER.debug(
                "Rule suggestion decision for transaction %s: key=%s create_rule=%s",
                self._source_transaction.id,
                suggestion.key,
                create_rule,
            )
            if create_rule:
                self._show_edit_status("Saving automation rule...")
                self.run_worker(
                    lambda: self._save_rule_suggestion(suggestion, index),
                    thread=True,
                    name=f"save-rule-{suggestion.key}",
                )
            else:
                self._prompt_rule_suggestion(index + 1)

        self.app.push_screen(RuleSuggestionScreen(suggestion.prompt), on_decision)

    def _save_rule_suggestion(self, suggestion: Any, index: int) -> None:
        try:
            LOGGER.debug(
                "Saving automation rule suggestion for transaction %s: key=%s",
                self._source_transaction.id,
                suggestion.key,
            )
            _app_module().save_automation_rule(suggestion.rule)
        except Exception as exc:
            LOGGER.exception(
                "Failed to save automation rule suggestion for transaction %s: key=%s",
                self._source_transaction.id,
                suggestion.key,
            )
            self.app.call_from_thread(
                self._show_edit_error,
                f"Review saved, but failed to save automation rule: {exc}",
            )
            return
        LOGGER.debug(
            "Saved automation rule suggestion for transaction %s: key=%s",
            self._source_transaction.id,
            suggestion.key,
        )
        self.app.call_from_thread(self._prompt_rule_suggestion, index + 1)

    def action_toggle_reviewed(self) -> None:
        self._transaction.reviewed = not _effective_transaction_reviewed(self._transaction)
        if self._source_receipt_transaction is not None and self._receipt_transaction is not None:
            self._receipt_transaction.reviewed = self._transaction.reviewed
        self._refresh_review_state()
        status = "Reviewed" if self._transaction.reviewed else "Unreviewed"
        self._mark_dirty(f"Marked transaction as {status}")

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
            table.add_row(self._category_display(allocation.category_id), allocation.amount)

    def _refresh_review_state(self) -> None:
        review_state = self.query_one("#review-state", Static)
        reviewed_text = (
            "Reviewed" if _effective_transaction_reviewed(self._transaction) else "Unreviewed"
        )
        pair_text = "Pair" if _is_transaction_pair(self._transaction) else "Single record"
        review_state.update(f"Review: {reviewed_text} | Type: {pair_text}")

    def _refresh_transaction_taxonomy(self) -> None:
        taxonomy = self.query_one("#transaction-taxonomy", Static)
        value = _effective_transaction_taxonomy(self._transaction) or "Unassigned"
        taxonomy.update(f"Taxonomy: {value}")

    def _refresh_transaction_mcc_description(self) -> None:
        mcc_description = self.query_one("#transaction-mcc-description", Static)
        value = self._transaction.mcc_description
        if value is None or not value.strip():
            mcc_description.add_class("hidden")
            mcc_description.update("")
            return
        mcc_description.remove_class("hidden")
        mcc_description.update(f"MCC description: {value.strip()}")

    def _validate_save(self) -> None:
        transaction_amount = _decimal_amount(
            _effective_transaction_amount(self._transaction), "Transaction amount"
        )
        if not self._has_receipt_items():
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
                    str(_effective_receipt_item_value(item, "net_amount")),
                    "Receipt item net amount",
                )
                for item in self._receipt_transaction.receipt.items
            ),
            Decimal("0"),
        )
        expected_receipt_total = -transaction_amount
        if receipt_item_total != expected_receipt_total:
            raise ValueError(
                "Save blocked: receipt item net amounts must add up to the sign-flipped "
                f"transaction amount. The receipt items total {receipt_item_total}, but the "
                f"expected total is {expected_receipt_total}."
            )

    def _rule_suggestions_for_save(self) -> tuple[Any, ...]:
        if not self._dirty:
            LOGGER.debug(
                "Skipping rule suggestions for transaction %s: screen is not dirty",
                self._source_transaction.id,
            )
            return ()
        if self._transaction.reviewed is not True:
            LOGGER.debug(
                "Skipping rule suggestions for transaction %s: transaction.reviewed is %s",
                self._source_transaction.id,
                self._transaction.reviewed,
            )
            return ()
        LOGGER.debug(
            "Generating rule suggestions for transaction %s: %s",
            self._source_transaction.id,
            self._rule_debug_summary(),
        )
        try:
            suggestions = tuple(
                _app_module().generate_rule_suggestions(
                    self._source_transaction,
                    self._transaction,
                    original_receipt_transaction=self._source_receipt_transaction,
                    edited_receipt_transaction=(
                        self._receipt_transaction
                        if self._source_receipt_transaction is not None
                        else None
                    ),
                )
            )
            LOGGER.debug(
                "Generated %d rule suggestion(s) for transaction %s: keys=%s",
                len(suggestions),
                self._source_transaction.id,
                [suggestion.key for suggestion in suggestions],
            )
            return suggestions
        except Exception as exc:
            LOGGER.exception(
                "Failed to generate rule suggestions for transaction %s",
                self._source_transaction.id,
            )
            self._show_edit_error(f"Skipping automation rule suggestions: {exc}")
            return ()

    def _rule_debug_summary(self) -> dict[str, object]:
        original_receipt_transaction = self._source_receipt_transaction
        edited_receipt_transaction = (
            self._receipt_transaction if original_receipt_transaction is not None else None
        )
        return {
            "dirty": self._dirty,
            "reviewed": self._transaction.reviewed,
            "effective_reviewed": _effective_transaction_reviewed(self._transaction),
            "has_receipt_items": self._has_receipt_items(),
            "original": self._transaction_rule_debug_summary(self._source_transaction),
            "edited": self._transaction_rule_debug_summary(self._transaction),
            "original_receipt": self._receipt_rule_debug_summary(original_receipt_transaction),
            "edited_receipt": self._receipt_rule_debug_summary(edited_receipt_transaction),
        }

    def _transaction_rule_debug_summary(self, transaction: Transaction | None) -> dict[str, object]:
        if transaction is None:
            return {}
        return {
            "id": transaction.id,
            "payee": _effective_transaction_payee(transaction),
            "amount": _effective_transaction_amount(transaction),
            "taxonomy": _effective_transaction_taxonomy(transaction),
            "reviewed": transaction.reviewed,
            "effective_reviewed": _effective_transaction_reviewed(transaction),
            "allocations": [
                {"category_id": allocation.category_id, "amount": allocation.amount}
                for allocation in _effective_category_allocations(transaction)
            ],
        }

    def _receipt_rule_debug_summary(self, transaction: Transaction | None) -> dict[str, object]:
        receipt = transaction.receipt if transaction is not None else None
        if transaction is None or receipt is None:
            return {}
        return {
            "id": transaction.id,
            "item_count": len(receipt.items),
            "items": [
                {
                    "index": index,
                    "description": _effective_receipt_item_value(item, "description"),
                    "category_id": _effective_receipt_item_value(item, "category_id"),
                    "taxonomy": _effective_receipt_item_value(item, "taxonomy"),
                    "net_amount": _effective_receipt_item_value(item, "net_amount"),
                }
                for index, item in enumerate(receipt.items)
            ],
        }

    def _focused_edit_table(self) -> DataTable[str] | None:
        allocations_table = cast(DataTable[str], self.query_one("#category-allocations", DataTable))
        receipt_items_table = cast(DataTable[str], self.query_one("#receipt-items", DataTable))
        if self._has_receipt_items():
            return receipt_items_table
        focused = self.focused
        if focused is allocations_table:
            return allocations_table
        if focused is receipt_items_table:
            return receipt_items_table
        return receipt_items_table if receipt_items_table.has_focus else allocations_table

    def _has_receipt_items(self) -> bool:
        receipt_transaction = self._receipt_transaction
        receipt = receipt_transaction.receipt if receipt_transaction is not None else None
        return bool(receipt is not None and receipt.items)

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

    def _category_display(self, category_id: str | None) -> str:
        if category_id is None:
            return ""
        return self._category_display_by_id.get(category_id, category_id)

    def _category_id_from_display(self, value: str) -> str:
        for category_id, display in self._category_display_by_id.items():
            if value == display:
                return category_id
        return value


ReceiptItemsScreen = TransactionReviewScreen
