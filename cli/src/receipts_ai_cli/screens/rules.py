from __future__ import annotations

# pyright: reportUnknownMemberType=false
import importlib
from collections.abc import Sequence
from typing import Any, cast

from receipts_ai.automation_rules import AutomationRule
from rich.markup import escape
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Header, Static


def _app_module() -> Any:
    return importlib.import_module("receipts_ai_cli.app")


class RulesScreen(Screen[None]):
    """Screen for reviewing and deleting automation rules."""

    BINDINGS = [
        ("d", "delete_rule", "Delete"),
        ("escape", "exit", "Back"),
        ("q", "exit", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._rules_by_id: dict[str, AutomationRule] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Loading rules...", id="rules-status")
        yield DataTable(id="rules")
        yield Static("d Delete rule | Esc/q Back", id="rules-controls")

    def on_mount(self) -> None:
        table = cast(DataTable[str], self.query_one("#rules", DataTable))
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Name", "Scope", "When", "Then", "Status")
        table.focus()
        self.run_worker(self._load_rules, thread=True, name="load-rules")

    def action_exit(self) -> None:
        self.dismiss()

    def action_delete_rule(self) -> None:
        rule_id = self._current_rule_id()
        if rule_id is None:
            return
        self.query_one("#rules-status", Static).update("Deleting rule...")
        self.run_worker(lambda: self._delete_rule(rule_id), thread=True, name=f"delete-{rule_id}")

    def _load_rules(self) -> None:
        try:
            rules = _app_module().automation_rules_from_firestore()
            category_options = tuple(_app_module().load_budget_category_options())
            category_ids = {option.category_id for option in category_options}
            invalid_reasons = {
                rule.id: _app_module().rule_invalid_reason(rule)
                for rule in rules
                if rule.status != "deleted"
            }
        except Exception as exc:
            self.app.call_from_thread(self._show_error, exc)
            return

        self.app.call_from_thread(self._show_rules, rules, category_ids, invalid_reasons)

    def _delete_rule(self, rule_id: str) -> None:
        try:
            _app_module().delete_automation_rule(rule_id)
        except Exception as exc:
            self.app.call_from_thread(self._show_error, exc)
            return
        self.app.call_from_thread(self._show_status, "Rule deleted")
        self._load_rules()

    def _show_rules(
        self,
        rules: Sequence[AutomationRule],
        category_ids: set[str],
        invalid_reasons: dict[str, str | None],
    ) -> None:
        table = cast(DataTable[str], self.query_one("#rules", DataTable))
        table.clear()
        visible_rules = [rule for rule in rules if rule.status != "deleted"]
        self._rules_by_id = {rule.id: rule for rule in visible_rules}
        for rule in visible_rules:
            invalid_reason = invalid_reasons.get(rule.id)
            status = "Invalid" if invalid_reason else "Enabled" if rule.enabled else "Disabled"
            if invalid_reason:
                status = f"Invalid: {invalid_reason}"
            table.add_row(
                escape(rule.name),
                rule.scope.value,
                escape(_format_conditions(rule)),
                escape(_format_actions(rule, category_ids)),
                escape(status),
                key=rule.id,
            )
        self._show_status(
            f"{len(visible_rules)} rule{'s' if len(visible_rules) != 1 else ''}"
            if visible_rules
            else "No automation rules"
        )

    def _show_status(self, message: str) -> None:
        status = self.query_one("#rules-status", Static)
        status.remove_class("error")
        status.update(message)

    def _show_error(self, exc: Exception) -> None:
        status = self.query_one("#rules-status", Static)
        status.add_class("error")
        status.update(f"Unable to load rules: {exc}")

    def _current_rule_id(self) -> str | None:
        table = cast(DataTable[str], self.query_one("#rules", DataTable))
        coordinate = table.cursor_coordinate
        if not table.is_valid_coordinate(coordinate):
            return None
        return str(table.coordinate_to_cell_key(coordinate).row_key.value)


def _format_conditions(rule: AutomationRule) -> str:
    return " and ".join(
        f"{condition.field.value} {condition.operator.value} {condition.value!r}"
        for condition in rule.conditions
    )


def _format_actions(rule: AutomationRule, category_ids: set[str]) -> str:
    parts: list[str] = []
    for action in rule.actions:
        if action.value is not None:
            parts.append(f"{action.type.value} {action.value!r}")
        elif action.category_id is not None:
            suffix = "" if action.category_id in category_ids else " (missing)"
            parts.append(f"{action.type.value} {action.category_id}{suffix}")
        elif action.allocations is not None:
            parts.append(
                f"{action.type.value} "
                + ", ".join(
                    f"{allocation.category_id} {allocation.percent}%"
                    for allocation in action.allocations
                )
            )
        else:
            parts.append(action.type.value)
    return "; ".join(parts)
