from __future__ import annotations

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option


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


class CategoryChoiceScreen(ModalScreen[str]):
    """Screen for choosing a budget category."""

    def __init__(self, value: str, choices: Sequence[str]) -> None:
        super().__init__()
        self._value = value
        self._choices = tuple(choices)

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="category-choice-dialog"):
                yield Label("Choose Category ID")
                yield OptionList(
                    *(Option(choice) for choice in self._choices),
                    id="category-choice-list",
                )
                with Horizontal(id="category-choice-buttons"):
                    yield Static("Enter to Choose / Esc to Cancel", id="category-choice-help")

    def on_mount(self) -> None:
        choice_list = self.query_one("#category-choice-list", OptionList)
        if self._value in self._choices:
            choice_list.highlighted = self._choices.index(self._value)
        choice_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_index < len(self._choices):
            self.dismiss(self._choices[event.option_index])

    def action_cancel(self) -> None:
        self.dismiss()

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
