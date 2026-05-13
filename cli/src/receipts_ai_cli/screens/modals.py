from __future__ import annotations

from collections.abc import Mapping, Sequence

from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from receipts_ai_cli.taxonomy_selection import taxonomy_selector_searcher


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

    def __init__(
        self, value: str, choices: Sequence[str], labels_by_choice: Mapping[str, str] | None = None
    ) -> None:
        super().__init__()
        self._value = value
        self._choices = tuple(choices)
        self._labels_by_choice = dict(labels_by_choice or {})

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="category-choice-dialog"):
                yield Label("Choose Category ID")
                yield OptionList(
                    *(
                        Option(self._labels_by_choice.get(choice, choice))
                        for choice in self._choices
                    ),
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


class TaxonomyChoiceScreen(ModalScreen[str | None]):
    """Screen for choosing or clearing a taxonomy override."""

    SEMANTIC_SEARCH_DEBOUNCE_SECONDS = 0.25

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+x", "clear", "Clear"),
        ("down", "focus_list", "Focus results"),
    ]

    def __init__(self, value: str) -> None:
        super().__init__()
        self._value = value
        self._searcher = taxonomy_selector_searcher()
        self._semantic_search_timer: Timer | None = None
        self._pending_semantic_query: str | None = None

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="taxonomy-choice-dialog"):
                yield Label("Choose Taxonomy")
                yield Input(placeholder="Type to search taxonomies", id="taxonomy-choice-input")
                yield Static("", id="taxonomy-choice-status")
                yield OptionList(id="taxonomy-choice-list")
                with Horizontal(id="taxonomy-choice-buttons"):
                    yield Static(
                        "Type to search, Enter to choose, Ctrl+X to clear, Esc to cancel",
                        id="taxonomy-choice-help",
                    )

    def on_mount(self) -> None:
        self._show_choices(self._searcher.choices, preferred_choice=self._value)
        self._update_status_for_empty_query()
        self.query_one("#taxonomy-choice-input", Input).focus()

    def on_unmount(self) -> None:
        if self._semantic_search_timer is not None:
            self._semantic_search_timer.stop()
            self._semantic_search_timer = None

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "taxonomy-choice-input":
            return

        query = event.value.strip()
        if not query:
            if self._semantic_search_timer is not None:
                self._semantic_search_timer.stop()
                self._semantic_search_timer = None
            self._pending_semantic_query = None
            self._show_choices(self._searcher.choices, preferred_choice=self._value)
            self._update_status_for_empty_query()
            return

        exact_matches = self._searcher.exact_matches(query)
        self._show_choices(exact_matches)
        self._pending_semantic_query = query
        self.query_one("#taxonomy-choice-status", Static).update(
            f'Query: "{query}" | {len(exact_matches)} text matches | waiting for semantic matches...'
        )
        self._schedule_semantic_search()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "taxonomy-choice-input":
            self._dismiss_highlighted_choice()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "taxonomy-choice-list":
            return
        self._dismiss_highlighted_choice()

    def action_cancel(self) -> None:
        self.dismiss()

    def action_clear(self) -> None:
        self.dismiss(None)

    def action_focus_list(self) -> None:
        self.query_one("#taxonomy-choice-list", OptionList).focus()

    def _dismiss_highlighted_choice(self) -> None:
        choice_list = self.query_one("#taxonomy-choice-list", OptionList)
        highlighted = choice_list.highlighted
        if highlighted is None or highlighted < 0 or highlighted >= choice_list.option_count:
            return
        self.dismiss(str(choice_list.get_option_at_index(highlighted).prompt))

    def _show_choices(
        self,
        choices: Sequence[str],
        *,
        preferred_choice: str | None = None,
    ) -> None:
        choice_list = self.query_one("#taxonomy-choice-list", OptionList)
        choice_list.set_options([Option(choice) for choice in choices])
        if not choices:
            choice_list.highlighted = None
            return
        if preferred_choice and preferred_choice in choices:
            choice_list.highlighted = choices.index(preferred_choice)
        else:
            choice_list.highlighted = 0
        choice_list.scroll_to_highlight()

    def _update_status_for_empty_query(self) -> None:
        self.query_one("#taxonomy-choice-status", Static).update(
            f"Showing all {len(self._searcher.choices)} taxonomies"
        )

    def _schedule_semantic_search(self) -> None:
        if self._semantic_search_timer is not None:
            self._semantic_search_timer.stop()
        self._semantic_search_timer = self.set_timer(
            self.SEMANTIC_SEARCH_DEBOUNCE_SECONDS,
            self._start_semantic_search_if_needed,
        )

    def _start_semantic_search_if_needed(self) -> None:
        self._semantic_search_timer = None
        query = self._pending_semantic_query
        if query is None or not query.strip():
            return

        self._pending_semantic_query = None
        active_query = self.query_one("#taxonomy-choice-input", Input).value.strip()
        if query != active_query:
            return

        semantic_matches = self._searcher.semantic_matches(query)
        active_query = self.query_one("#taxonomy-choice-input", Input).value.strip()
        if query != active_query:
            self._schedule_semantic_search()
            return

        combined_matches = self._searcher.combine_matches(
            query,
            semantic_matches=semantic_matches,
        )
        self._show_choices(combined_matches)
        semantic_error = self._searcher.semantic_error
        if semantic_error is None:
            status = (
                f'Query: "{query}" | {len(self._searcher.exact_matches(query))} text matches | '
                f"{len(semantic_matches)} semantic matches"
            )
        else:
            status = (
                f'Query: "{query}" | {len(self._searcher.exact_matches(query))} text matches | '
                "semantic matches unavailable; see log for details"
            )
        self.query_one("#taxonomy-choice-status", Static).update(status)


class RuleSuggestionScreen(ModalScreen[bool]):
    """Screen for confirming one suggested automation rule."""

    BINDINGS = [
        ("escape", "skip", "Skip"),
        ("n", "skip", "Skip"),
        ("y", "create", "Create"),
    ]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="rule-suggestion-dialog"):
                yield Label("Create automation rule?")
                yield Static(self._prompt, id="rule-suggestion-prompt")
                with Horizontal(id="rule-suggestion-buttons"):
                    yield Button("Create", id="rule-suggestion-create", variant="primary")
                    yield Button("Skip", id="rule-suggestion-skip")

    def on_mount(self) -> None:
        self.query_one("#rule-suggestion-create", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rule-suggestion-create":
            self.dismiss(True)
        elif event.button.id == "rule-suggestion-skip":
            self.dismiss(False)

    def action_create(self) -> None:
        self.dismiss(True)

    def action_skip(self) -> None:
        self.dismiss(False)
