from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Label


class ReceiptsAIApp(App[None]):
    """A tiny Textual app while the CLI project infrastructure takes shape."""

    CSS = """
    Screen {
        align: center middle;
    }

    Label {
        width: auto;
        content-align: center middle;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Hello from Receipts AI")
        yield Footer()


def main() -> None:
    ReceiptsAIApp().run()
