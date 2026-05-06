from receipts_ai_cli.app import ReceiptsAIApp


def test_app_title_defaults_to_class_name() -> None:
    app = ReceiptsAIApp()

    assert app.title == "ReceiptsAIApp"
