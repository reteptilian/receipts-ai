from __future__ import annotations

# pyright: reportPrivateUsage=false
import logging
from argparse import Namespace
from datetime import date
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from receipts_ai.budget_categories import BudgetCategoryChoice
from receipts_ai.models.transaction import (
    CategoryAllocation,
    Receipt,
    ReceiptItem,
    RecordType,
    Source,
    Transaction,
    TransactionUserOverrides,
    UserCategoryAllocation,
)
from textual import events
from textual.coordinate import Coordinate
from textual.geometry import Size
from textual.widgets import DataTable, Input, OptionList, Static

from receipts_ai_cli.app import (
    ReceiptItemsScreen,
    ReceiptsAIApp,
    TransactionReviewScreen,
    _configure_logging,
    _log_firestore_configuration,
    main,
)
from receipts_ai_cli.screens.modals import TaxonomyChoiceScreen


def _category_choice(category_id: str, path: str) -> BudgetCategoryChoice:
    return BudgetCategoryChoice(category_id=category_id, path=tuple(path.split(" > ")))


from receipts_ai_cli.taxonomy_selection import TaxonomySearcher


def _transaction(
    transaction_id: str,
    *,
    transaction_date: date,
    amount: str,
) -> Transaction:
    return Transaction(
        id=transaction_id,
        source=Source.bank_statement,
        account_id="checking",
        transaction_date=transaction_date,
        payee=f"Payee {transaction_id}",
        description=f"Description {transaction_id}",
        amount=amount,
        currency="USD",
        ingestion_filename="checking.ofx",
    )


class _FakeTaxonomySearcher:
    choices = (
        "Food, Beverages & Tobacco > Beverages > Coffee",
        "Vehicles & Parts > Vehicle Fuels & Lubricants",
    )
    semantic_error = None

    def exact_matches(self, query: str) -> tuple[str, ...]:
        folded_query = query.casefold()
        return tuple(choice for choice in self.choices if folded_query in choice.casefold())

    def semantic_matches(self, _query: str) -> tuple[str, ...]:
        return ()

    def combine_matches(
        self,
        query: str,
        *,
        semantic_matches: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if not query.strip():
            return self.choices
        combined: list[str] = []
        for choice in (*self.exact_matches(query), *semantic_matches):
            if choice not in combined:
                combined.append(choice)
        return tuple(combined)


def test_app_title_defaults_to_class_name() -> None:
    app = ReceiptsAIApp()

    assert app.title == "ReceiptsAIApp"


def test_configure_logging_writes_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "receipts-ai.log"

    _configure_logging(log_level="INFO", log_file=log_path)
    logging.getLogger("receipts_ai_cli.test").info("hello log file")
    root_logger = logging.getLogger()

    assert log_path.read_text(encoding="utf-8") != ""
    assert "hello log file" in log_path.read_text(encoding="utf-8")
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], logging.FileHandler)


def test_configure_logging_without_log_file_stays_off_terminal() -> None:
    _configure_logging(log_level="INFO", log_file=None)
    root_logger = logging.getLogger()

    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], logging.NullHandler)


def test_log_firestore_configuration_logs_emulator(caplog: pytest.LogCaptureFixture) -> None:
    with patch("receipts_ai_cli.app.config_value") as mock_config_value:

        def fake_config_value(key: str) -> str | None:
            return {
                "FIRESTORE_EMULATOR_HOST": "127.0.0.1:8080",
                "FIREBASE_SERVICE_ACCT_KEY_FILEPATH": None,
            }[key]

        mock_config_value.side_effect = fake_config_value

        with caplog.at_level(logging.INFO):
            _log_firestore_configuration()

    assert "Configured to use Firestore emulator at 127.0.0.1:8080" in caplog.text


def test_log_firestore_configuration_logs_service_account(caplog: pytest.LogCaptureFixture) -> None:
    with patch("receipts_ai_cli.app.config_value") as mock_config_value:

        def fake_config_value(key: str) -> str | None:
            return {
                "FIRESTORE_EMULATOR_HOST": None,
                "FIREBASE_SERVICE_ACCT_KEY_FILEPATH": "/tmp/firebase.json",
            }[key]

        mock_config_value.side_effect = fake_config_value

        with caplog.at_level(logging.INFO):
            _log_firestore_configuration()

    assert (
        "Configured to use Cloud Firestore with service account file /tmp/firebase.json"
        in caplog.text
    )


def test_main_configures_logging_then_runs_app(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path("/tmp/receipts-ai-dev.config")
    parse_args_result = Namespace(config_file=config_path, log_file=None, log_level="INFO")
    config_calls: list[Path | None] = []
    configure_calls: list[tuple[str, Path | None]] = []

    def fake_configure_config_file(config_file: Path | None) -> None:
        config_calls.append(config_file)

    def fake_configure_logging(*, log_level: str, log_file: Path | None) -> None:
        configure_calls.append((log_level, log_file))

    monkeypatch.setattr("receipts_ai_cli.app._parse_args", lambda: parse_args_result)
    monkeypatch.setattr(
        "receipts_ai_cli.app.configure_config_file",
        fake_configure_config_file,
    )
    monkeypatch.setattr(
        "receipts_ai_cli.app._configure_logging",
        fake_configure_logging,
    )

    firestore_calls = 0
    run_calls = 0

    def fake_log_firestore_configuration() -> None:
        nonlocal firestore_calls
        firestore_calls += 1

    def fake_run(_self: ReceiptsAIApp) -> None:
        nonlocal run_calls
        run_calls += 1

    monkeypatch.setattr(
        "receipts_ai_cli.app._log_firestore_configuration", fake_log_firestore_configuration
    )
    monkeypatch.setattr(ReceiptsAIApp, "run", fake_run)

    main()

    assert config_calls == [config_path]
    assert configure_calls == [("INFO", None)]
    assert firestore_calls == 1
    assert run_calls == 1


def test_taxonomy_searcher_combines_exact_matches_before_semantic_matches() -> None:
    searcher = TaxonomySearcher.__new__(TaxonomySearcher)
    searcher._choices = (
        "Sony",
        "Sony > Cameras",
        "Electronics > Video Game Accessories > Monitors",
        "Home > Decor",
    )
    searcher._semantic_error = None
    searcher._embedding_client = None

    assert searcher.exact_matches("Sony") == (
        "Sony",
        "Sony > Cameras",
    )
    assert searcher.combine_matches(
        "gaming screen",
        semantic_matches=("Electronics > Video Game Accessories > Monitors",),
    ) == ("Electronics > Video Game Accessories > Monitors",)
    assert searcher.combine_matches(
        "Sony",
        semantic_matches=(
            "Electronics > Video Game Accessories > Monitors",
            "Sony > Cameras",
        ),
    ) == (
        "Sony",
        "Sony > Cameras",
        "Electronics > Video Game Accessories > Monitors",
    )


@pytest.mark.anyio
async def test_app_displays_transactions_in_table() -> None:
    transaction = Transaction(
        id="transaction_1",
        source=Source.bank_statement,
        account_id="checking",
        transaction_date=date(2026, 5, 6),
        payee="Coffee Shop",
        description="POS PURCHASE COFFEE",
        amount="-7.5",
        currency="USD",
        ingestion_filename="checking.ofx",
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        row = table.get_row_at(0)

    assert table.row_count == 1
    assert row == [
        "2026-05-06",
        "Coffee Shop",
        "POS PURCHASE COFFEE",
        "",
        "Unreviewed",
        "",
        "checking.ofx",
        "",
        "-7.50 USD",
    ]


@pytest.mark.anyio
async def test_transaction_table_description_column_has_capped_fixed_width() -> None:
    transaction = Transaction(
        id="transaction_1",
        source=Source.bank_statement,
        account_id="checking",
        transaction_date=date(2026, 5, 6),
        payee="Coffee Shop",
        description="X" * 200,
        amount="-7.5",
        currency="USD",
        ingestion_filename="checking.ofx",
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        app.on_resize(events.Resize(Size(240, 24), Size(240, 24)))
        description_column = table.ordered_columns[2]

    assert description_column.auto_width is False
    assert description_column.width == 52


@pytest.mark.anyio
async def test_app_displays_transaction_overrides_in_table() -> None:
    transaction = _transaction(
        "transaction_1",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.user_overrides = TransactionUserOverrides(
        transaction_date=date(2026, 5, 7),
        payee="Edited Coffee Shop",
        amount="-8.25",
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        row = table.get_row_at(0)

    assert row[0] == "2026-05-07"
    assert row[1] == "Edited Coffee Shop"
    assert row[8] == "-8.25 USD"


@pytest.mark.anyio
async def test_app_displays_transaction_taxonomy_in_table() -> None:
    transaction = _transaction(
        "transaction_1",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.taxonomy = "Food, Beverages & Tobacco > Beverages > Coffee"
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        row = table.get_row_at(0)

    assert row[3] == "Food, Beverages & Tobacco > Beverages > Coffee"


@pytest.mark.anyio
async def test_app_displays_transaction_category_allocations_in_table() -> None:
    no_category = _transaction(
        "no_category",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    single_category = _transaction(
        "single_category",
        transaction_date=date(2026, 5, 7),
        amount="-8.25",
    )
    single_category.category_allocations = [
        CategoryAllocation(category_id="Taxes > Income Taxes", amount="-8.25")
    ]
    multiple_categories = _transaction(
        "multiple_categories",
        transaction_date=date(2026, 5, 8),
        amount="-100",
    )
    multiple_categories.category_allocations = [
        CategoryAllocation(category_id="Meals > Coffee", amount="-35"),
        CategoryAllocation(category_id="Taxes > Income Taxes", amount="-65"),
    ]
    app = ReceiptsAIApp(
        transaction_loader=lambda: [no_category, single_category, multiple_categories]
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        rows = [table.get_row_at(row_index) for row_index in range(table.row_count)]

    assert rows[0][5] == ""
    assert rows[1][5] == "Taxes > Income Taxes"
    assert rows[2][5] == "Taxes > Income Taxes, ..."


@pytest.mark.anyio
async def test_app_displays_transaction_category_allocation_overrides() -> None:
    transaction = _transaction(
        "transaction_1",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Original", amount="-7.5")]
    transaction.user_overrides = TransactionUserOverrides(
        category_allocations=[
            UserCategoryAllocation(category_id="Edited", amount="-7.5"),
        ]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        row = table.get_row_at(0)

    assert row[5] == "Edited"


@pytest.mark.anyio
async def test_app_marks_transactions_with_receipt_items() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.receipt = Receipt(
        items=[
            ReceiptItem(
                description="Latte",
                amount="7.5",
                net_amount="7.5",
            )
        ]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        row = table.get_row_at(0)

    assert row[7] == "Y"


@pytest.mark.anyio
async def test_app_collapses_linked_bst_and_rbt_with_receipt_indicator() -> None:
    bst = _transaction(
        "statement",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    bst.record_type = RecordType.bank_statement
    bst.linked_receipt_based_transaction_id = "receipt"
    rbt = Transaction(
        id="receipt",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        transaction_date=date(2026, 5, 6),
        payee="Coffee Shop",
        amount="-7.5",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Latte", amount="7.5", net_amount="7.5")]),
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [bst, rbt])

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        row = table.get_row_at(0)

    assert table.row_count == 1
    assert row[1] == "Payee statement"
    assert row[7] == "Y"


@pytest.mark.anyio
async def test_app_links_selected_bst_and_rbt_then_refreshes() -> None:
    bst = _transaction(
        "statement",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    bst.record_type = RecordType.bank_statement
    rbt = Transaction(
        id="receipt",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        transaction_date=date(2026, 5, 6),
        payee="Coffee Shop",
        amount="-7.5",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Latte", amount="7.5", net_amount="7.5")]),
    )
    transactions = [bst, rbt]

    def link_transactions(bank_statement_id: str, receipt_based_id: str) -> None:
        assert bank_statement_id == "statement"
        assert receipt_based_id == "receipt"
        bst.linked_receipt_based_transaction_id = receipt_based_id
        rbt.linked_transaction_ids = [bank_statement_id]

    app = ReceiptsAIApp(transaction_loader=lambda: list(transactions))

    with patch(
        "receipts_ai_cli.app.link_bank_statement_transaction_to_receipt",
        side_effect=link_transactions,
    ) as mock_link:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("space", "down", "space", "l")
            await pilot.pause(0.5)

            table = cast(DataTable[str], app.query_one("#transactions", DataTable))

        mock_link.assert_called_once_with("statement", "receipt")

    assert table.row_count == 1
    assert table.get_row_at(0)[7] == "Y"


@pytest.mark.anyio
async def test_app_unlinks_current_linked_bst_then_refreshes() -> None:
    bst = _transaction(
        "statement",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    bst.record_type = RecordType.bank_statement
    bst.linked_receipt_based_transaction_id = "receipt"
    rbt = Transaction(
        id="receipt",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        linked_transaction_ids=["statement"],
        transaction_date=date(2026, 5, 6),
        payee="Coffee Shop",
        amount="-7.5",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Latte", amount="7.5", net_amount="7.5")]),
    )

    def unlink_transaction(bank_statement_id: str) -> None:
        assert bank_statement_id == "statement"
        bst.linked_receipt_based_transaction_id = None
        rbt.linked_transaction_ids = []

    app = ReceiptsAIApp(transaction_loader=lambda: [bst, rbt])

    with patch(
        "receipts_ai_cli.app.unlink_bank_statement_transaction_from_receipt",
        side_effect=unlink_transaction,
    ) as mock_unlink:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("u")
            await pilot.pause(0.5)

            table = cast(DataTable[str], app.query_one("#transactions", DataTable))

        mock_unlink.assert_called_once_with("statement")

    assert table.row_count == 2


@pytest.mark.anyio
async def test_enter_on_transaction_with_receipt_items_opens_items_screen() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.receipt = Receipt(
        items=[
            ReceiptItem(
                description="Latte",
                quantity=2,
                unit_price="3.75",
                amount="7.5",
                discount_amount="-1",
                discount_description="Promo",
                net_amount="6.5",
                category_id="coffee",
                taxonomy="Food > Drink > Latte",
            )
        ]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.1)

        screen = app.screen
        allocations_title = screen.query_one("#category-allocations-title", Static)
        allocations = cast(DataTable[str], screen.query_one("#category-allocations", DataTable))
        table = cast(DataTable[str], screen.query_one("#receipt-items", DataTable))
        date_input = screen.query_one("#receipt-date", Input)
        payee_input = screen.query_one("#receipt-payee", Input)
        description_input = screen.query_one("#receipt-description", Input)
        amount_input = screen.query_one("#receipt-amount", Input)
        controls = screen.query_one("#review-controls", Static)
        row = table.get_row_at(0)

    assert isinstance(screen, ReceiptItemsScreen)
    assert allocations_title.has_class("hidden")
    assert allocations.has_class("hidden")
    assert "Add allocation" not in str(controls.content)
    assert "Delete allocation" not in str(controls.content)
    assert date_input.value == "2026-05-06"
    assert amount_input.value == "-7.5"
    assert payee_input.value == "Payee receipt"
    assert description_input.value == "Description receipt"
    assert row == [
        "Latte",
        "2.0",
        "3.75",
        "7.5",
        "-1",
        "Promo",
        "6.5",
        "item",
        "coffee",
        "Food > Drink > Latte",
    ]


@pytest.mark.anyio
async def test_enter_on_transaction_without_receipt_items_opens_review_screen() -> None:
    transaction = _transaction(
        "no_receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        screen = app.screen
        receipt_items = cast(DataTable[str], screen.query_one("#receipt-items", DataTable))
        controls = screen.query_one("#review-controls", Static)

    assert isinstance(screen, TransactionReviewScreen)
    assert receipt_items.row_count == 0
    assert "Save and Exit" in str(controls.content)
    assert "Exit Without Saving" in str(controls.content)


@pytest.mark.anyio
async def test_transaction_taxonomy_is_focusable_and_editable_with_keyboard() -> None:
    transaction = _transaction(
        "no_receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Gas", amount="-7.5")]
    transaction.taxonomy = "Vehicles & Parts > Vehicle Fuels & Lubricants"
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch(
        "receipts_ai_cli.screens.modals.taxonomy_selector_searcher",
        return_value=_FakeTaxonomySearcher(),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            for _ in range(8):
                if app.screen.focused is app.screen.query_one("#transaction-taxonomy", Static):
                    break
                await pilot.press("tab")
                await pilot.pause(0.1)

            assert app.screen.focused is app.screen.query_one("#transaction-taxonomy", Static)

            await pilot.press("enter")
            await pilot.pause(0.1)

            assert isinstance(app.screen, TaxonomyChoiceScreen)


@pytest.mark.anyio
async def test_transaction_taxonomy_opens_picker_on_click() -> None:
    transaction = _transaction(
        "no_receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Gas", amount="-7.5")]
    transaction.taxonomy = "Vehicles & Parts > Vehicle Fuels & Lubricants"
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch(
        "receipts_ai_cli.screens.modals.taxonomy_selector_searcher",
        return_value=_FakeTaxonomySearcher(),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            clicked = await pilot.click("#transaction-taxonomy")
            await pilot.pause(0.1)

            assert clicked
            assert isinstance(app.screen, TaxonomyChoiceScreen)


@pytest.mark.anyio
async def test_review_screen_displays_mcc_description_as_read_only_reference() -> None:
    transaction = _transaction(
        "mcc",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.mcc_description = "Grocery Stores, Supermarkets"
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        mcc_description = app.screen.query_one("#transaction-mcc-description", Static)

    assert str(mcc_description.content) == "MCC description: Grocery Stores, Supermarkets"
    assert not mcc_description.has_class("hidden")


@pytest.mark.anyio
async def test_review_screen_hides_blank_mcc_description_reference() -> None:
    transaction = _transaction(
        "no_mcc",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        mcc_description = app.screen.query_one("#transaction-mcc-description", Static)

    assert str(mcc_description.content) == ""
    assert mcc_description.has_class("hidden")


@pytest.mark.anyio
async def test_app_displays_transactions_sorted_by_date_then_amount() -> None:
    transactions = [
        _transaction("late", transaction_date=date(2026, 5, 7), amount="-1"),
        _transaction("higher_amount", transaction_date=date(2026, 5, 6), amount="12"),
        _transaction("earlier", transaction_date=date(2026, 5, 5), amount="100"),
        _transaction("lower_amount", transaction_date=date(2026, 5, 6), amount="2"),
    ]
    app = ReceiptsAIApp(transaction_loader=lambda: transactions)

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        displayed_dates_and_amounts = [
            (table.get_row_at(row_index)[0], table.get_row_at(row_index)[8])
            for row_index in range(table.row_count)
        ]

    assert displayed_dates_and_amounts == [
        ("2026-05-05", "100.00 USD"),
        ("2026-05-06", "2.00 USD"),
        ("2026-05-06", "12.00 USD"),
        ("2026-05-07", "-1.00 USD"),
    ]


@pytest.mark.anyio
async def test_receipt_header_inputs_update_transaction_overrides() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    transaction.receipt = Receipt(
        items=[
            ReceiptItem(
                description="Latte",
                amount="7.5",
                net_amount="7.5",
            )
        ]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            date_input = app.screen.query_one("#receipt-date", Input)
            date_input.focus()
            date_input.value = "2026-05-07"
            await pilot.press("enter")
            payee_input = app.screen.query_one("#receipt-payee", Input)
            payee_input.focus()
            payee_input.value = "Edited Payee"
            await pilot.press("enter")
            description_input = app.screen.query_one("#receipt-description", Input)
            description_input.focus()
            description_input.value = "Edited Description"
            await pilot.press("enter")
            assert transaction.user_overrides is None
            cast(TransactionReviewScreen, app.screen).action_save_and_exit()
            await pilot.pause(0.1)

        assert mock_save.called

    assert transaction.user_overrides is not None
    assert transaction.user_overrides.transaction_date == date(2026, 5, 7)
    assert transaction.user_overrides.payee == "Edited Payee"
    assert transaction.user_overrides.description == "Edited Description"


@pytest.mark.anyio
async def test_receipt_item_cells_update_item_overrides() -> None:
    item = ReceiptItem(
        description="Latte",
        quantity=2,
        unit_price="3.75",
        amount="7.5",
        net_amount="7.5",
    )
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    transaction.receipt = Receipt(items=[item])
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("right", "right", "right", "enter")
            editor_input = app.screen.query_one("#cell-edit-input", Input)
            editor_input.value = "8.25"
            await pilot.press("enter")
            assert item.user_overrides is None
            await pilot.press("s")
            await pilot.pause(0.1)

            table = cast(DataTable[str], app.query_one("#transactions", DataTable))
            row = table.get_row_at(0)

        assert mock_save.called

    assert transaction.receipt is not None
    assert transaction.receipt.items[0].user_overrides is not None
    assert transaction.receipt.items[0].user_overrides.amount == "8.25"
    assert row[8] == "-7.50 USD"


@pytest.mark.anyio
async def test_receipt_item_description_update() -> None:
    item = ReceiptItem(
        description="Original Latte",
        amount="7.5",
        net_amount="7.5",
    )
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    transaction.receipt = Receipt(items=[item])
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)
            # First column is Description
            await pilot.press("enter")
            editor_input = app.screen.query_one("#cell-edit-input", Input)
            editor_input.value = "New Latte Name"
            await pilot.press("enter")
            assert item.user_overrides is None
            await pilot.press("s")
            await pilot.pause(0.1)

            table = cast(DataTable[str], app.query_one("#transactions", DataTable))
            row = table.get_row_at(0)

        assert mock_save.called

    assert transaction.receipt is not None
    assert transaction.receipt.items[0].user_overrides is not None
    assert transaction.receipt.items[0].user_overrides.description == "New Latte Name"
    assert row[0] == "2026-05-06"


@pytest.mark.anyio
async def test_category_allocation_category_id_uses_budget_category_picker() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with (
        patch(
            "receipts_ai_cli.app.load_budget_category_options",
            return_value=(
                _category_choice("taxes.income_taxes", "Taxes > Income Taxes"),
                _category_choice("miscellaneous.uncategorized", "Miscellaneous > Uncategorized"),
            ),
        ),
        patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save,
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")

            choice_list = app.screen.query_one("#category-choice-list", OptionList)
            assert choice_list.option_count == 2
            await pilot.pause(0.1)

            await pilot.press("down", "enter")
            await pilot.pause(0.1)

            allocations_table = cast(
                DataTable[str],
                app.screen.query_one("#category-allocations", DataTable),
            )
            row = allocations_table.get_row_at(0)

            await pilot.press("s")
            await pilot.pause(0.1)

            assert mock_save.called

    assert row[0] == "Miscellaneous > Uncategorized"
    assert transaction.user_overrides is not None
    assert transaction.user_overrides.category_allocations is not None
    assert transaction.user_overrides.category_allocations[0].category_id == (
        "miscellaneous.uncategorized"
    )


@pytest.mark.anyio
async def test_receipt_item_category_id_uses_budget_category_picker() -> None:
    item = ReceiptItem(
        description="Latte",
        amount="7.5",
        net_amount="7.5",
        category_id="Coffee",
    )
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    transaction.receipt = Receipt(items=[item])
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with (
        patch(
            "receipts_ai_cli.app.load_budget_category_options",
            return_value=(
                _category_choice("coffee", "Coffee"),
                _category_choice("miscellaneous.uncategorized", "Miscellaneous > Uncategorized"),
            ),
        ),
        patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save,
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            receipt_items = cast(DataTable[str], app.screen.query_one("#receipt-items", DataTable))
            receipt_items.cursor_coordinate = Coordinate(0, 8)
            cast(TransactionReviewScreen, app.screen).action_edit_cell()
            await pilot.pause(0.1)

            choice_list = app.screen.query_one("#category-choice-list", OptionList)
            assert choice_list.option_count == 2

            await pilot.press("down", "enter")
            await pilot.pause(0.1)

            receipt_items = cast(DataTable[str], app.screen.query_one("#receipt-items", DataTable))
            row = receipt_items.get_row_at(0)

            await pilot.press("s")
            await pilot.pause(0.1)

            assert mock_save.called

    assert row[8] == "Miscellaneous > Uncategorized"
    assert transaction.receipt is not None
    assert transaction.receipt.items[0].user_overrides is not None
    assert transaction.receipt.items[0].user_overrides.category_id == (
        "miscellaneous.uncategorized"
    )


@pytest.mark.anyio
async def test_receipt_item_taxonomy_uses_advanced_taxonomy_picker() -> None:
    class FakeTaxonomySearcher:
        choices = (
            "Sony > Cameras",
            "Electronics > Video Game Accessories > Monitors",
            "Home > Decor",
        )
        semantic_error = None

        def exact_matches(self, query: str) -> tuple[str, ...]:
            folded_query = query.casefold()
            return tuple(choice for choice in self.choices if folded_query in choice.casefold())

        def semantic_matches(self, query: str) -> tuple[str, ...]:
            if query == "gaming screen":
                return ("Electronics > Video Game Accessories > Monitors",)
            return ()

        def combine_matches(
            self,
            query: str,
            *,
            semantic_matches: tuple[str, ...] = (),
        ) -> tuple[str, ...]:
            if not query.strip():
                return self.choices
            combined: list[str] = []
            for choice in (*self.exact_matches(query), *semantic_matches):
                if choice not in combined:
                    combined.append(choice)
            return tuple(combined)

    item = ReceiptItem(
        description="Gaming Display",
        amount="299.99",
        net_amount="299.99",
        taxonomy="Electronics > Displays",
    )
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-299.99",
    )
    transaction.receipt = Receipt(items=[item])
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with (
        patch(
            "receipts_ai_cli.screens.modals.taxonomy_selector_searcher",
            return_value=FakeTaxonomySearcher(),
        ),
        patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save,
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            receipt_items = cast(DataTable[str], app.screen.query_one("#receipt-items", DataTable))
            receipt_items.cursor_coordinate = Coordinate(0, 9)
            cast(TransactionReviewScreen, app.screen).action_edit_cell()
            await pilot.pause(0.1)

            taxonomy_input = app.screen.query_one("#taxonomy-choice-input", Input)
            choice_list = app.screen.query_one("#taxonomy-choice-list", OptionList)
            assert taxonomy_input.value == ""
            assert choice_list.option_count == 3

            for key in "gaming screen":
                await pilot.press(key)
            await pilot.pause(0.5)

            assert choice_list.option_count == 1
            assert str(choice_list.get_option_at_index(0).prompt) == (
                "Electronics > Video Game Accessories > Monitors"
            )

            await pilot.press("enter")
            await pilot.pause(0.1)

            row = receipt_items.get_row_at(0)

            await pilot.press("s")
            await pilot.pause(0.1)

            assert mock_save.called

    assert row[9] == "Electronics > Video Game Accessories > Monitors"
    assert transaction.receipt is not None
    assert transaction.receipt.items[0].user_overrides is not None
    assert transaction.receipt.items[0].user_overrides.taxonomy == (
        "Electronics > Video Game Accessories > Monitors"
    )


@pytest.mark.anyio
async def test_transaction_taxonomy_uses_advanced_taxonomy_picker_and_can_clear() -> None:
    class FakeTaxonomySearcher:
        choices = (
            "Food, Beverages & Tobacco > Beverages > Coffee",
            "Home & Garden > Kitchen & Dining > Tableware",
        )
        semantic_error = None

        def exact_matches(self, query: str) -> tuple[str, ...]:
            folded_query = query.casefold()
            return tuple(choice for choice in self.choices if folded_query in choice.casefold())

        def semantic_matches(self, query: str) -> tuple[str, ...]:
            if query == "coffee beans":
                return ("Food, Beverages & Tobacco > Beverages > Coffee",)
            return ()

        def combine_matches(
            self,
            query: str,
            *,
            semantic_matches: tuple[str, ...] = (),
        ) -> tuple[str, ...]:
            if not query.strip():
                return self.choices
            combined: list[str] = []
            for choice in (*self.exact_matches(query), *semantic_matches):
                if choice not in combined:
                    combined.append(choice)
            return tuple(combined)

    transaction = _transaction(
        "transaction_1",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    transaction.taxonomy = "Food, Beverages & Tobacco > Food Items"
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with (
        patch(
            "receipts_ai_cli.screens.modals.taxonomy_selector_searcher",
            return_value=FakeTaxonomySearcher(),
        ),
        patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save,
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            taxonomy_status = app.screen.query_one("#transaction-taxonomy", Static)
            assert str(taxonomy_status.content) == (
                "Taxonomy: Food, Beverages & Tobacco > Food Items"
            )

            await pilot.press("t")
            await pilot.pause(0.1)

            taxonomy_input = app.screen.query_one("#taxonomy-choice-input", Input)
            choice_list = app.screen.query_one("#taxonomy-choice-list", OptionList)
            assert taxonomy_input.value == ""
            assert choice_list.option_count == 2

            for key in "coffee beans":
                await pilot.press(key)
            await pilot.pause(0.5)
            await pilot.press("enter")
            await pilot.pause(0.1)

            taxonomy_status = app.screen.query_one("#transaction-taxonomy", Static)
            assert str(taxonomy_status.content) == (
                "Taxonomy: Food, Beverages & Tobacco > Beverages > Coffee"
            )

            await pilot.press("t")
            await pilot.pause(0.1)
            cast(TaxonomyChoiceScreen, app.screen).action_clear()
            await pilot.pause(0.2)

            assert isinstance(app.screen, TransactionReviewScreen)
            taxonomy_status = app.screen.query_one("#transaction-taxonomy", Static)
            assert str(taxonomy_status.content) == "Taxonomy: Unassigned"

            await pilot.press("s")
            await pilot.pause(0.1)

            assert mock_save.called

    assert transaction.user_overrides is not None
    assert transaction.user_overrides.taxonomy == ""


@pytest.mark.anyio
async def test_save_refuses_mismatched_category_allocations_without_receipt_items() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-6.00")]
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            cast(TransactionReviewScreen, app.screen).action_save_and_exit()
            await pilot.pause(0.1)

            status = app.screen.query_one("#receipt-edit-status", Static)

        mock_save.assert_not_called()

    assert "Save blocked: category allocations must add up" in str(status.content)


@pytest.mark.anyio
async def test_save_uses_receipt_items_instead_of_category_allocations() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-6.00")]
    transaction.receipt = Receipt(
        items=[ReceiptItem(description="Latte", amount="7.5", net_amount="7.5")]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            cast(TransactionReviewScreen, app.screen).action_save_and_exit()
            await pilot.pause(0.1)

        assert mock_save.called


@pytest.mark.anyio
async def test_save_without_new_edits_does_not_generate_rule_suggestions() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.reviewed = True
    transaction.user_overrides = TransactionUserOverrides(
        category_allocations=[
            UserCategoryAllocation(category_id="Coffee", amount="-7.5"),
        ]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with (
        patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save,
        patch("receipts_ai_cli.app.generate_rule_suggestions") as mock_generate,
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            cast(TransactionReviewScreen, app.screen).action_save_and_exit()
            await pilot.pause(0.1)

        assert mock_save.called
        mock_generate.assert_not_called()


@pytest.mark.anyio
async def test_save_validates_receipt_item_net_amounts() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.receipt = Receipt(
        items=[ReceiptItem(description="Latte", amount="8.5", net_amount="7.5")]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            cast(TransactionReviewScreen, app.screen).action_save_and_exit()
            await pilot.pause(0.1)

        assert mock_save.called


@pytest.mark.anyio
async def test_save_refuses_mismatched_receipt_item_net_amounts() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.receipt = Receipt(
        items=[ReceiptItem(description="Latte", amount="7.5", net_amount="6.5")]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            cast(TransactionReviewScreen, app.screen).action_save_and_exit()
            await pilot.pause(0.1)

            status = app.screen.query_one("#receipt-edit-status", Static)

        mock_save.assert_not_called()

    assert "Save blocked: receipt item net amounts must add up" in str(status.content)


@pytest.mark.anyio
async def test_invalid_header_edit_explains_the_problem() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.1)

        date_input = app.screen.query_one("#receipt-date", Input)
        date_input.focus()
        date_input.value = "May 6"
        await pilot.press("enter")
        await pilot.pause(0.1)

        status = app.screen.query_one("#receipt-edit-status", Static)

    assert "Could not update transaction date" in str(status.content)
    assert "YYYY-MM-DD" in str(status.content)


@pytest.mark.anyio
async def test_escape_cancels_active_header_edit_without_exiting_review_screen() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.1)

        payee_input = app.screen.query_one("#receipt-payee", Input)
        payee_input.focus()
        payee_input.value = "Accidental r"
        await pilot.press("escape")
        await pilot.pause(0.1)

        screen = app.screen
        focused = app.focused

    assert isinstance(screen, TransactionReviewScreen)
    assert payee_input.value == "Payee receipt"
    assert isinstance(focused, DataTable)
    assert transaction.user_overrides is None


@pytest.mark.anyio
async def test_linked_transaction_review_saves_bst_overrides_and_rbt_items() -> None:
    bst = _transaction(
        "statement",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    bst.record_type = RecordType.bank_statement
    bst.linked_receipt_based_transaction_id = "receipt"
    bst.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-7.5")]
    rbt = Transaction(
        id="receipt",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        transaction_date=date(2026, 5, 6),
        payee="Receipt Payee",
        amount="-7.5",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Latte", amount="7.5", net_amount="7.5")]),
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [bst, rbt])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            payee_input = app.screen.query_one("#receipt-payee", Input)
            receipt_items = cast(DataTable[str], app.screen.query_one("#receipt-items", DataTable))
            assert payee_input.value == "Payee statement"
            assert receipt_items.get_row_at(0)[0] == "Latte"

            cast(TransactionReviewScreen, app.screen).action_save_and_exit()
            await pilot.pause(0.1)

        mock_save.assert_called_once()

    args, kwargs = mock_save.call_args
    assert args[0] == "statement"
    assert kwargs["reviewed"] is False
    assert kwargs["receipt_transaction_id"] == "receipt"
    assert kwargs["receipt_items"][0].description == "Latte"


@pytest.mark.anyio
async def test_transactions_table_shows_review_and_pair_status() -> None:
    bst = _transaction(
        "statement",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    bst.record_type = RecordType.bank_statement
    bst.linked_receipt_based_transaction_id = "receipt"
    bst.linked_transaction_ids = ["receipt"]
    bst.reviewed = True
    rbt = Transaction(
        id="receipt",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        linked_transaction_ids=["statement"],
        transaction_date=date(2026, 5, 6),
        payee="Receipt Payee",
        amount="-7.5",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Latte", amount="7.5", net_amount="7.5")]),
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [bst, rbt])

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        row = table.get_row_at(0)

    assert row[4] == "Reviewed | Pair"


@pytest.mark.anyio
async def test_review_toggle_is_drafted_until_save_and_persists_for_pair() -> None:
    bst = _transaction(
        "statement",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    bst.record_type = RecordType.bank_statement
    bst.linked_receipt_based_transaction_id = "receipt"
    bst.linked_transaction_ids = ["receipt"]
    rbt = Transaction(
        id="receipt",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        linked_transaction_ids=["statement"],
        transaction_date=date(2026, 5, 6),
        payee="Receipt Payee",
        amount="-7.5",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Latte", amount="7.5", net_amount="7.5")]),
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [bst, rbt])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)

            review_state = app.screen.query_one("#review-state", Static)
            assert str(review_state.content) == "Review: Unreviewed | Type: Pair"
            assert bst.reviewed is False
            assert rbt.reviewed is False

            await pilot.press("r")
            await pilot.pause(0.1)

            review_state = app.screen.query_one("#review-state", Static)
            status = app.screen.query_one("#receipt-edit-status", Static)
            assert str(review_state.content) == "Review: Reviewed | Type: Pair"
            assert "unsaved" in str(status.content)
            assert bst.reviewed is False
            assert rbt.reviewed is False

            await pilot.press("s")
            await pilot.pause(0.1)

        mock_save.assert_called_once()

    assert bst.reviewed is True
    assert rbt.reviewed is True
    _, kwargs = mock_save.call_args
    assert kwargs["reviewed"] is True
    assert kwargs["receipt_transaction_id"] == "receipt"


@pytest.mark.anyio
async def test_app_reloads_transactions_on_return_from_receipt_items() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    transaction.receipt = Receipt(
        items=[
            ReceiptItem(
                description="Latte",
                amount="7.5",
                net_amount="7.5",
            )
        ]
    )

    # Use a mutable list to simulate changing data in Firestore
    transactions = [transaction]

    def loader() -> list[Transaction]:
        return list(transactions)

    app = ReceiptsAIApp(transaction_loader=loader)

    async with app.run_test() as pilot:
        await pilot.pause()

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        assert table.get_row_at(0)[1] == "Payee receipt"

        # Go to items screen
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert isinstance(app.screen, ReceiptItemsScreen)

        # Simulate an edit that would be persisted to Firestore
        transaction.user_overrides = TransactionUserOverrides(payee="Updated Payee")

        # Go back
        await pilot.press("escape")
        await pilot.pause(0.5)  # Wait for refresh

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))
        assert table.get_row_at(0)[1] == "Updated Payee"


@pytest.mark.anyio
async def test_opening_receipt_items_screen_launches_external_viewer() -> None:
    transaction = _transaction(
        "receipt",
        transaction_date=date(2026, 5, 6),
        amount="-7.5",
    )
    # Using setattr to bypass potential static type checking in the test environment
    object.__setattr__(transaction, "ingestion_type", "receipt_img")
    object.__setattr__(transaction, "ingestion_file_url", "/path/to/receipt.jpg")

    transaction.receipt = Receipt(
        items=[
            ReceiptItem(
                description="Latte",
                amount="7.5",
                net_amount="7.5",
            )
        ]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app._open_file_in_external_viewer") as mock_open:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.2)  # Wait for worker to start

            assert isinstance(app.screen, ReceiptItemsScreen)

        mock_open.assert_called_once_with("/path/to/receipt.jpg")
