from datetime import date
from typing import cast
from unittest.mock import patch

import pytest
from receipts_ai.models.transaction import (
    CategoryAllocation,
    Receipt,
    ReceiptItem,
    RecordType,
    Source,
    Transaction,
    TransactionUserOverrides,
)
from textual.widgets import DataTable, Input, Static

from receipts_ai_cli.app import ReceiptItemsScreen, ReceiptsAIApp, TransactionReviewScreen


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


def test_app_title_defaults_to_class_name() -> None:
    app = ReceiptsAIApp()

    assert app.title == "ReceiptsAIApp"


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
        "checking.ofx",
        "",
        "-7.50 USD",
    ]


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
    assert row[5] == "-8.25 USD"


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

    assert row[4] == "Y"


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
    assert row[4] == "Y"


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
    assert table.get_row_at(0)[4] == "Y"


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
                taxonomy1="Food",
                taxonomy2="Drink",
                taxonomy9="Latte",
            )
        ]
    )
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.1)

        screen = app.screen
        table = cast(DataTable[str], screen.query_one("#receipt-items", DataTable))
        date_input = screen.query_one("#receipt-date", Input)
        payee_input = screen.query_one("#receipt-payee", Input)
        description_input = screen.query_one("#receipt-description", Input)
        amount_input = screen.query_one("#receipt-amount", Input)
        row = table.get_row_at(0)

    assert isinstance(screen, ReceiptItemsScreen)
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
        "Food",
        "Drink",
        "",
        "",
        "",
        "",
        "",
        "",
        "Latte",
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
            (table.get_row_at(row_index)[0], table.get_row_at(row_index)[5])
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
        amount="-8.25",
    )
    transaction.category_allocations = [CategoryAllocation(category_id="Coffee", amount="-8.25")]
    transaction.receipt = Receipt(items=[item])
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.save_transaction_review_edits") as mock_save:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("tab")
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
    assert row[5] == "-8.25 USD"


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
            await pilot.press("tab")
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
async def test_save_refuses_mismatched_category_allocations() -> None:
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

            status = app.screen.query_one("#receipt-edit-status", Static)

        mock_save.assert_not_called()

    assert "Save blocked: category allocations must add up" in str(status.content)


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
    assert kwargs["receipt_transaction_id"] == "receipt"
    assert kwargs["receipt_items"][0].description == "Latte"


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
