from datetime import date
from typing import cast
from unittest.mock import patch

import pytest
from receipts_ai.models.transaction import (
    Receipt,
    ReceiptItem,
    Source,
    Transaction,
    TransactionUserOverrides,
)
from textual.widgets import DataTable, Input

from receipts_ai_cli.app import ReceiptItemsScreen, ReceiptsAIApp


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
        amount_input = screen.query_one("#receipt-amount", Input)
        row = table.get_row_at(0)

    assert isinstance(screen, ReceiptItemsScreen)
    assert date_input.value == "2026-05-06"
    assert payee_input.value == "Payee receipt"
    assert amount_input.value == "-7.5"
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
async def test_enter_on_transaction_without_receipt_items_stays_on_transactions() -> None:
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

        table = cast(DataTable[str], app.query_one("#transactions", DataTable))

    assert table.row_count == 1


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

    with patch("receipts_ai_cli.app.set_transaction_user_overrides") as mock_set:
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
            amount_input = app.screen.query_one("#receipt-amount", Input)
            amount_input.focus()
            amount_input.value = "-8.25"
            await pilot.press("enter")
            await pilot.pause(0.1)

        assert mock_set.called

    assert transaction.user_overrides is not None
    assert transaction.user_overrides.transaction_date == date(2026, 5, 7)
    assert transaction.user_overrides.payee == "Edited Payee"
    assert transaction.user_overrides.amount == "-8.25"


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
    transaction.receipt = Receipt(items=[item])
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.set_receipt_item_user_overrides") as mock_set:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("right", "right", "right", "enter")
            editor_input = app.screen.query_one("#cell-edit-input", Input)
            editor_input.value = "8.25"
            await pilot.press("enter")
            await pilot.pause(0.1)

            table = cast(DataTable[str], app.screen.query_one("#receipt-items", DataTable))
            row = table.get_row_at(0)

        assert mock_set.called

    assert item.user_overrides is not None
    assert item.user_overrides.amount == "8.25"
    assert row[3] == "8.25"


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
    transaction.receipt = Receipt(items=[item])
    app = ReceiptsAIApp(transaction_loader=lambda: [transaction])

    with patch("receipts_ai_cli.app.set_receipt_item_user_overrides") as mock_set:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause(0.1)
            # First column is Description
            await pilot.press("enter")
            editor_input = app.screen.query_one("#cell-edit-input", Input)
            editor_input.value = "New Latte Name"
            await pilot.press("enter")
            await pilot.pause(0.1)

            table = cast(DataTable[str], app.screen.query_one("#receipt-items", DataTable))
            row = table.get_row_at(0)

        assert mock_set.called

    assert item.user_overrides is not None
    assert item.user_overrides.description == "New Latte Name"
    assert row[0] == "New Latte Name"


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
