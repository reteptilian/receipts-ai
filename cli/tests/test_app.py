from datetime import date
from typing import cast

import pytest
from receipts_ai.models.transaction import Receipt, ReceiptItem, Source, Transaction
from textual.widgets import DataTable

from receipts_ai_cli.app import ReceiptsAIApp


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
                netAmount="7.5",
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
