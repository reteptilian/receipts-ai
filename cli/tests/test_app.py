from datetime import date
from typing import cast

import pytest
from receipts_ai.models.transaction import Source, Transaction
from textual.widgets import DataTable

from receipts_ai_cli.app import ReceiptsAIApp


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
    assert row == ["2026-05-06", "Coffee Shop", "checking.ofx", "-7.50 USD"]
