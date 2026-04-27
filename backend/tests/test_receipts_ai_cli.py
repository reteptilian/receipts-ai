from __future__ import annotations

import csv
import sys
from datetime import date
from io import StringIO
from pathlib import Path

import pytest

from receipts_ai import receipts_ai
from receipts_ai.models.transaction import (
    ExtractionMetadata,
    Receipt,
    ReceiptItem,
    Source,
    Transaction,
)
from receipts_ai.receipts_ai import (
    CSV_FIELDNAMES,
    main,
    write_receipt_items_csv,
    write_receipt_json,
    write_transaction_json,
    write_transaction_receipt_items_csv,
)


def test_writes_one_csv_row_per_receipt_item():
    receipt = Receipt(
        subtotal="10.00",
        total="11.00",
        extraction=ExtractionMetadata(model="prebuilt-receipt", confidence=0.97),
        items=[
            ReceiptItem(
                description="Coffee",
                raw_description="COF",
                brave_search_result="Coffee product search result",
                quantity=2,
                unit_price="3.50",
                amount="7.00",
            ),
            ReceiptItem(description="Bagel", amount="3.00", confidence=0.91),
        ],
    )
    output = StringIO()

    write_receipt_items_csv(receipt, output)

    header, *_ = output.getvalue().splitlines()
    assert header == ",".join(CSV_FIELDNAMES)

    rows = list(csv.DictReader(StringIO(output.getvalue())))
    assert rows == [
        {
            "transaction_id": "",
            "transaction_date": "",
            "payee": "",
            "transaction_amount": "",
            "transaction_currency": "",
            "receipt_id": "",
            "source_document_id": "",
            "receipt_number": "",
            "receipt_subtotal": "10.00",
            "receipt_total": "11.00",
            "extraction_model": "prebuilt-receipt",
            "extraction_confidence": "0.97",
            "item_index": "1",
            "item_id": "",
            "item_description": "Coffee",
            "item_raw_description": "COF",
            "item_brave_search_result": "Coffee product search result",
            "item_quantity": "2.0",
            "item_unit_price": "3.50",
            "item_amount": "7.00",
            "item_line_type": "item",
            "item_category_id": "",
            "item_confidence": "",
        },
        {
            "transaction_id": "",
            "transaction_date": "",
            "payee": "",
            "transaction_amount": "",
            "transaction_currency": "",
            "receipt_id": "",
            "source_document_id": "",
            "receipt_number": "",
            "receipt_subtotal": "10.00",
            "receipt_total": "11.00",
            "extraction_model": "prebuilt-receipt",
            "extraction_confidence": "0.97",
            "item_index": "2",
            "item_id": "",
            "item_description": "Bagel",
            "item_raw_description": "",
            "item_brave_search_result": "",
            "item_quantity": "",
            "item_unit_price": "",
            "item_amount": "3.00",
            "item_line_type": "item",
            "item_category_id": "",
            "item_confidence": "0.91",
        },
    ]


def test_writes_transaction_fields_on_each_csv_receipt_item_row():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-11.00",
        currency="USD",
        receipt=Receipt(
            subtotal="10.00",
            total="11.00",
            items=[ReceiptItem(description="Coffee", amount="7.00")],
        ),
    )
    output = StringIO()

    write_transaction_receipt_items_csv(transaction, output)

    rows = list(csv.DictReader(StringIO(output.getvalue())))
    assert rows[0]["transaction_id"] == "receipt_1"
    assert rows[0]["transaction_date"] == "2026-04-27"
    assert rows[0]["payee"] == "Coffee Shop"
    assert rows[0]["transaction_amount"] == "-11.00"
    assert rows[0]["transaction_currency"] == "USD"
    assert rows[0]["item_description"] == "Coffee"


def test_json_output_preserves_nested_receipt_struct():
    receipt = Receipt(
        total="7.00",
        extraction=ExtractionMetadata(model="prebuilt-receipt", raw_text="Coffee 7.00"),
        items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")],
    )
    output = StringIO()

    write_receipt_json(receipt, output)

    assert '"total": "7.00"' in output.getvalue()
    assert '"rawText": "Coffee 7.00"' in output.getvalue()
    assert '"rawDescription": "COF"' in output.getvalue()
    assert '"items": [' in output.getvalue()


def test_transaction_json_output_wraps_nested_receipt_struct():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(
            total="7.00",
            items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")],
        ),
    )
    output = StringIO()

    write_transaction_json(transaction, output)

    assert '"transactionDate": "2026-04-27"' in output.getvalue()
    assert '"payee": "Coffee Shop"' in output.getvalue()
    assert '"receipt": {' in output.getvalue()
    assert '"rawDescription": "COF"' in output.getvalue()


def test_main_can_enrich_items_with_brave_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(
        items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")]
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=receipt,
    )
    calls: list[Transaction] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai",
            "--brave-search",
            "--brave-search-delay-seconds",
            "1.1",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(
        receipts_ai,
        "analyze_receipt_file",
        fake_analyze_receipt_file,
    )
    monkeypatch.setattr(
        receipts_ai,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        calls.append(transaction_to_enrich)
        assert request_delay_seconds == 1.1
        assert transaction_to_enrich.receipt is not None
        transaction_to_enrich.receipt.items[0].brave_search_result = "search payload"
        return transaction_to_enrich

    monkeypatch.setattr(
        receipts_ai,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )

    main()

    assert calls == [transaction]
    assert receipt.items[0].brave_search_result == "search payload"
