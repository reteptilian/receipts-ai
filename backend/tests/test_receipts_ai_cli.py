from __future__ import annotations

import csv
import sys
from io import StringIO
from pathlib import Path

import pytest

from receipts_ai import receipts_ai
from receipts_ai.models.transaction import ExtractionMetadata, Receipt, ReceiptItem
from receipts_ai.receipts_ai import (
    CSV_FIELDNAMES,
    main,
    write_receipt_items_csv,
    write_receipt_json,
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


def test_main_can_enrich_items_with_brave_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(
        items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")]
    )
    calls: list[Receipt] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_receipt_from_document_intelligence_result(_result: object) -> Receipt:
        return receipt

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
        "receipt_from_document_intelligence_result",
        fake_receipt_from_document_intelligence_result,
    )

    def fake_enrich_receipt_items_with_brave_search(
        receipt_to_enrich: Receipt, *, request_delay_seconds: float | None = None
    ) -> Receipt:
        calls.append(receipt_to_enrich)
        assert request_delay_seconds == 1.1
        receipt_to_enrich.items[0].brave_search_result = "search payload"
        return receipt_to_enrich

    monkeypatch.setattr(
        receipts_ai,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )

    main()

    assert calls == [receipt]
    assert receipt.items[0].brave_search_result == "search payload"
