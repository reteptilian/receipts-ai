from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from receipts_ai import ingest_receipts
from receipts_ai.models.receipt_data_extraction import (
    ExtractedReceiptItem,
    ReceiptDataExtraction,
    ReceiptDataExtractionMetadata,
    ReceiptExtractionPipeline,
)
from receipts_ai.review_models import ReceiptReviewStatus
from receipts_ai.review_service import compare_receipt_data, write_training_jsonl
from receipts_ai.review_store import ReceiptReviewStore
from receipts_ai.review_streamlit_app import (
    _receipt_preview_image,
    _sanity_check_failures,
)


def _receipt_data(
    *,
    merchant_name: str = "Coffee Shop",
    total: str = "7.50",
    raw_text: str = "Coffee Shop\nLatte 7.50",
) -> ReceiptDataExtraction:
    return ReceiptDataExtraction(
        merchant_name=merchant_name,
        transaction_date=date(2026, 5, 9),
        currency="USD",
        subtotal=total,
        total_tax="0.00",
        total=total,
        items=[ExtractedReceiptItem(description="Latte", amount=total)],
        extraction=ReceiptDataExtractionMetadata(
            pipeline=ReceiptExtractionPipeline.azure,
            model="prebuilt-receipt",
            confidence=0.9,
            raw_text=raw_text,
        ),
    )


def test_review_store_round_trips_reviewed_receipt_data(tmp_path: Path):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image")
    receipt_sha = ingest_receipts.sha256_hex(b"image")
    store = ReceiptReviewStore(tmp_path / "reviews.sqlite")

    source = store.upsert_source(sha256_hex=receipt_sha, image_path=receipt_path)
    extraction = store.save_extraction(
        receipt_sha256_hex=receipt_sha,
        pipeline="azure",
        model="prebuilt-receipt",
        receipt_data=_receipt_data(),
    )
    review = store.save_review(
        receipt_sha256_hex=receipt_sha,
        corrected_receipt_data=_receipt_data(merchant_name="Corrected Shop"),
        status=ReceiptReviewStatus.reviewed,
        source_extraction_id=extraction.id,
    )

    assert source.image_path == str(receipt_path.resolve())
    assert extraction.id is not None
    reviewed_data = store.reviewed_receipt_data(receipt_sha)
    assert review.status == ReceiptReviewStatus.reviewed
    assert reviewed_data is not None
    assert reviewed_data.merchant_name == "Corrected Shop"


def test_process_receipt_prefers_reviewed_data(tmp_path: Path):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image")
    receipt_sha = ingest_receipts.sha256_hex(b"image")
    store = ReceiptReviewStore(tmp_path / "reviews.sqlite")
    store.upsert_source(sha256_hex=receipt_sha, image_path=receipt_path)
    store.save_review(
        receipt_sha256_hex=receipt_sha,
        corrected_receipt_data=_receipt_data(merchant_name="Reviewed Shop"),
        status=ReceiptReviewStatus.reviewed,
    )

    transaction = ingest_receipts._process_receipt(
        receipt_path,
        pipeline="azure",
        cache=None,
        review_store=store,
    )

    assert transaction.payee == "Reviewed Shop"
    assert transaction.ingestion_file_sha256_hex == receipt_sha
    assert transaction.receipt is not None
    assert transaction.receipt.items[0].description == "Latte"


def test_compare_receipt_data_reports_field_mismatches():
    result = compare_receipt_data(
        _receipt_data(),
        _receipt_data(total="8.00"),
        receipt_sha256_hex="0" * 64,
        candidate_pipeline="visionkit_ollama",
    )

    mismatches = {field.path for field in result.fields if not field.matches}

    assert "subtotal" in mismatches
    assert "total" in mismatches
    assert "items[0].amount" in mismatches
    assert result.score < 1


def test_compare_receipt_data_fuzzy_compares_merchant_name():
    result = compare_receipt_data(
        _receipt_data(merchant_name="COSTCO"),
        _receipt_data(merchant_name="COSTCO WHOLESALE"),
        receipt_sha256_hex="0" * 64,
        candidate_pipeline="visionkit_ollama",
    )

    merchant_field = next(field for field in result.fields if field.path == "merchantName")

    assert merchant_field.similarity is not None
    assert not merchant_field.matches


def test_write_training_jsonl_uses_reviewed_receipts(tmp_path: Path):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image")
    receipt_sha = ingest_receipts.sha256_hex(b"image")
    store = ReceiptReviewStore(tmp_path / "reviews.sqlite")
    store.upsert_source(sha256_hex=receipt_sha, image_path=receipt_path)
    store.save_review(
        receipt_sha256_hex=receipt_sha,
        corrected_receipt_data=_receipt_data(),
        status=ReceiptReviewStatus.reviewed,
    )
    output_path = tmp_path / "training.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        write_training_jsonl(store, file)

    output = output_path.read_text(encoding="utf-8")
    payload = json.loads(output)
    assistant_content = json.loads(payload["messages"][2]["content"])

    assert "OCR:\\nCoffee Shop\\nLatte 7.50" in output
    assert assistant_content["merchantName"] == "Coffee Shop"
    assert assistant_content["tax"] == "0.00"
    assert assistant_content["items"][0] == {
        "description": "Latte",
        "amount": "7.50",
        "discount": "0.00",
    }
    assert receipt_sha in output


def test_sanity_check_failures_report_total_and_subtotal_mismatches():
    items = pd.DataFrame(
        [
            {"amount": "10.00", "discount_amount": "-2.00", "line_type": "item"},
            {"amount": "1.00", "discount_amount": None, "line_type": "tax"},
        ]
    )

    failures = _sanity_check_failures(
        subtotal="9.00",
        total_tax="1.00",
        total="12.00",
        items=items,
    )

    assert failures == [
        {
            "check": "total = subtotal + tax",
            "expected": "10.00",
            "actual": "12.00",
        },
        {
            "check": "subtotal = sum(item net amounts)",
            "expected": "8.00",
            "actual": "9.00",
        },
    ]


def test_sanity_check_failures_pass_when_amounts_balance():
    items = [
        {"amount": "10.00", "discount_amount": "-2.00", "line_type": "item"},
        {"amount": "1.00", "discount_amount": None, "line_type": "tax"},
    ]

    assert (
        _sanity_check_failures(
            subtotal="8.00",
            total_tax="1.00",
            total="9.00",
            items=items,
        )
        == []
    )


def test_sanity_check_failures_reject_positive_item_discount_amount():
    items = [
        {"amount": "10.00", "discount_amount": "2.00", "line_type": "item"},
    ]

    failures = _sanity_check_failures(
        subtotal="12.00",
        total_tax="0.00",
        total="12.00",
        items=items,
    )

    assert failures == [
        {
            "check": "items[0].discount_amount is negative",
            "expected": "negative discount amount",
            "actual": "2.00",
        },
    ]


def test_receipt_preview_image_renders_pdf_to_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pdf_path = tmp_path / "receipt.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    preview_path = tmp_path / "receipt.png"
    preview_path.write_bytes(b"png bytes")

    @contextmanager
    def fake_preview(path: Path) -> Generator[Path]:
        assert path == pdf_path
        yield preview_path

    monkeypatch.setattr(
        "receipts_ai.review_streamlit_app._visionkit_ocr_image_path",
        fake_preview,
    )

    assert _receipt_preview_image(pdf_path) == b"png bytes"
