from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
from datetime import date
from pathlib import Path

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
