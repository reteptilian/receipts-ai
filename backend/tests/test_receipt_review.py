from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from receipts_ai import ingest_receipts, review_cli, review_service
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
    _next_unreviewed_sha,
    _receipt_preview_image,
    _receipt_queue_entries,
    _sanity_check_failures,
)


def _receipt_data(
    *,
    merchant_name: str = "Coffee Shop",
    receipt_date: date | None = None,
    total: str = "7.50",
    raw_text: str = "Coffee Shop\nLatte 7.50",
) -> ReceiptDataExtraction:
    return ReceiptDataExtraction(
        merchant_name=merchant_name,
        transaction_date=receipt_date or date(2026, 5, 9),
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


def test_receipt_queue_entries_mark_unreviewed_and_sort_by_receipt_date(tmp_path: Path):
    early_path = tmp_path / "early.png"
    later_path = tmp_path / "later.png"
    early_path.write_bytes(b"early")
    later_path.write_bytes(b"later")
    early_sha = ingest_receipts.sha256_hex(b"early")
    later_sha = ingest_receipts.sha256_hex(b"later")
    store = ReceiptReviewStore(tmp_path / "reviews.sqlite")
    store.upsert_source(sha256_hex=later_sha, image_path=later_path)
    store.upsert_source(sha256_hex=early_sha, image_path=early_path)
    store.save_extraction(
        receipt_sha256_hex=later_sha,
        pipeline="azure",
        receipt_data=_receipt_data(
            merchant_name="Later Shop",
            receipt_date=date(2026, 5, 9),
        ),
    )
    early_extraction = store.save_extraction(
        receipt_sha256_hex=early_sha,
        pipeline="azure",
        receipt_data=_receipt_data(
            merchant_name="Early Shop",
            receipt_date=date(2026, 5, 8),
        ),
    )
    store.save_review(
        receipt_sha256_hex=early_sha,
        corrected_receipt_data=_receipt_data(
            merchant_name="Early Shop",
            receipt_date=date(2026, 5, 8),
        ),
        status=ReceiptReviewStatus.draft,
        source_extraction_id=early_extraction.id,
    )
    store.save_review(
        receipt_sha256_hex=later_sha,
        corrected_receipt_data=_receipt_data(
            merchant_name="Later Shop",
            receipt_date=date(2026, 5, 9),
        ),
        status=ReceiptReviewStatus.reviewed,
    )

    entries = _receipt_queue_entries(store)

    assert [entry.sha256_hex for entry in entries] == [early_sha, later_sha]
    assert entries[0].label.startswith("* 2026-05-08  Early Shop")
    assert entries[0].reviewed is False
    assert entries[1].label.startswith("2026-05-09  Later Shop")
    assert entries[1].reviewed is True


def test_next_unreviewed_sha_skips_current_receipt(tmp_path: Path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    third_path = tmp_path / "third.png"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    third_path.write_bytes(b"third")
    first_sha = ingest_receipts.sha256_hex(b"first")
    second_sha = ingest_receipts.sha256_hex(b"second")
    third_sha = ingest_receipts.sha256_hex(b"third")
    store = ReceiptReviewStore(tmp_path / "reviews.sqlite")
    store.upsert_source(sha256_hex=first_sha, image_path=first_path)
    store.upsert_source(sha256_hex=second_sha, image_path=second_path)
    store.upsert_source(sha256_hex=third_sha, image_path=third_path)
    for sha, day in [(first_sha, 1), (second_sha, 2), (third_sha, 3)]:
        store.save_extraction(
            receipt_sha256_hex=sha,
            pipeline="azure",
            receipt_data=_receipt_data(receipt_date=date(2026, 5, day)),
        )
    store.save_review(
        receipt_sha256_hex=second_sha,
        corrected_receipt_data=_receipt_data(receipt_date=date(2026, 5, 2)),
        status=ReceiptReviewStatus.reviewed,
    )

    entries = _receipt_queue_entries(store)

    assert _next_unreviewed_sha(entries, first_sha) == third_sha
    assert _next_unreviewed_sha(entries, second_sha) == third_sha
    assert _next_unreviewed_sha(entries, third_sha) == first_sha


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


def test_review_cli_accepts_log_level_after_import_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_path = tmp_path / "receipt.pdf"
    db_path = tmp_path / "reviews.sqlite"

    def fake_import_receipt_for_review(
        receipt_path_arg: Path,
        *,
        store: ReceiptReviewStore,
        pipeline: str,
        cache: object,
        force: bool,
    ) -> SimpleNamespace:
        assert isinstance(store, ReceiptReviewStore)
        assert receipt_path_arg == receipt_path
        assert pipeline == "azure"
        assert cache is None
        assert force is False
        return SimpleNamespace(id=123, receipt_sha256_hex="a" * 64)

    monkeypatch.setattr(review_cli, "import_receipt_for_review", fake_import_receipt_for_review)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-review",
            "--db",
            str(db_path),
            "import",
            "--log-level",
            "INFO",
            str(receipt_path),
        ],
    )

    review_cli.main()

    assert f"{receipt_path}: stored extraction 123 for {'a' * 64}" in capsys.readouterr().out


def test_run_receipt_pipeline_logs_receipt_path_before_azure_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    receipt_path = tmp_path / "receipt.sqlite"

    def fake_analyze_receipt_file(path: Path) -> object:
        assert path == receipt_path
        raise RuntimeError("unsupported file")

    monkeypatch.setattr(review_service, "analyze_receipt_file", fake_analyze_receipt_file)

    with caplog.at_level(logging.INFO, logger=review_service.__name__):
        with pytest.raises(RuntimeError, match="unsupported file"):
            review_service.run_receipt_pipeline(receipt_path, pipeline="azure")

    assert f"Running receipt pipeline: path={receipt_path} pipeline=azure" in caplog.text


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


def test_sanity_check_failures_report_missing_tax_line_item():
    items = [
        {"amount": "10.00", "discount_amount": None, "line_type": "item"},
    ]

    failures = _sanity_check_failures(
        subtotal="10.00",
        total_tax="1.00",
        total="11.00",
        items=items,
    )

    assert failures == [
        {
            "check": "tax = sum(tax line amounts)",
            "expected": "0.00",
            "actual": "1.00",
        },
    ]


def test_sanity_check_failures_report_tax_line_total_mismatch():
    items = [
        {"amount": "10.00", "discount_amount": None, "line_type": "item"},
        {"amount": "0.50", "discount_amount": None, "line_type": "tax"},
    ]

    failures = _sanity_check_failures(
        subtotal="10.00",
        total_tax="1.00",
        total="11.00",
        items=items,
    )

    assert failures == [
        {
            "check": "tax = sum(tax line amounts)",
            "expected": "0.50",
            "actual": "1.00",
        },
    ]


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
