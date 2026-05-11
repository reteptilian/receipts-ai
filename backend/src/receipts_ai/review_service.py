from __future__ import annotations

# pyright: reportPrivateUsage=false, reportPrivateLocalImportUsage=false
import json
import logging
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TextIO

from receipts_ai.cache import SqliteCallCache
from receipts_ai.document_intelligence import analyze_receipt_file
from receipts_ai.ingest_receipts import (
    RECEIPT_PIPELINES,
    VISIONKIT_OLLAMA_CACHE_VERSION,
    UrlLibOllamaClient,
    _mean_confidence,
    _ollama_timeout_seconds,
    _ollama_url,
    _receipt_data_from_ollama_response,
    _receipt_ollama_model,
    _receipt_ollama_output_schema,
    _receipt_ollama_think,
    _visionkit_ollama_receipt_prompt,
    _visionkit_text_lines,
    _visionkit_text_observations,
    file_url_from_path,
    populate_transaction_ingestion_metadata,
    sha256_hex,
)
from receipts_ai.models.receipt_data_extraction import (
    ReceiptDataExtraction,
    ReceiptDataExtractionMetadata,
    ReceiptExtractionPipeline,
)
from receipts_ai.models.transaction import IngestionType, Transaction
from receipts_ai.receipt_extraction import (
    receipt_data_from_document_intelligence_result,
    transaction_from_receipt_data,
)
from receipts_ai.review_models import (
    ReceiptComparisonField,
    ReceiptComparisonResult,
    ReceiptExtractionRecord,
    ReceiptReviewStatus,
)
from receipts_ai.review_store import ReceiptReviewStore

LOGGER = logging.getLogger(__name__)


def import_receipt_for_review(
    receipt_path: Path,
    *,
    store: ReceiptReviewStore,
    pipeline: str = "azure",
    cache: SqliteCallCache | None = None,
    force: bool = False,
) -> ReceiptExtractionRecord:
    LOGGER.info("Importing receipt for review: path=%s pipeline=%s", receipt_path, pipeline)
    receipt_sha256_hex = sha256_hex(receipt_path.read_bytes())
    store.upsert_source(sha256_hex=receipt_sha256_hex, image_path=receipt_path)
    if not force:
        existing = store.latest_extraction(receipt_sha256_hex, pipeline=pipeline)
        if existing is not None:
            LOGGER.info(
                "Using existing receipt extraction: path=%s pipeline=%s extraction_id=%s",
                receipt_path,
                pipeline,
                existing.id,
            )
            return existing

    extraction = run_receipt_pipeline(receipt_path, pipeline=pipeline, cache=cache)
    return store.save_extraction(
        receipt_sha256_hex=receipt_sha256_hex,
        pipeline=pipeline,
        model=extraction.receipt_data.extraction.model,
        receipt_data=extraction.receipt_data,
        prompt=extraction.prompt,
        ocr_text=extraction.ocr_text,
        output_schema=extraction.output_schema,
        raw_response=extraction.raw_response,
    )


def reviewed_transaction_for_receipt(
    receipt_path: Path,
    *,
    store: ReceiptReviewStore,
) -> Transaction | None:
    receipt_sha256_hex = sha256_hex(receipt_path.read_bytes())
    receipt_data = store.reviewed_receipt_data(receipt_sha256_hex)
    if receipt_data is None:
        return None
    return populate_transaction_ingestion_metadata(
        transaction_from_receipt_data(receipt_data),
        ingestion_filename=receipt_path.name,
        ingestion_file_url=file_url_from_path(receipt_path),
        ingestion_file_sha256_hex=receipt_sha256_hex,
        ingestion_type=IngestionType.receipt_img,
    )


class PipelineExtractionResult:
    def __init__(
        self,
        *,
        receipt_data: ReceiptDataExtraction,
        prompt: str | None = None,
        ocr_text: str | None = None,
        output_schema: dict[str, Any] | None = None,
        raw_response: str | None = None,
    ) -> None:
        self.receipt_data = receipt_data
        self.prompt = prompt
        self.ocr_text = ocr_text
        self.output_schema = output_schema
        self.raw_response = raw_response


def run_receipt_pipeline(
    receipt_path: Path,
    *,
    pipeline: str,
    cache: SqliteCallCache | None = None,
) -> PipelineExtractionResult:
    if pipeline not in RECEIPT_PIPELINES:
        raise ValueError(f"unsupported receipt pipeline: {pipeline}")
    LOGGER.info("Running receipt pipeline: path=%s pipeline=%s", receipt_path, pipeline)
    if pipeline == "azure":
        result = (
            analyze_receipt_file(receipt_path, cache=cache)
            if cache is not None
            else analyze_receipt_file(receipt_path)
        )
        return PipelineExtractionResult(
            receipt_data=receipt_data_from_document_intelligence_result(result),
        )
    if pipeline == "visionkit_ollama":
        return _run_visionkit_ollama_pipeline(receipt_path, cache=cache)
    raise ValueError(f"unsupported receipt pipeline: {pipeline}")


def save_review(
    store: ReceiptReviewStore,
    *,
    receipt_sha256_hex: str,
    corrected_receipt_data: ReceiptDataExtraction,
    status: ReceiptReviewStatus,
    source_extraction_id: int | None = None,
    notes: str | None = None,
) -> None:
    store.save_review(
        receipt_sha256_hex=receipt_sha256_hex,
        corrected_receipt_data=corrected_receipt_data,
        status=status,
        source_extraction_id=source_extraction_id,
        notes=notes,
    )


def compare_reviewed_receipt_to_pipeline(
    receipt_path: Path,
    *,
    store: ReceiptReviewStore,
    candidate_pipeline: str,
    cache: SqliteCallCache | None = None,
) -> ReceiptComparisonResult:
    receipt_sha256_hex = sha256_hex(receipt_path.read_bytes())
    reviewed = store.review(receipt_sha256_hex)
    if reviewed is None or reviewed.status != ReceiptReviewStatus.reviewed:
        raise ValueError(f"receipt {receipt_sha256_hex} does not have reviewed data")

    candidate = import_receipt_for_review(
        receipt_path,
        store=store,
        pipeline=candidate_pipeline,
        cache=cache,
        force=True,
    )
    result = compare_receipt_data(
        reviewed.corrected_receipt_data,
        candidate.receipt_data,
        receipt_sha256_hex=receipt_sha256_hex,
        candidate_pipeline=candidate_pipeline,
        candidate_extraction_id=candidate.id,
    )
    store.save_comparison(result)
    return result


def compare_receipt_data(
    expected: ReceiptDataExtraction,
    actual: ReceiptDataExtraction,
    *,
    receipt_sha256_hex: str,
    candidate_pipeline: str,
    candidate_extraction_id: int | None = None,
) -> ReceiptComparisonResult:
    fields: list[ReceiptComparisonField] = []
    _append_field(fields, "merchantName", expected.merchant_name, actual.merchant_name, fuzzy=True)
    _append_field(fields, "transactionDate", expected.transaction_date, actual.transaction_date)
    _append_field(fields, "currency", expected.currency, actual.currency)
    _append_field(fields, "receiptNumber", expected.receipt_number, actual.receipt_number)
    _append_decimal_field(fields, "subtotal", expected.subtotal, actual.subtotal)
    _append_decimal_field(fields, "totalTax", expected.total_tax, actual.total_tax)
    _append_decimal_field(fields, "total", expected.total, actual.total)
    _append_decimal_field(
        fields,
        "items.count",
        str(len(expected.items)),
        str(len(actual.items)),
    )

    max_items = max(len(expected.items), len(actual.items))
    for index in range(max_items):
        expected_item = expected.items[index] if index < len(expected.items) else None
        actual_item = actual.items[index] if index < len(actual.items) else None
        prefix = f"items[{index}]"
        _append_field(
            fields,
            f"{prefix}.description",
            expected_item.description if expected_item is not None else None,
            actual_item.description if actual_item is not None else None,
            fuzzy=True,
        )
        _append_decimal_field(
            fields,
            f"{prefix}.amount",
            expected_item.amount if expected_item is not None else None,
            actual_item.amount if actual_item is not None else None,
        )
        _append_decimal_field(
            fields,
            f"{prefix}.discountAmount",
            expected_item.discount_amount if expected_item is not None else None,
            actual_item.discount_amount if actual_item is not None else None,
        )
        _append_field(
            fields,
            f"{prefix}.lineType",
            expected_item.line_type if expected_item is not None else None,
            actual_item.line_type if actual_item is not None else None,
        )

    mismatch_count = sum(1 for field in fields if not field.matches)
    score = 1.0 if not fields else (len(fields) - mismatch_count) / len(fields)
    return ReceiptComparisonResult(
        receipt_sha256_hex=receipt_sha256_hex,
        candidate_extraction_id=candidate_extraction_id,
        candidate_pipeline=candidate_pipeline,
        field_count=len(fields),
        mismatch_count=mismatch_count,
        score=score,
        fields=fields,
        created_at=datetime.now(UTC),
    )


def write_training_jsonl(store: ReceiptReviewStore, file: TextIO) -> None:
    schema = _receipt_ollama_output_schema()
    for source, review in store.reviewed_training_examples():
        receipt_data = review.corrected_receipt_data
        user_content = "OCR:\n" + (receipt_data.extraction.raw_text or "")
        assistant_content = json.dumps(
            _receipt_data_training_payload(receipt_data),
            ensure_ascii=False,
            sort_keys=True,
        )
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract receipt data from OCR text as JSON matching the provided schema."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            ],
            "metadata": {
                "image_sha256": source.sha256_hex,
                "image_path": source.image_path,
                "image_file_url": source.file_url,
                "schema": schema,
                "review_status": review.status.value,
                "reviewed_at": (
                    review.reviewed_at.isoformat() if review.reviewed_at is not None else None
                ),
                "source_extraction_id": review.source_extraction_id,
            },
        }
        file.write(json.dumps(example, ensure_ascii=False, sort_keys=True))
        file.write("\n")


def _run_visionkit_ollama_pipeline(
    receipt_path: Path,
    *,
    cache: SqliteCallCache | None,
) -> PipelineExtractionResult:
    observations = _visionkit_text_observations(receipt_path)
    lines = _visionkit_text_lines(observations)
    if not lines:
        raise ValueError(f"VisionKit OCR did not find text in {receipt_path}")
    raw_text = "\n".join(lines)
    model = _receipt_ollama_model()
    schema = _receipt_ollama_output_schema()
    prompt = _visionkit_ollama_receipt_prompt(lines)
    think = _receipt_ollama_think()
    request = {
        "version": VISIONKIT_OLLAMA_CACHE_VERSION,
        "model": model,
        "think": think,
        "prompt": prompt,
        "format": schema,
    }
    response: str | None = None
    if cache is not None:
        cached_response = cache.get("ollama_receipt_extraction", request)
        if isinstance(cached_response, str):
            response = cached_response
    if response is None:
        response = UrlLibOllamaClient(
            url=_ollama_url(),
            model=model,
            timeout_seconds=_ollama_timeout_seconds(),
            think=think,
        ).complete_structured(
            prompt,
            options={"temperature": 0},
            output_format=schema,
        )
        if cache is not None:
            cache.set("ollama_receipt_extraction", request, response)

    receipt_data = _receipt_data_from_ollama_response(response, raw_text=raw_text)
    receipt_data.extraction = ReceiptDataExtractionMetadata(
        pipeline=ReceiptExtractionPipeline.visionkit_ollama,
        model=f"ocrmac+{model}",
        confidence=_mean_confidence(observations),
        raw_text=raw_text,
    )
    return PipelineExtractionResult(
        receipt_data=receipt_data,
        prompt=prompt,
        ocr_text=raw_text,
        output_schema=schema,
        raw_response=response,
    )


def _append_field(
    fields: list[ReceiptComparisonField],
    path: str,
    expected: object | None,
    actual: object | None,
    *,
    fuzzy: bool = False,
) -> None:
    expected_text = _text_or_none(expected)
    actual_text = _text_or_none(actual)
    if expected_text == actual_text:
        similarity = 1.0
        matches = True
    elif fuzzy and expected_text is not None and actual_text is not None:
        similarity = SequenceMatcher(None, expected_text.casefold(), actual_text.casefold()).ratio()
        matches = similarity >= 0.92
    else:
        similarity = None
        matches = False
    fields.append(
        ReceiptComparisonField(
            path=path,
            expected=expected_text,
            actual=actual_text,
            similarity=similarity,
            matches=matches,
        )
    )


def _append_decimal_field(
    fields: list[ReceiptComparisonField],
    path: str,
    expected: str | None,
    actual: str | None,
) -> None:
    expected_decimal = _decimal_or_none(expected)
    actual_decimal = _decimal_or_none(actual)
    matches = expected_decimal == actual_decimal
    fields.append(
        ReceiptComparisonField(
            path=path,
            expected=expected,
            actual=actual,
            similarity=1.0 if matches else None,
            matches=matches,
        )
    )


def _decimal_or_none(value: str | None) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(value).quantize(Decimal("0.0001")).normalize()
    except InvalidOperation:
        return None


def _text_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _receipt_data_training_payload(receipt_data: ReceiptDataExtraction) -> dict[str, object]:
    items = [
        {
            "description": item.description,
            "amount": item.amount,
            "discount": _training_discount(item.discount_amount),
        }
        for item in receipt_data.items
        if item.line_type.value != "tax"
    ]
    return {
        "analysis": f"Reviewed receipt with {len(items)} item(s).",
        "merchantName": receipt_data.merchant_name,
        "transactionDate": receipt_data.transaction_date.isoformat(),
        "items": items,
        "subtotal": receipt_data.subtotal or receipt_data.total,
        "tax": receipt_data.total_tax or "0.00",
        "total": receipt_data.total,
    }


def _training_discount(value: str | None) -> str:
    if value is None:
        return "0.00"
    try:
        return format(abs(Decimal(value)), "f")
    except InvalidOperation:
        return value.removeprefix("-")
