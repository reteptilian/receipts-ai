from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
import csv
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, Unpack, cast

import pytest

from receipts_ai import firestore_client, ingest_receipts
from receipts_ai.ingest_receipts import (
    CSV_FIELDNAMES,
    main,
    transaction_firestore_document,
    upsert_transaction_to_firestore,
    write_receipt_items_csv,
    write_receipt_json,
    write_transaction_json,
    write_transaction_receipt_items_csv,
    write_transactions_json,
)
from receipts_ai.models.receipt_data_extraction import (
    ExtractedReceiptItem,
    ReceiptDataExtraction,
    ReceiptDataExtractionMetadata,
    ReceiptExtractionPipeline,
)
from receipts_ai.models.transaction import (
    ExtractionMetadata,
    IngestionType,
    LineType,
    Receipt,
    RecordType,
    Source,
    Transaction,
)
from receipts_ai.models.transaction import (
    ReceiptItem as GeneratedReceiptItem,
)


class ReceiptItemKwargs(TypedDict, total=False):
    id: str | None
    description: str
    raw_description: str | None
    brave_search_result: str | None
    quantity: float | None
    unit_price: str | None
    amount: str
    discount_amount: str | None
    discount_description: str | None
    net_amount: str
    line_type: LineType | None
    category_id: str | None
    taxonomy: str | None
    confidence: float | None


def ReceiptItem(**kwargs: Unpack[ReceiptItemKwargs]) -> GeneratedReceiptItem:  # noqa: N802
    if "amount" in kwargs and "net_amount" not in kwargs:
        kwargs["net_amount"] = kwargs["amount"]
    return GeneratedReceiptItem(**kwargs)


def test_visionkit_text_lines_groups_observations_top_to_bottom_and_left_to_right():
    observations = [
        ingest_receipts._VisionKitTextObservation(
            text="TOTAL", confidence=0.9, x=0.1, y=0.1, width=0.2, height=0.08
        ),
        ingest_receipts._VisionKitTextObservation(
            text="2.50", confidence=0.9, x=0.7, y=0.12, width=0.1, height=0.08
        ),
        ingest_receipts._VisionKitTextObservation(
            text="Coffee", confidence=0.9, x=0.2, y=0.72, width=0.2, height=0.1
        ),
        ingest_receipts._VisionKitTextObservation(
            text="Shop", confidence=0.9, x=0.5, y=0.74, width=0.2, height=0.1
        ),
    ]

    assert ingest_receipts._visionkit_text_lines(observations) == [
        "Coffee Shop",
        "TOTAL 2.50",
    ]


def test_visionkit_text_lines_rejects_observations_with_mismatched_heights():
    observations = [
        ingest_receipts._VisionKitTextObservation(
            text="Item A", confidence=0.9, x=0.1, y=0.72, width=0.2, height=0.08
        ),
        ingest_receipts._VisionKitTextObservation(
            text="Item B", confidence=0.9, x=0.1, y=0.62, width=0.2, height=0.08
        ),
        ingest_receipts._VisionKitTextObservation(
            text="SPURIOUS", confidence=0.2, x=0.5, y=0.6, width=0.2, height=0.3
        ),
    ]

    assert ingest_receipts._visionkit_text_lines(observations) == [
        "Item A",
        "SPURIOUS",
        "Item B",
    ]


def test_visionkit_ollama_pipeline_builds_transaction_from_constrained_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image")
    requests: list[dict[str, object]] = []

    class FakeOllamaClient:
        def __init__(self, *, url: str, model: str, timeout_seconds: float, think: bool) -> None:
            requests.append(
                {"url": url, "model": model, "timeout_seconds": timeout_seconds, "think": think}
            )

        def complete_structured(
            self,
            prompt: str,
            *,
            options: dict[str, object],
            output_format: dict[str, object],
        ) -> str:
            requests.append({"prompt": prompt, "options": options, "output_format": output_format})
            return json.dumps(
                {
                    "analysis": "Latte has no associated discount line.",
                    "merchantName": "Coffee Shop",
                    "transactionDate": "2026-05-09",
                    "subtotal": "7.50",
                    "tax": "0.00",
                    "total": "7.50",
                    "items": [{"description": "Latte", "amount": "7.50", "discount": "0.00"}],
                }
            )

    monkeypatch.setattr(
        ingest_receipts,
        "_visionkit_text_observations",
        lambda _path, *, debug_image_path=None: [
            ingest_receipts._VisionKitTextObservation(
                text="Coffee Shop", confidence=0.8, x=0.1, y=0.8, width=0.5, height=0.1
            ),
            ingest_receipts._VisionKitTextObservation(
                text="Latte 7.50", confidence=0.6, x=0.1, y=0.6, width=0.5, height=0.1
            ),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "UrlLibOllamaClient", FakeOllamaClient)
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.test")
    monkeypatch.setenv("VISIONKIT_OLLAMA_MODEL", "gemma4:e4b")
    monkeypatch.delenv("OLLAMA_RECEIPT_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL_NAME", raising=False)

    transaction = ingest_receipts._process_receipt(
        receipt_path, pipeline="visionkit_ollama", cache=None
    )

    assert transaction.payee == "Coffee Shop"
    assert transaction.transaction_date == date(2026, 5, 9)
    assert transaction.amount == "-7.50"
    assert transaction.ingestion_filename == "receipt.png"
    assert transaction.ingestion_type == "receipt_img"
    assert transaction.record_type == "receipt_based"
    assert transaction.receipt is not None
    assert transaction.receipt.extraction is not None
    assert transaction.receipt.extraction.model == "ocrmac+gemma4:e4b"
    assert transaction.receipt.extraction.confidence == 0.7
    assert transaction.receipt.extraction.raw_text == "Coffee Shop\nLatte 7.50"
    assert transaction.receipt.items[0].description == "Latte"
    assert transaction.receipt.items[0].discount_amount is None
    assert requests[0]["model"] == "gemma4:e4b"
    assert requests[0]["think"] is True
    assert requests[1]["prompt"] == (
        "Extract receipt data from this OCR text.\n\n"
        "Put item-level instant savings in the item discount.\n\n"
        "OCR:\n"
        "Coffee Shop\n"
        "Latte 7.50"
    )
    assert requests[1]["options"] == {"temperature": 0}
    assert requests[1]["output_format"] == ingest_receipts._receipt_ollama_output_schema()


def test_receipt_data_from_ollama_lines_writes_pretty_response_to_prompt_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prompt_log_path = tmp_path / "logs" / "ollama-prompts.log"

    class FakeOllamaClient:
        def __init__(self, *, url: str, model: str, timeout_seconds: float, think: bool) -> None:
            pass

        def complete_structured(
            self,
            prompt: str,
            *,
            options: dict[str, object],
            output_format: dict[str, object],
        ) -> str:
            del prompt, options, output_format
            return json.dumps(
                {
                    "analysis": "Latte has no associated discount line.",
                    "merchantName": "Coffee Shop",
                    "transactionDate": "2026-05-09",
                    "subtotal": "7.50",
                    "tax": "0.00",
                    "total": "7.50",
                    "items": [{"description": "Latte", "amount": "7.50", "discount": "0.00"}],
                },
                separators=(",", ":"),
            )

    monkeypatch.setattr(ingest_receipts, "UrlLibOllamaClient", FakeOllamaClient)
    monkeypatch.setenv("RECEIPTS_AI_OLLAMA_PROMPT_LOG", str(prompt_log_path))

    ingest_receipts._receipt_data_from_ollama_lines(
        ["Coffee Shop", "Latte 7.50"],
        raw_text="Coffee Shop\nLatte 7.50",
        model="gemma4:e4b",
        cache=None,
    )

    content = prompt_log_path.read_text(encoding="utf-8")
    assert "OLLAMA RECEIPT RESPONSE " in content
    assert "cached: false" in content
    assert '"merchantName": "Coffee Shop"' in content
    assert '"items": [\n    {\n      "amount": "7.50"' in content


def test_receipt_data_from_ollama_response_maps_discount_schema_to_extraction_items():
    receipt_data = ingest_receipts._receipt_data_from_ollama_response(
        json.dumps(
            {
                "analysis": "Granola matched instant savings 1.25.",
                "merchantName": "Market",
                "transactionDate": "2026-05-09",
                "items": [
                    {"description": "Granola", "amount": "6.99", "discount": "1.25"},
                    {"description": "Milk", "amount": "3.50", "discount": "0.00"},
                ],
                "subtotal": "10.49",
                "tax": "0.45",
                "total": "9.69",
            }
        ),
        raw_text="Market\nGranola 6.99\nInstant Savings -1.25",
    )

    assert receipt_data.merchant_name == "Market"
    assert receipt_data.subtotal == "10.49"
    assert receipt_data.total_tax == "0.45"
    assert receipt_data.currency == "USD"
    assert receipt_data.items[0].discount_amount == "-1.25"
    assert receipt_data.items[1].discount_amount is None
    assert [(item.description, item.amount, item.line_type) for item in receipt_data.items] == [
        ("Granola", "6.99", LineType.item),
        ("Milk", "3.50", LineType.item),
        ("Sales tax", "0.45", LineType.tax),
    ]


def test_receipt_data_from_ollama_response_omits_zero_tax_line_item():
    receipt_data = ingest_receipts._receipt_data_from_ollama_response(
        json.dumps(
            {
                "analysis": "Latte has no associated discount line.",
                "merchantName": "Coffee Shop",
                "transactionDate": "2026-05-09",
                "items": [{"description": "Latte", "amount": "7.50", "discount": "0.00"}],
                "subtotal": "7.50",
                "tax": "0.00",
                "total": "7.50",
            }
        ),
        raw_text="Coffee Shop\nLatte 7.50",
    )

    assert [(item.description, item.amount, item.line_type) for item in receipt_data.items] == [
        ("Latte", "7.50", LineType.item),
    ]


def test_receipt_data_from_ollama_response_rejects_implausible_date_year():
    with pytest.raises(ValueError, match="transactionDate"):
        ingest_receipts._receipt_data_from_ollama_response(
            json.dumps(
                {
                    "analysis": "Latte has no associated discount line.",
                    "merchantName": "Coffee Shop",
                    "transactionDate": "0420-06-20",
                    "items": [{"description": "Latte", "amount": "7.50", "discount": "0.00"}],
                    "subtotal": "7.50",
                    "tax": "0.00",
                    "total": "7.50",
                }
            ),
            raw_text="Coffee Shop\nLatte 7.50",
        )


def test_receipt_ollama_output_schema_requires_modern_transaction_date_year():
    schema = ingest_receipts._receipt_ollama_output_schema()

    properties = cast(dict[str, object], schema["properties"])
    transaction_date_schema = cast(dict[str, object], properties["transactionDate"])

    assert transaction_date_schema["pattern"] == "^[12][0-9]{3}-[0-9]{2}-[0-9]{2}$"


def test_visionkit_ollama_pipeline_can_disable_thinking(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VISIONKIT_OLLAMA_THINK", "false")
    monkeypatch.delenv("VISIONKIT_OLLAMA_THINKING", raising=False)
    monkeypatch.delenv("OLLAMA_RECEIPT_THINK", raising=False)
    monkeypatch.delenv("OLLAMA_THINK", raising=False)

    assert ingest_receipts._receipt_ollama_think() is False


def test_visionkit_ollama_pipeline_passes_debug_image_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image")
    debug_image_path = tmp_path / "debug.png"
    debug_paths: list[Path | None] = []

    def fake_text_observations(
        _path: Path,
        *,
        debug_image_path: Path | None = None,
    ) -> list[ingest_receipts._VisionKitTextObservation]:
        debug_paths.append(debug_image_path)
        return [
            ingest_receipts._VisionKitTextObservation(
                text="Coffee Shop", confidence=0.8, x=0.1, y=0.8, width=0.5, height=0.1
            ),
            ingest_receipts._VisionKitTextObservation(
                text="Latte 7.50", confidence=0.6, x=0.1, y=0.6, width=0.5, height=0.1
            ),
        ]

    monkeypatch.setattr(ingest_receipts, "_visionkit_text_observations", fake_text_observations)
    monkeypatch.setattr(
        ingest_receipts,
        "_receipt_data_from_ollama_lines",
        lambda _lines, *, raw_text, model, cache: ReceiptDataExtraction(
            merchant_name="Coffee Shop",
            transaction_date=date(2026, 5, 9),
            currency="USD",
            total="7.50",
            items=[ExtractedReceiptItem(description="Latte", amount="7.50")],
            extraction=ReceiptDataExtractionMetadata(
                pipeline=ReceiptExtractionPipeline.visionkit_ollama,
                raw_text=raw_text,
            ),
        ),
    )

    ingest_receipts._process_receipt(
        receipt_path,
        pipeline="visionkit_ollama",
        cache=None,
        visionkit_ocr_debug_image_path=debug_image_path,
    )

    assert debug_paths == [debug_image_path]


def test_main_visionkit_ocr_debug_image_folder_supports_multiple_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    receipt_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for receipt_path in receipt_paths:
        receipt_path.write_bytes(b"image")
    debug_folder = tmp_path / "debug"
    debug_paths: list[Path | None] = []

    def fake_process_receipt(
        receipt_path: Path,
        *,
        pipeline: str,
        cache: object,
        visionkit_ocr_debug_image_path: Path | None = None,
    ) -> Transaction:
        assert pipeline == "visionkit_ollama"
        assert cache is None
        debug_paths.append(visionkit_ocr_debug_image_path)
        return Transaction(
            id=f"receipt_{receipt_path.stem}",
            source=Source.receipt,
            transaction_date=date(2026, 5, 9),
            payee=receipt_path.stem,
            amount="-1.00",
            currency="USD",
            receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="1.00")]),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--pipeline",
            "visionkit_ollama",
            "--visionkit-ocr-debug-image-folder",
            str(debug_folder),
            str(receipt_paths[0]),
            str(receipt_paths[1]),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "_process_receipt", fake_process_receipt)

    main()

    assert capsys.readouterr().out
    assert debug_paths == [
        debug_folder / "first.visionkit-ocr-debug.png",
        debug_folder / "second.visionkit-ocr-debug.png",
    ]


def test_visionkit_pdf_ocr_image_path_renders_single_page_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pdf_path = tmp_path / "receipt.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    calls: list[tuple[Path, Path]] = []

    def fake_render(input_path: Path, output_path: Path) -> None:
        calls.append((input_path, output_path))
        output_path.write_bytes(b"png")

    monkeypatch.setattr(ingest_receipts, "_pdf_page_count", lambda _path: 1)
    monkeypatch.setattr(ingest_receipts, "_render_pdf_first_page_to_png", fake_render)

    with ingest_receipts._visionkit_ocr_image_path(pdf_path) as image_path:
        assert image_path.name == "page-1.png"
        assert image_path.read_bytes() == b"png"

    assert calls == [(pdf_path, calls[0][1])]
    assert not calls[0][1].exists()


def test_visionkit_pdf_ocr_image_path_rejects_multi_page_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pdf_path = tmp_path / "receipt.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(ingest_receipts, "_pdf_page_count", lambda _path: 2)

    with pytest.raises(ValueError, match="only supports single-page PDFs"):
        with ingest_receipts._visionkit_ocr_image_path(pdf_path):
            pass


def test_visionkit_text_observations_ocr_pdf_rendered_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pdf_path = tmp_path / "receipt.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    ocr_paths: list[Path] = []
    debug_calls: list[tuple[Path, Path, list[ingest_receipts._VisionKitTextObservation]]] = []

    def fake_render(_input_path: Path, output_path: Path) -> None:
        output_path.write_bytes(b"png")

    class FakeOCR:
        def __init__(self, image: str, *, recognition_level: str) -> None:
            assert recognition_level == "accurate"
            ocr_paths.append(Path(image))

        def recognize(self) -> list[tuple[str, float, tuple[float, float, float, float]]]:
            return [("Coffee", 0.9, (0.1, 0.2, 0.3, 0.4))]

    monkeypatch.setattr(ingest_receipts, "_pdf_page_count", lambda _path: 1)
    monkeypatch.setattr(ingest_receipts, "_render_pdf_first_page_to_png", fake_render)
    monkeypatch.setattr(
        ingest_receipts,
        "_write_visionkit_ocr_debug_image",
        lambda image_path, output_path, observations: debug_calls.append(
            (image_path, output_path, observations)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ocrmac",
        SimpleNamespace(ocrmac=SimpleNamespace(OCR=FakeOCR)),
    )

    observations = ingest_receipts._visionkit_text_observations(pdf_path)

    assert [observation.text for observation in observations] == ["Coffee"]
    assert ocr_paths[0].name == "page-1.png"
    assert debug_calls == []

    debug_image_path = tmp_path / "debug.png"
    ingest_receipts._visionkit_text_observations(pdf_path, debug_image_path=debug_image_path)

    assert debug_calls[0][0].name == "page-1.png"
    assert debug_calls[0][1] == debug_image_path
    assert [observation.text for observation in debug_calls[0][2]] == ["Coffee"]


def test_write_visionkit_ocr_debug_image_draws_numbered_boxes_and_legend(tmp_path: Path):
    pytest.importorskip("PIL")
    from PIL import Image, ImageStat

    image_path = tmp_path / "receipt.png"
    output_path = tmp_path / "receipt.visionkit-ocr-debug.png"
    Image.new("RGB", (100, 80), "white").save(image_path)

    ingest_receipts._write_visionkit_ocr_debug_image(
        image_path,
        output_path,
        [
            ingest_receipts._VisionKitTextObservation(
                text="Coffee", confidence=0.9, x=0.1, y=0.2, width=0.3, height=0.2
            ),
            ingest_receipts._VisionKitTextObservation(
                text="7.50", confidence=0.8, x=0.6, y=0.5, width=0.2, height=0.1
            ),
        ],
    )

    with Image.open(output_path) as output_image:
        assert output_image.size[0] > 100
        assert output_image.size[1] == 80
        box_region = output_image.crop((8, 46, 42, 66))
        legend_region = output_image.crop((110, 10, output_image.size[0] - 4, 70))
        assert any(channel_min < 255 for channel_min, _ in ImageStat.Stat(box_region).extrema)
        assert any(channel_min < 255 for channel_min, _ in ImageStat.Stat(legend_region).extrema)


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
                discount_amount="-1.00",
                discount_description="/1779212",
                net_amount="6.00",
                category_id="Food & Dining > Fast Food & Coffee",
                taxonomy="Food, Beverages & Tobacco > Beverages",
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
            "transaction_description": "",
            "combined_description": "Coffee",
            "transaction_amount": "",
            "transaction_currency": "",
            "ingestion_datetime": "",
            "ingestion_filename": "",
            "ingestion_file_url": "",
            "ingestion_file_sha256_hex": "",
            "ingestion_type": "",
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
            "item_discount_amount": "-1.00",
            "item_discount_description": "/1779212",
            "item_net_amount": "6.00",
            "item_line_type": "item",
            "item_category_id": "Food & Dining > Fast Food & Coffee",
            "item_taxonomy_1": "Food, Beverages & Tobacco",
            "item_taxonomy_2": "Beverages",
            "item_taxonomy_3": "",
            "item_taxonomy_4": "",
            "item_taxonomy_5": "",
            "item_taxonomy_6": "",
            "item_taxonomy_7": "",
            "item_taxonomy_8": "",
            "item_taxonomy_9": "",
            "item_confidence": "",
        },
        {
            "transaction_id": "",
            "transaction_date": "",
            "payee": "",
            "transaction_description": "",
            "combined_description": "Bagel",
            "transaction_amount": "",
            "transaction_currency": "",
            "ingestion_datetime": "",
            "ingestion_filename": "",
            "ingestion_file_url": "",
            "ingestion_file_sha256_hex": "",
            "ingestion_type": "",
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
            "item_discount_amount": "",
            "item_discount_description": "",
            "item_net_amount": "3.00",
            "item_line_type": "item",
            "item_category_id": "",
            "item_taxonomy_1": "",
            "item_taxonomy_2": "",
            "item_taxonomy_3": "",
            "item_taxonomy_4": "",
            "item_taxonomy_5": "",
            "item_taxonomy_6": "",
            "item_taxonomy_7": "",
            "item_taxonomy_8": "",
            "item_taxonomy_9": "",
            "item_confidence": "0.91",
        },
    ]


def test_writes_transaction_fields_on_each_csv_receipt_item_row():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        ingestion_datetime=datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC),
        ingestion_filename="receipt.pdf",
        ingestion_file_url="file:///tmp/receipt.pdf",
        ingestion_file_sha256_hex="0" * 64,
        ingestion_type=IngestionType.receipt_img,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        description="",
        amount="-11.00",
        currency="USD",
        receipt=Receipt(
            subtotal="10.00",
            total="11.00",
            items=[
                ReceiptItem(
                    description="Coffee",
                    amount="7.00",
                    category_id="Food & Dining > Fast Food & Coffee",
                )
            ],
        ),
    )
    output = StringIO()

    write_transaction_receipt_items_csv(transaction, output)

    rows = list(csv.DictReader(StringIO(output.getvalue())))
    assert rows[0]["transaction_id"] == "receipt_1"
    assert rows[0]["transaction_date"] == "2026-04-27"
    assert rows[0]["payee"] == "Coffee Shop"
    assert rows[0]["transaction_description"] == ""
    assert rows[0]["combined_description"] == "Coffee"
    assert rows[0]["transaction_amount"] == "-11.00"
    assert rows[0]["transaction_currency"] == "USD"
    assert rows[0]["ingestion_datetime"] == "2026-05-06T07:08:09+00:00"
    assert rows[0]["ingestion_filename"] == "receipt.pdf"
    assert rows[0]["ingestion_file_url"] == "file:///tmp/receipt.pdf"
    assert rows[0]["ingestion_file_sha256_hex"] == "0" * 64
    assert rows[0]["ingestion_type"] == "receipt_img"
    assert rows[0]["item_description"] == "Coffee"
    assert "item_category_id" not in rows[0]
    assert rows[0]["category_allocation.category_id"] == ("Food & Dining > Fast Food & Coffee")
    assert rows[0]["category_allocation.amount"] == "-7.00"


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
        record_type=RecordType.receipt_based,
        ingestion_datetime=datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC),
        ingestion_filename="receipt.pdf",
        ingestion_file_url="file:///tmp/receipt.pdf",
        ingestion_file_sha256_hex="0" * 64,
        ingestion_type=IngestionType.receipt_img,
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


def test_transactions_json_output_writes_json_array():
    transactions = [
        Transaction(
            id="receipt_1",
            source=Source.receipt,
            transaction_date=date(2026, 4, 27),
            payee="Coffee Shop",
            amount="-7.00",
            currency="USD",
            receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
        ),
        Transaction(
            id="receipt_2",
            source=Source.receipt,
            transaction_date=date(2026, 4, 28),
            payee="Bakery",
            amount="-5.00",
            currency="USD",
            receipt=Receipt(items=[ReceiptItem(description="Bagel", amount="5.00")]),
        ),
    ]
    output = StringIO()

    write_transactions_json(transactions, output)

    payload = json.loads(output.getvalue())
    assert [transaction["id"] for transaction in payload] == ["receipt_1", "receipt_2"]
    assert payload[0]["receipt"]["items"][0]["description"] == "Coffee"
    assert payload[1]["receipt"]["items"][0]["description"] == "Bagel"


def test_transaction_firestore_document_uses_json_safe_aliases():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        ingestion_datetime=datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC),
        ingestion_filename="receipt.pdf",
        ingestion_file_url="file:///tmp/receipt.pdf",
        ingestion_file_sha256_hex="0" * 64,
        ingestion_type=IngestionType.receipt_img,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(
            total="7.00",
            items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")],
        ),
    )

    document = transaction_firestore_document(transaction)

    assert document["transactionDate"] == "2026-04-27"
    assert document["source"] == "receipt"
    assert document["recordType"] == "receipt_based"
    assert document["ingestionDatetime"] == "2026-05-06T07:08:09Z"
    assert document["ingestionFilename"] == "receipt.pdf"
    assert document["ingestionFileUrl"] == "file:///tmp/receipt.pdf"
    assert document["ingestionFileSha256Hex"] == "0" * 64
    assert document["ingestionType"] == "receipt_img"
    assert document["receipt"] == {
        "total": "7.00",
        "items": [
            {
                "description": "Coffee",
                "rawDescription": "COF",
                "amount": "7.00",
                "netAmount": "7.00",
                "lineType": "item",
            }
        ],
    }


def test_upsert_transaction_to_firestore_merges_transaction_document(
    caplog: pytest.LogCaptureFixture,
):
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    calls: list[tuple[str, str, dict[str, object], bool]] = []

    class FakeDocument:
        def __init__(self, collection: str, document_id: str) -> None:
            self.collection = collection
            self.document_id = document_id

        def set(self, document_data: dict[str, object], *, merge: bool = False) -> None:
            calls.append((self.collection, self.document_id, document_data, merge))

    class FakeCollection:
        def __init__(self, collection: str) -> None:
            self.collection = collection

        def document(self, document_id: str) -> FakeDocument:
            return FakeDocument(self.collection, document_id)

    class FakeFirestoreClient:
        def collection(self, collection_path: str) -> FakeCollection:
            return FakeCollection(collection_path)

    with caplog.at_level(logging.INFO, logger=ingest_receipts.__name__):
        upsert_transaction_to_firestore(
            transaction, client=FakeFirestoreClient(), collection="test-transactions"
        )

    assert calls == [
        (
            "test-transactions",
            "receipt_1",
            transaction_firestore_document(transaction),
            True,
        )
    ]
    assert (
        "Upserting transaction receipt_1 to Firestore collection test-transactions" in caplog.text
    )
    assert (
        "Firestore upsert completed for transaction receipt_1 in collection test-transactions"
        in caplog.text
    )


def test_create_firestore_client_sets_emulator_env_from_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    sentinel_client = object()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FIRESTORE_EMULATOR_HOST", raising=False)
    (tmp_path / ".receipts_ai.config").write_text(
        "\n".join(
            (
                "FIRESTORE_EMULATOR_HOST=127.0.0.1:8080",
                "FIREBASE_PROJECT_ID=receipts-ai-local",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        firestore_client,
        "_create_firestore_emulator_client",
        lambda: sentinel_client,
    )

    client = firestore_client.create_firestore_client()

    assert client is sentinel_client
    assert os.environ["FIRESTORE_EMULATOR_HOST"] == "127.0.0.1:8080"


def test_create_firestore_client_reads_selected_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    sentinel_client = object()
    home_path = tmp_path / "home"
    home_path.mkdir()
    config_path = tmp_path / "dev.receipts_ai.config"

    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("RECEIPTS_AI_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("FIRESTORE_EMULATOR_HOST", raising=False)
    (home_path / ".receipts_ai.config").write_text(
        "FIRESTORE_EMULATOR_HOST=home:8080\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            (
                "FIRESTORE_EMULATOR_HOST=127.0.0.1:8080",
                "FIREBASE_PROJECT_ID=receipts-ai-local",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        firestore_client,
        "_create_firestore_emulator_client",
        lambda: sentinel_client,
    )

    client = firestore_client.create_firestore_client()

    assert client is sentinel_client
    assert os.environ["FIRESTORE_EMULATOR_HOST"] == "127.0.0.1:8080"


def test_json_output_includes_receipt_item_category():
    receipt = Receipt(
        total="4.49",
        items=[
            ReceiptItem(
                description="Saltines",
                amount="4.49",
                category_id="Food & Dining > Groceries",
            )
        ],
    )
    output = StringIO()

    write_receipt_json(receipt, output)

    assert '"categoryId": "Food & Dining > Groceries"' in output.getvalue()


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
            "receipts-ai-ingest-receipts",
            "--brave-search",
            "--brave-search-delay-seconds",
            "1.1",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(
        ingest_receipts,
        "analyze_receipt_file",
        fake_analyze_receipt_file,
    )
    monkeypatch.setattr(
        ingest_receipts,
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

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction,
    ) -> Transaction:
        assert transaction_to_clean.receipt is not None
        assert transaction_to_clean.receipt.items[0].brave_search_result == "search payload"
        transaction_to_clean.receipt.items[0].description = "Coffee"
        return transaction_to_clean

    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )

    main()

    assert calls == [transaction]
    assert receipt.items[0].brave_search_result == "search payload"


def test_main_after_filters_receipts_before_enrichment_and_upsert(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    calls: list[str] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        _ = transaction_to_enrich, request_delay_seconds
        calls.append("enrich")
        return transaction_to_enrich

    def fake_upsert_transaction_to_firestore(
        transaction_to_upsert: Transaction, *, collection: str, apply_rules: bool = True
    ) -> None:
        assert apply_rules is False
        _ = transaction_to_upsert, collection
        calls.append("upsert")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--after",
            "2026-04-28",
            "--brave-search",
            "--upsert-firestore",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "upsert_transaction_to_firestore",
        fake_upsert_transaction_to_firestore,
    )

    main()

    assert calls == []
    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert rows == []


def test_main_processes_multiple_receipts_as_combined_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_1_path = tmp_path / "receipt-1.pdf"
    receipt_2_path = tmp_path / "receipt-2.pdf"
    receipt_1_path.write_bytes(b"receipt 1")
    receipt_2_path.write_bytes(b"receipt 2")
    analyze_calls: list[Path] = []

    def fake_analyze_receipt_file(path: Path) -> Path:
        analyze_calls.append(path)
        return path

    def fake_transaction_from_document_intelligence_result(result: object) -> Transaction:
        assert isinstance(result, Path)
        path = result
        receipt_number = "001" if path == receipt_1_path else "002"
        amount = "1.00" if path == receipt_1_path else "2.00"
        return Transaction(
            id=f"transaction_{receipt_number}",
            source=Source.receipt,
            transaction_date=date(2026, 4, 27),
            payee=f"Store {receipt_number}",
            amount=f"-{amount}",
            currency="USD",
            receipt=Receipt(
                receipt_number=receipt_number,
                items=[
                    ReceiptItem(
                        description=f"Item {receipt_number}",
                        amount=amount,
                    )
                ],
            ),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["receipts-ai-ingest-receipts", str(receipt_1_path), str(receipt_2_path)],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert analyze_calls == [receipt_1_path, receipt_2_path]
    assert [row["transaction_id"] for row in rows] == [
        ingest_receipts.receipt_image_transaction_id(hashlib.sha256(b"receipt 1").hexdigest()),
        ingest_receipts.receipt_image_transaction_id(hashlib.sha256(b"receipt 2").hexdigest()),
    ]
    assert [row["ingestion_file_sha256_hex"] for row in rows] == [
        hashlib.sha256(b"receipt 1").hexdigest(),
        hashlib.sha256(b"receipt 2").hexdigest(),
    ]
    assert [row["item_description"] for row in rows] == ["Item 001", "Item 002"]


def test_main_fails_when_review_db_has_no_reviewed_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"unreviewed receipt")
    review_db_path = tmp_path / "reviews.sqlite"

    def fail_analyze_receipt_file(_path: Path) -> object:
        raise AssertionError("unreviewed receipt should not be extracted")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--review-db",
            str(review_db_path),
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fail_analyze_receipt_file)

    with pytest.raises(ValueError, match="does not have reviewed data"):
        main()


def test_main_allow_unreviewed_runs_pipeline_when_review_db_has_no_reviewed_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"unreviewed receipt")
    review_db_path = tmp_path / "reviews.sqlite"
    analyze_calls: list[Path] = []

    def fake_analyze_receipt_file(path: Path) -> Path:
        analyze_calls.append(path)
        return path

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return Transaction(
            id="pipeline_id",
            source=Source.receipt,
            transaction_date=date(2026, 4, 27),
            payee="Pipeline Shop",
            amount="-3.00",
            currency="USD",
            receipt=Receipt(items=[ReceiptItem(description="Tea", amount="3.00")]),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--review-db",
            str(review_db_path),
            "--allow-unreviewed",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert analyze_calls == [receipt_path]
    assert [row["payee"] for row in rows] == ["Pipeline Shop"]


def test_receipt_image_transaction_id_is_pipeline_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"same receipt image")

    def transaction(transaction_id: str) -> Transaction:
        return Transaction(
            id=transaction_id,
            source=Source.receipt,
            transaction_date=date(2026, 4, 27),
            payee="Coffee Shop",
            amount="-7.00",
            currency="USD",
            receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
        )

    monkeypatch.setattr(
        ingest_receipts,
        "_transaction_from_azure_receipt",
        lambda _path, *, cache: transaction("azure_raw_text_id"),
    )
    monkeypatch.setattr(
        ingest_receipts,
        "_transaction_from_visionkit_ollama_receipt",
        lambda _path, *, cache, ocr_debug_image_path=None: transaction("visionkit_raw_text_id"),
    )

    azure_transaction = ingest_receipts._process_receipt(
        receipt_path,
        pipeline="azure",
        cache=None,
    )
    visionkit_transaction = ingest_receipts._process_receipt(
        receipt_path,
        pipeline="visionkit_ollama",
        cache=None,
    )

    expected_id = ingest_receipts.receipt_image_transaction_id(
        hashlib.sha256(b"same receipt image").hexdigest()
    )
    assert azure_transaction.id == expected_id
    assert visionkit_transaction.id == expected_id


def test_main_wraps_brave_search_client_when_cache_file_is_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    cache_path = tmp_path / "api-cache.sqlite"
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

    class FakeBraveClient:
        def search(self, query: str) -> dict[str, object]:
            return {"query": query}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_analyze_receipt_file(path: Path, *, cache: object) -> dict[str, Path]:
        assert cache.__class__.__name__ == "SqliteCallCache"
        return {"result": path}

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction,
        *,
        client: object,
        request_delay_seconds: float | None = None,
    ) -> Transaction:
        assert request_delay_seconds is None
        assert client.__class__.__name__ == "CachedBraveSearchClient"
        assert transaction_to_enrich.receipt is not None
        transaction_to_enrich.receipt.items[0].brave_search_result = "search payload"
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction, *, client: object
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedCategoryModelClient"
        assert transaction_to_clean.receipt is not None
        assert transaction_to_clean.receipt.items[0].brave_search_result == "search payload"
        return transaction_to_clean

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--brave-search",
            "--cache-file",
            str(cache_path),
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(ingest_receipts, "create_brave_search_client", lambda: FakeBraveClient())
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )

    main()

    assert receipt.items[0].brave_search_result == "search payload"


def test_main_wraps_ollama_client_when_cache_file_is_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    cache_path = tmp_path / "api-cache.sqlite"
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )

    class FakeBraveClient:
        def search(self, query: str) -> dict[str, object]:
            return {"query": query}

    class FakeCategoryClient:
        def complete(self, prompt: str) -> str:
            return "Food & Dining" if "top level" in prompt else "Fast Food & Coffee"

    def fake_analyze_receipt_file(path: Path, *, cache: object) -> dict[str, Path]:
        assert cache.__class__.__name__ == "SqliteCallCache"
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction,
        *,
        client: object,
        request_delay_seconds: float | None = None,
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedBraveSearchClient"
        assert request_delay_seconds is None
        return transaction_to_enrich

    def fake_categorize_receipt_items(
        transaction_to_categorize: Transaction, *, client: object
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedCategoryModelClient"
        assert transaction_to_categorize.receipt is not None
        transaction_to_categorize.receipt.items[
            0
        ].category_id = "Food & Dining > Fast Food & Coffee"
        return transaction_to_categorize

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction, *, client: object
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedCategoryModelClient"
        assert transaction_to_clean.receipt is not None
        transaction_to_clean.receipt.items[0].description = "Coffee"
        return transaction_to_clean

    def fake_classify_receipt_items_by_product_taxonomy(
        transaction_to_classify: Transaction, *, client: object
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedCategoryModelClient"
        assert transaction_to_classify.receipt is not None
        transaction_to_classify.receipt.items[0].taxonomy = "Food, Beverages & Tobacco"
        return transaction_to_classify

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--categorize-items",
            "--cache-file",
            str(cache_path),
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(ingest_receipts, "create_brave_search_client", lambda: FakeBraveClient())
    monkeypatch.setattr(
        ingest_receipts, "create_ollama_category_client", lambda: FakeCategoryClient()
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(ingest_receipts, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy",
        fake_classify_receipt_items_by_product_taxonomy,
    )

    main()

    assert transaction.receipt is not None
    assert transaction.receipt.items[0].category_id == "Food & Dining > Fast Food & Coffee"
    assert transaction.receipt.items[0].taxonomy == "Food, Beverages & Tobacco"


def test_main_can_categorize_items_after_brave_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(
        items=[ReceiptItem(description="Saltines", raw_description="NBSC SALTINE", amount="4.49")]
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=receipt,
    )
    calls: list[str] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        assert request_delay_seconds == 0.25
        calls.append("brave")
        assert transaction_to_enrich.receipt is not None
        transaction_to_enrich.receipt.items[0].brave_search_result = "search payload"
        return transaction_to_enrich

    def fake_categorize_receipt_items(transaction_to_categorize: Transaction) -> Transaction:
        calls.append("categorize")
        assert transaction_to_categorize.receipt is not None
        assert transaction_to_categorize.receipt.items[0].brave_search_result == "search payload"
        assert transaction_to_categorize.receipt.items[0].description == "Nabisco Saltine Crackers"
        transaction_to_categorize.receipt.items[0].category_id = "Food & Dining > Groceries"
        return transaction_to_categorize

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction,
    ) -> Transaction:
        calls.append("clean")
        assert transaction_to_clean.receipt is not None
        assert transaction_to_clean.receipt.items[0].brave_search_result == "search payload"
        transaction_to_clean.receipt.items[0].description = "Nabisco Saltine Crackers"
        return transaction_to_clean

    def fake_classify_receipt_items_by_product_taxonomy(
        transaction_to_classify: Transaction,
    ) -> Transaction:
        calls.append("taxonomy")
        assert transaction_to_classify.receipt is not None
        assert transaction_to_classify.receipt.items[0].brave_search_result == "search payload"
        transaction_to_classify.receipt.items[0].taxonomy = "Food, Beverages & Tobacco"
        return transaction_to_classify

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--categorize-items",
            "--brave-search-delay-seconds",
            "0.25",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(ingest_receipts, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy",
        fake_classify_receipt_items_by_product_taxonomy,
    )

    main()

    assert calls == ["brave", "clean", "categorize", "taxonomy"]
    assert receipt.items[0].category_id == "Food & Dining > Groceries"
    assert receipt.items[0].taxonomy == "Food, Beverages & Tobacco"


def test_main_categorizes_items_with_flattened_budget_categories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(items=[ReceiptItem(description="Saltines", amount="4.49")])
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=receipt,
    )

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        _ = request_delay_seconds
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(transaction_to_clean: Transaction) -> Transaction:
        return transaction_to_clean

    def fake_categorize_receipt_items(transaction_to_categorize: Transaction) -> Transaction:
        assert transaction_to_categorize.receipt is not None
        transaction_to_categorize.receipt.items[0].category_id = "Food & Dining > Groceries"
        return transaction_to_categorize

    def fake_classify_receipt_items_by_product_taxonomy(
        transaction_to_classify: Transaction,
    ) -> Transaction:
        return transaction_to_classify

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--categorize-items",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(ingest_receipts, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy",
        fake_classify_receipt_items_by_product_taxonomy,
    )

    main()

    assert receipt.items[0].category_id == "Food & Dining > Groceries"


def test_main_can_use_vector_product_taxonomy_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(items=[ReceiptItem(description="Apple AirPods Pro 3", amount="249.00")])
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Apple",
        amount="-249.00",
        currency="USD",
        receipt=receipt,
    )
    calls: list[str] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        _ = request_delay_seconds
        calls.append("brave")
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(transaction_to_clean: Transaction) -> Transaction:
        calls.append("clean")
        return transaction_to_clean

    def fake_categorize_receipt_items(transaction_to_categorize: Transaction) -> Transaction:
        calls.append("categorize")
        assert transaction_to_categorize.receipt is not None
        transaction_to_categorize.receipt.items[0].category_id = "Shopping & Retail > Electronics"
        return transaction_to_categorize

    def fake_classify_receipt_items_by_product_taxonomy_vector_search(
        transaction_to_classify: Transaction,
    ) -> Transaction:
        calls.append("vector-taxonomy")
        assert transaction_to_classify.receipt is not None
        transaction_to_classify.receipt.items[0].taxonomy = "Electronics > Audio > Headphones"
        return transaction_to_classify

    def fail_greedy_taxonomy(_transaction: Transaction) -> Transaction:
        raise AssertionError("greedy taxonomy should not be called")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--categorize-items",
            "--product-taxonomy-method",
            "vector",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(ingest_receipts, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy",
        fail_greedy_taxonomy,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy_vector_search",
        fake_classify_receipt_items_by_product_taxonomy_vector_search,
    )

    main()

    assert calls == ["brave", "clean", "categorize", "vector-taxonomy"]
    assert receipt.items[0].category_id == "Shopping & Retail > Electronics"
    assert receipt.items[0].taxonomy == "Electronics > Audio > Headphones"


def test_main_can_upsert_processed_transaction_to_firestore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    upserts: list[tuple[Transaction, str]] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        assert request_delay_seconds is None
        assert transaction_to_enrich.receipt is not None
        transaction_to_enrich.receipt.items[0].brave_search_result = "search payload"
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction,
    ) -> Transaction:
        assert transaction_to_clean.receipt is not None
        transaction_to_clean.receipt.items[0].description = "Clean Coffee"
        return transaction_to_clean

    def fake_upsert_transaction_to_firestore(
        transaction_to_upsert: Transaction, *, collection: str, apply_rules: bool = True
    ) -> None:
        assert apply_rules is False
        upserts.append((transaction_to_upsert, collection))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--brave-search",
            "--upsert-firestore",
            "--firestore-collection",
            "processed-transactions",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "upsert_transaction_to_firestore",
        fake_upsert_transaction_to_firestore,
    )

    main()

    assert upserts == [(transaction, "processed-transactions")]
    assert transaction.receipt is not None
    assert transaction.receipt.items[0].description == "Clean Coffee"
    assert transaction.receipt.items[0].brave_search_result == "search payload"
