from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Protocol, TextIO, cast

from pydantic import BaseModel, ConfigDict, Field

from receipts_ai.brave_search import (
    CachedBraveSearchClient,
    create_brave_search_client,
    enrich_receipt_items_with_brave_search,
)
from receipts_ai.cache import SqliteCallCache
from receipts_ai.categorization import (
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_OLLAMA_URL,
    OLLAMA_PROMPT_LOG_ENV_VARS,
    OLLAMA_TIMEOUT_ENV_VARS,
    OLLAMA_URL_ENV_VARS,
    CachedCategoryModelClient,
    CategoryModelClient,
    UrlLibOllamaClient,
    categorize_receipt_items,
    classify_receipt_items_by_product_taxonomy,
    classify_receipt_items_by_product_taxonomy_vector_search,
    clean_receipt_item_descriptions,
    create_ollama_category_client,
)
from receipts_ai.config import first_config_value
from receipts_ai.document_intelligence import analyze_receipt_file
from receipts_ai.firestore_client import (
    DEFAULT_FIRESTORE_COLLECTION,
    FirestoreClient,
    create_firestore_client,
)
from receipts_ai.models.receipt_data_extraction import (
    ExtractedReceiptItem,
    ReceiptDataExtraction,
    ReceiptDataExtractionMetadata,
    ReceiptExtractionPipeline,
)
from receipts_ai.models.transaction import IngestionType, Receipt, RecordType, Transaction
from receipts_ai.receipt_extraction import (
    transaction_from_document_intelligence_result,
    transaction_from_receipt_data,
)
from receipts_ai.taxonomy import (
    MAX_TAXONOMY_LEVELS,
    effective_receipt_item_taxonomy,
    effective_transaction_taxonomy,
    split_taxonomy_path,
)
from receipts_ai.transactions import transaction_combined_description

RECEIPT_PIPELINES: tuple[str, ...] = ("azure", "visionkit_ollama")
DEFAULT_RECEIPT_OLLAMA_MODEL = "gemma4:e4b"
RECEIPT_OLLAMA_MODEL_ENV_VARS = (
    "VISIONKIT_OLLAMA_MODEL",
    "OLLAMA_RECEIPT_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_MODEL_NAME",
)
VISIONKIT_OLLAMA_CACHE_VERSION = "visionkit_ollama_receipt_v1"
CSV_FIELDNAMES: tuple[str, ...] = (
    "transaction_id",
    "transaction_date",
    "payee",
    "transaction_description",
    "combined_description",
    "transaction_amount",
    "transaction_currency",
    "ingestion_datetime",
    "ingestion_filename",
    "ingestion_file_url",
    "ingestion_file_sha256_hex",
    "ingestion_type",
    "receipt_id",
    "source_document_id",
    "receipt_number",
    "receipt_subtotal",
    "receipt_total",
    "extraction_model",
    "extraction_confidence",
    "item_index",
    "item_id",
    "item_description",
    "item_raw_description",
    "item_brave_search_result",
    "item_quantity",
    "item_unit_price",
    "item_amount",
    "item_discount_amount",
    "item_discount_description",
    "item_net_amount",
    "item_line_type",
    "item_category_id",
    "item_taxonomy_1",
    "item_taxonomy_2",
    "item_taxonomy_3",
    "item_taxonomy_4",
    "item_taxonomy_5",
    "item_taxonomy_6",
    "item_taxonomy_7",
    "item_taxonomy_8",
    "item_taxonomy_9",
    "item_confidence",
)
TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES: tuple[str, ...] = tuple(
    fieldname
    for fieldname in CSV_FIELDNAMES
    if fieldname not in {"item_brave_search_result", "item_category_id"}
) + (
    "category_allocation.category_id",
    "category_allocation.amount",
)

LOGGER = logging.getLogger(__name__)


class _OllamaReceiptDataExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    merchant_name: Annotated[str, Field(alias="merchantName", min_length=1)]
    transaction_date: Annotated[date, Field(alias="transactionDate")]
    currency: Annotated[str, Field(pattern="^[A-Z]{3}$")] = "USD"
    receipt_number: Annotated[str | None, Field(alias="receiptNumber", min_length=1)] = None
    subtotal: str | None = None
    total_tax: Annotated[str | None, Field(alias="totalTax")] = None
    total: str
    items: Annotated[list[ExtractedReceiptItem], Field(min_length=1)]


class _OcrMacOCR(Protocol):
    def recognize(self, px: bool = False) -> list[object]: ...


@dataclass(frozen=True)
class _VisionKitTextObservation:
    text: str
    confidence: float | None
    x: float
    y: float
    width: float
    height: float

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class _VisionKitTextLine:
    observations: list[_VisionKitTextObservation]

    @property
    def center_y(self) -> float:
        return sum(observation.center_y for observation in self.observations) / len(
            self.observations
        )

    @property
    def height(self) -> float:
        return max(observation.height for observation in self.observations)

    @property
    def text(self) -> str:
        return " ".join(
            observation.text for observation in sorted(self.observations, key=lambda item: item.x)
        )


_VISIONKIT_LINE_MIN_HEIGHT_RATIO = 0.75
_VISIONKIT_LINE_MAX_HEIGHT_RATIO = 1.5


def _visionkit_observation_matches_line(
    observation: _VisionKitTextObservation,
    line: _VisionKitTextLine,
) -> bool:
    return (
        abs(observation.center_y - line.center_y) <= line.height / 2
        and observation.height >= line.height * _VISIONKIT_LINE_MIN_HEIGHT_RATIO
        and observation.height <= line.height * _VISIONKIT_LINE_MAX_HEIGHT_RATIO
    )


def _visionkit_text_observation_summary(observation: _VisionKitTextObservation) -> str:
    confidence = (
        f"{observation.confidence:.3f}" if observation.confidence is not None else "unknown"
    )
    return (
        f"text={observation.text!r} conf={confidence} "
        f"bbox=(x={observation.x:.4f}, y={observation.y:.4f}, "
        f"w={observation.width:.4f}, h={observation.height:.4f}) "
        f"center_y={observation.center_y:.4f}"
    )


def _visionkit_text_line_summary(line: _VisionKitTextLine) -> str:
    return (
        f"text={line.text!r} obs={len(line.observations)} "
        f"center_y={line.center_y:.4f} height={line.height:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one or more receipts."
    )
    parser.add_argument(
        "receipts",
        metavar="receipt",
        nargs="+",
        type=Path,
        help="Path to a receipt image or PDF. Provide multiple paths to process them together.",
    )
    parser.add_argument(
        "--pipeline",
        choices=RECEIPT_PIPELINES,
        default="azure",
        help="Receipt extraction pipeline to use. Defaults to azure.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format. CSV writes one row per receipt item; JSON preserves the nested struct.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write output to a file instead of stdout.",
    )
    parser.add_argument(
        "--after",
        type=parse_after_date,
        help="Ignore transactions before this date. Use YYYY-MM-DD.",
    )
    parser.add_argument(
        "--brave-search",
        action="store_true",
        help=(
            "Populate each receipt item braveSearchResult with Brave Search summaries "
            "and clean item descriptions with Ollama."
        ),
    )
    parser.add_argument(
        "--brave-search-delay-seconds",
        type=float,
        help="Sleep this many seconds between Brave Search requests.",
    )
    parser.add_argument(
        "--categorize",
        "--categorize-items",
        action="store_true",
        dest="categorize_items",
        help=(
            "Use Ollama to populate each receipt item categoryId and product taxonomy "
            "from Brave Search results."
        ),
    )
    parser.add_argument(
        "--product-taxonomy-method",
        choices=("greedy", "vector"),
        default="greedy",
        help=(
            "Product taxonomy classification method used with --categorize. "
            "'greedy' walks the taxonomy tree with Ollama; 'vector' retrieves nearest "
            "taxonomy embedding paths and asks Ollama to rank them. Defaults to greedy."
        ),
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        help="Cache Azure Document Intelligence, Brave Search, and Ollama responses in this SQLite database.",
    )
    parser.add_argument(
        "--upsert-firestore",
        action="store_true",
        help=(
            "Upsert the processed transaction into Cloud Firestore. Set "
            "FIRESTORE_EMULATOR_HOST for a local emulator or "
            "FIREBASE_SERVICE_ACCT_KEY_FILEPATH for production."
        ),
    )
    parser.add_argument(
        "--firestore-collection",
        default=DEFAULT_FIRESTORE_COLLECTION,
        help=f"Firestore collection to upsert into. Defaults to {DEFAULT_FIRESTORE_COLLECTION}.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        help="Show logs at this level.",
    )
    parser.add_argument(
        "--ollama-prompt-log",
        type=Path,
        help=(
            "Append human-readable Ollama request traces to this file, including "
            "pretty-printed request properties, options, format, and prompt text."
        ),
    )
    parser.add_argument(
        "--visionkit-ocr-debug-image-folder",
        type=Path,
        help=(
            "When using --pipeline visionkit_ollama, write PNGs with OCR observation "
            "bounding boxes and numbered text legends to this folder."
        ),
    )
    args = parser.parse_args()
    if args.ollama_prompt_log is not None:
        os.environ[OLLAMA_PROMPT_LOG_ENV_VARS[0]] = str(args.ollama_prompt_log)
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(name)s:%(message)s")

    cache = SqliteCallCache(args.cache_file) if args.cache_file is not None else None
    category_client = (
        CachedCategoryModelClient(cache=cache, client_factory=create_ollama_category_client)
        if cache is not None
        else None
    )

    transactions: list[Transaction] = []
    for receipt_path in args.receipts:
        visionkit_ocr_debug_image_path = (
            _visionkit_ocr_debug_image_path(args.visionkit_ocr_debug_image_folder, receipt_path)
            if args.visionkit_ocr_debug_image_folder is not None
            else None
        )
        transaction = _process_receipt(
            receipt_path,
            pipeline=args.pipeline,
            cache=cache,
            visionkit_ocr_debug_image_path=visionkit_ocr_debug_image_path,
        )
        if transaction.receipt is None:
            raise ValueError(f"transaction from {receipt_path} does not contain a receipt")
        if not transaction_is_on_or_after(transaction, args.after):
            LOGGER.info(
                "Skipping transaction %s from %s dated %s before --after %s",
                transaction.id,
                receipt_path,
                transaction.transaction_date,
                args.after,
            )
            continue

        if args.brave_search or args.categorize_items:
            if cache is not None:
                enrich_receipt_items_with_brave_search(
                    transaction,
                    client=CachedBraveSearchClient(
                        cache=cache, client_factory=create_brave_search_client
                    ),
                    request_delay_seconds=args.brave_search_delay_seconds,
                )
            else:
                enrich_receipt_items_with_brave_search(
                    transaction, request_delay_seconds=args.brave_search_delay_seconds
                )
            if category_client is not None:
                clean_receipt_item_descriptions(transaction, client=category_client)
            else:
                clean_receipt_item_descriptions(transaction)
        if args.categorize_items:
            if category_client is not None:
                categorize_receipt_items(transaction, client=category_client)
                _classify_receipt_items_by_product_taxonomy(
                    transaction,
                    method=args.product_taxonomy_method,
                    client=category_client,
                )
            else:
                categorize_receipt_items(transaction)
                _classify_receipt_items_by_product_taxonomy(
                    transaction,
                    method=args.product_taxonomy_method,
                )
        if args.upsert_firestore:
            upsert_transaction_to_firestore(transaction, collection=args.firestore_collection)
        transactions.append(transaction)

    _write_transactions(transactions, output_format=args.format, output_path=args.output)


def _write_transactions(
    transactions: list[Transaction] | tuple[Transaction, ...],
    *,
    output_format: str,
    output_path: Path | None = None,
) -> None:
    if output_path is None:
        _write_transactions_to_file(transactions, output_format=output_format, file=sys.stdout)
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        _write_transactions_to_file(
            transactions,
            output_format=output_format,
            file=file,
        )


def _write_transactions_to_file(
    transactions: list[Transaction] | tuple[Transaction, ...],
    *,
    output_format: str,
    file: TextIO,
) -> None:
    if output_format == "csv":
        write_transactions_receipt_items_csv(transactions, file)
        return

    if output_format == "json":
        if len(transactions) == 1:
            write_transaction_json(transactions[0], file)
        else:
            write_transactions_json(transactions, file)
        return

    raise ValueError(f"unsupported output format: {output_format}")


def _classify_receipt_items_by_product_taxonomy(
    transaction: Transaction,
    *,
    method: str,
    client: CategoryModelClient | None = None,
) -> Transaction:
    if method == "greedy":
        return (
            classify_receipt_items_by_product_taxonomy(transaction, client=client)
            if client is not None
            else classify_receipt_items_by_product_taxonomy(transaction)
        )
    if method == "vector":
        return (
            classify_receipt_items_by_product_taxonomy_vector_search(
                transaction,
                client=client,
            )
            if client is not None
            else classify_receipt_items_by_product_taxonomy_vector_search(transaction)
        )
    raise ValueError(f"unsupported product taxonomy method: {method}")


def _process_receipt(
    receipt_path: Path,
    *,
    pipeline: str,
    cache: SqliteCallCache | None,
    visionkit_ocr_debug_image_path: Path | None = None,
) -> Transaction:
    if pipeline == "azure":
        return populate_transaction_ingestion_metadata(
            _transaction_from_azure_receipt(receipt_path, cache=cache),
            ingestion_filename=receipt_path.name,
            ingestion_file_url=file_url_from_path(receipt_path),
            ingestion_file_sha256_hex=sha256_hex(receipt_path.read_bytes()),
            ingestion_type=IngestionType.receipt_img,
        )

    if pipeline == "visionkit_ollama":
        return populate_transaction_ingestion_metadata(
            _transaction_from_visionkit_ollama_receipt(
                receipt_path,
                cache=cache,
                ocr_debug_image_path=visionkit_ocr_debug_image_path,
            ),
            ingestion_filename=receipt_path.name,
            ingestion_file_url=file_url_from_path(receipt_path),
            ingestion_file_sha256_hex=sha256_hex(receipt_path.read_bytes()),
            ingestion_type=IngestionType.receipt_img,
        )

    raise ValueError(f"unsupported receipt pipeline: {pipeline}")


def _transaction_from_azure_receipt(
    receipt_path: Path,
    *,
    cache: SqliteCallCache | None,
) -> Transaction:
    result = (
        analyze_receipt_file(receipt_path, cache=cache)
        if cache is not None
        else analyze_receipt_file(receipt_path)
    )
    return transaction_from_document_intelligence_result(result)


def _transaction_from_visionkit_ollama_receipt(
    receipt_path: Path,
    *,
    cache: SqliteCallCache | None,
    ocr_debug_image_path: Path | None = None,
) -> Transaction:
    observations = _visionkit_text_observations(
        receipt_path,
        debug_image_path=ocr_debug_image_path,
    )
    lines = _visionkit_text_lines(observations)
    LOGGER.info(
        "VisionKit OCR grouped %d observation(s) into %d line(s) for %s",
        len(observations),
        len(lines),
        receipt_path,
    )
    if not lines:
        raise ValueError(f"VisionKit OCR did not find text in {receipt_path}")

    raw_text = "\n".join(lines)
    model = _receipt_ollama_model()
    receipt_data = _receipt_data_from_ollama_lines(lines, raw_text=raw_text, model=model, cache=cache)
    receipt_data.extraction = ReceiptDataExtractionMetadata(
        pipeline=ReceiptExtractionPipeline.visionkit_ollama,
        model=f"ocrmac+{model}",
        confidence=_mean_confidence(observations),
        raw_text=raw_text,
    )
    return transaction_from_receipt_data(receipt_data)


def _visionkit_text_observations(
    receipt_path: Path,
    *,
    debug_image_path: Path | None = None,
) -> list[_VisionKitTextObservation]:
    try:
        from ocrmac import ocrmac
    except ImportError as error:
        raise RuntimeError(
            "The visionkit_ollama pipeline requires ocrmac. Install it with `uv add ocrmac`."
        ) from error

    with _visionkit_ocr_image_path(receipt_path) as image_path:
        LOGGER.info("Running VisionKit OCR for %s using image %s", receipt_path, image_path)
        ocr = cast(_OcrMacOCR, ocrmac.OCR(str(image_path), recognition_level="accurate"))
        annotations = ocr.recognize()
        observations: list[_VisionKitTextObservation] = []
        for annotation in annotations:
            observation = _visionkit_text_observation(annotation)
            if observation is not None:
                observations.append(observation)
        LOGGER.info(
            "VisionKit OCR produced %d text observation(s) for %s",
            len(observations),
            receipt_path,
        )
        for index, observation in enumerate(observations, start=1):
            LOGGER.debug(
                "VisionKit OCR observation %d/%d: %s",
                index,
                len(observations),
                _visionkit_text_observation_summary(observation),
            )
        if observations and debug_image_path is not None:
            _write_visionkit_ocr_debug_image(
                image_path,
                debug_image_path,
                observations,
            )
            LOGGER.info(
                "Wrote VisionKit OCR debug image with %d observation(s) to %s",
                len(observations),
                debug_image_path,
            )
        return observations


@contextmanager
def _visionkit_ocr_image_path(receipt_path: Path) -> Generator[Path, None, None]:
    if receipt_path.suffix.lower() != ".pdf":
        yield receipt_path
        return

    page_count = _pdf_page_count(receipt_path)
    if page_count == 0:
        raise ValueError(f"PDF receipt does not contain any pages: {receipt_path}")
    if page_count > 1:
        raise ValueError(
            f"visionkit_ollama only supports single-page PDFs for now; "
            f"{receipt_path} has {page_count} pages"
        )

    with TemporaryDirectory(prefix="receipts-ai-pdf-") as temp_dir:
        image_path = Path(temp_dir) / "page-1.png"
        _render_pdf_first_page_to_png(receipt_path, image_path)
        yield image_path


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        quartz = _quartz_module()
    except ImportError as error:
        raise RuntimeError("PDF input for visionkit_ollama requires PyObjC Quartz.") from error

    document = _quartz_pdf_document(quartz, pdf_path)
    return int(quartz.CGPDFDocumentGetNumberOfPages(document))


def _render_pdf_first_page_to_png(
    pdf_path: Path,
    output_path: Path,
    *,
    dpi: int = 200,
) -> None:
    try:
        quartz = _quartz_module()
    except ImportError as error:
        raise RuntimeError("PDF input for visionkit_ollama requires PyObjC Quartz.") from error

    document = _quartz_pdf_document(quartz, pdf_path)
    page = quartz.CGPDFDocumentGetPage(document, 1)
    if page is None:
        raise ValueError(f"PDF receipt does not contain page 1: {pdf_path}")

    media_box = quartz.CGPDFPageGetBoxRect(page, quartz.kCGPDFMediaBox)
    width_points = float(media_box.size.width)
    height_points = float(media_box.size.height)
    scale = dpi / 72
    width_pixels = max(1, round(width_points * scale))
    height_pixels = max(1, round(height_points * scale))

    color_space = quartz.CGColorSpaceCreateDeviceRGB()
    context = quartz.CGBitmapContextCreate(
        None,
        width_pixels,
        height_pixels,
        8,
        0,
        color_space,
        quartz.kCGImageAlphaPremultipliedLast,
    )
    if context is None:
        raise RuntimeError(f"Could not create bitmap context for PDF receipt: {pdf_path}")

    output_rect = quartz.CGRectMake(0, 0, width_pixels, height_pixels)
    quartz.CGContextSetRGBFillColor(context, 1, 1, 1, 1)
    quartz.CGContextFillRect(context, output_rect)
    quartz.CGContextScaleCTM(context, scale, scale)
    transform = quartz.CGPDFPageGetDrawingTransform(
        page,
        quartz.kCGPDFMediaBox,
        quartz.CGRectMake(0, 0, width_points, height_points),
        0,
        True,
    )
    quartz.CGContextConcatCTM(context, transform)
    quartz.CGContextDrawPDFPage(context, page)

    image = quartz.CGBitmapContextCreateImage(context)
    if image is None:
        raise RuntimeError(f"Could not render PDF receipt page: {pdf_path}")

    output_url = _quartz_file_url(quartz, output_path)
    destination = quartz.CGImageDestinationCreateWithURL(output_url, "public.png", 1, None)
    if destination is None:
        raise RuntimeError(f"Could not create PNG destination: {output_path}")
    quartz.CGImageDestinationAddImage(destination, image, None)
    if not quartz.CGImageDestinationFinalize(destination):
        raise RuntimeError(f"Could not write rendered PDF page to {output_path}")


def _quartz_module() -> Any:
    return importlib.import_module("Quartz")


def _quartz_pdf_document(quartz: Any, pdf_path: Path) -> Any:
    document = quartz.CGPDFDocumentCreateWithURL(_quartz_file_url(quartz, pdf_path))
    if document is None:
        raise ValueError(f"Could not open PDF receipt: {pdf_path}")
    return document


def _quartz_file_url(quartz: Any, path: Path) -> Any:
    path_bytes = str(path.resolve()).encode("utf-8")
    return quartz.CFURLCreateFromFileSystemRepresentation(
        None,
        path_bytes,
        len(path_bytes),
        False,
    )


def _visionkit_text_observation(annotation: object) -> _VisionKitTextObservation | None:
    if not isinstance(annotation, tuple | list):
        return None

    values = cast(tuple[object, ...] | list[object], annotation)
    if len(values) < 3:
        return None

    text = values[0]
    confidence = values[1]
    bounding_box = values[2]
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(bounding_box, tuple | list):
        return None

    box = cast(tuple[object, ...] | list[object], bounding_box)
    if len(box) < 4:
        return None

    x = _float_or_none(box[0])
    y = _float_or_none(box[1])
    width = _float_or_none(box[2])
    height = _float_or_none(box[3])
    if x is None or y is None or width is None or height is None or height <= 0:
        return None

    return _VisionKitTextObservation(
        text=text.strip(),
        confidence=_float_or_none(confidence),
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _visionkit_ocr_debug_image_path(output_folder: Path, receipt_path: Path) -> Path:
    return output_folder / f"{receipt_path.stem}.visionkit-ocr-debug.png"


def _write_visionkit_ocr_debug_image(
    image_path: Path,
    output_path: Path,
    observations: list[_VisionKitTextObservation],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError(
            "Writing VisionKit OCR debug images requires Pillow, which is installed with ocrmac."
        ) from error

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(image)
    image_width, image_height = image.size
    legend_entries = [
        f"{index}. {observation.text}" for index, observation in enumerate(observations, start=1)
    ]
    legend_text_width = max(
        (round(draw_probe.textlength(entry, font=font)) for entry in legend_entries),
        default=0,
    )
    legend_width = min(max(260, legend_text_width + 24), 520)
    legend_line_height = 16
    canvas_height = max(image_height, 32 + legend_line_height * len(legend_entries))
    canvas = Image.new("RGB", (image_width + legend_width, canvas_height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    box_color = (218, 45, 45)
    label_fill = (218, 45, 45)
    label_text_fill = (255, 255, 255)
    line_width = max(2, round(min(image_width, image_height) / 500))
    label_padding = max(2, line_width)

    for index, observation in enumerate(observations, start=1):
        left, top, right, bottom = _visionkit_observation_pixel_box(
            observation,
            image_width=image_width,
            image_height=image_height,
        )
        draw.rectangle((left, top, right, bottom), outline=box_color, width=line_width)

        label = str(index)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_left = left
        label_top = max(0, top - text_height - label_padding * 2)
        label_right = min(image_width, label_left + text_width + label_padding * 2)
        label_bottom = label_top + text_height + label_padding * 2
        draw.rectangle((label_left, label_top, label_right, label_bottom), fill=label_fill)
        draw.text(
            (label_left + label_padding, label_top + label_padding),
            label,
            fill=label_text_fill,
            font=font,
        )

    legend_x = image_width + 12
    legend_y = 12
    draw.text((legend_x, legend_y), "OCR observations", fill=(0, 0, 0), font=font)
    legend_y += 20
    for entry in legend_entries:
        draw.text((legend_x, legend_y), entry, fill=(0, 0, 0), font=font)
        legend_y += legend_line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _visionkit_observation_pixel_box(
    observation: _VisionKitTextObservation,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left = round(observation.x * image_width)
    top = round((1 - observation.y - observation.height) * image_height)
    right = round((observation.x + observation.width) * image_width)
    bottom = round((1 - observation.y) * image_height)
    return (
        max(0, min(image_width, left)),
        max(0, min(image_height, top)),
        max(0, min(image_width, right)),
        max(0, min(image_height, bottom)),
    )


def _visionkit_text_lines(observations: list[_VisionKitTextObservation]) -> list[str]:
    lines: list[_VisionKitTextLine] = []
    sorted_observations = sorted(observations, key=lambda item: item.center_y, reverse=True)
    LOGGER.debug(
        "Sorting %d VisionKit observation(s) by descending center_y for line grouping",
        len(sorted_observations),
    )
    for index, observation in enumerate(sorted_observations, start=1):
        LOGGER.debug(
            "Sorted observation %d/%d: %s",
            index,
            len(sorted_observations),
            _visionkit_text_observation_summary(observation),
        )
    for observation in sorted_observations:
        LOGGER.debug(
            "Grouping observation into line candidates: %s",
            _visionkit_text_observation_summary(observation),
        )
        for line_index, line in enumerate(lines, start=1):
            delta = abs(observation.center_y - line.center_y)
            threshold = line.height / 2
            height_ratio = observation.height / line.height
            match = _visionkit_observation_matches_line(observation, line)
            LOGGER.debug(
                "Comparing against line %d: %s delta_center_y=%.4f threshold=%.4f "
                "height_ratio=%.4f match=%s",
                line_index,
                _visionkit_text_line_summary(line),
                delta,
                threshold,
                height_ratio,
                match,
            )
        matching_line = next(
            (
                line
                for line in lines
                if _visionkit_observation_matches_line(observation, line)
            ),
            None,
        )
        if matching_line is None:
            new_line = _VisionKitTextLine(observations=[observation])
            lines.append(new_line)
            LOGGER.debug("Created new line: %s", _visionkit_text_line_summary(new_line))
        else:
            matching_line.observations.append(observation)
            LOGGER.debug(
                "Added observation to existing line: %s",
                _visionkit_text_line_summary(matching_line),
            )
    for index, line in enumerate(lines, start=1):
        sorted_line_observations = sorted(line.observations, key=lambda item: item.x)
        LOGGER.debug(
            "Final grouped line %d/%d: %s",
            index,
            len(lines),
            _visionkit_text_line_summary(line),
        )
        for observation_index, observation in enumerate(sorted_line_observations, start=1):
            LOGGER.debug(
                "  line %d observation %d/%d: %s",
                index,
                observation_index,
                len(sorted_line_observations),
                _visionkit_text_observation_summary(observation),
            )
    return [line.text for line in lines if line.text]


def _receipt_data_from_ollama_lines(
    lines: list[str],
    *,
    raw_text: str,
    model: str,
    cache: SqliteCallCache | None,
) -> ReceiptDataExtraction:
    schema = _OllamaReceiptDataExtraction.model_json_schema(by_alias=True)
    prompt = _visionkit_ollama_receipt_prompt(lines)
    request = {
        "version": VISIONKIT_OLLAMA_CACHE_VERSION,
        "model": model,
        "prompt": prompt,
        "format": schema,
    }
    if cache is not None:
        cached_response = cache.get("ollama_receipt_extraction", request)
        if isinstance(cached_response, str):
            LOGGER.info("Using cached Ollama receipt extraction: line_count=%s", len(lines))
            return _receipt_data_from_ollama_response(cached_response, raw_text=raw_text)

    response = UrlLibOllamaClient(
        url=_ollama_url(),
        model=model,
        timeout_seconds=_ollama_timeout_seconds(),
    ).complete_structured(
        prompt,
        options={"temperature": 0},
        output_format=schema,
    )
    if cache is not None:
        cache.set("ollama_receipt_extraction", request, response)
        LOGGER.info("Cached Ollama receipt extraction: line_count=%s", len(lines))
    return _receipt_data_from_ollama_response(response, raw_text=raw_text)


def _receipt_data_from_ollama_response(response: str, *, raw_text: str) -> ReceiptDataExtraction:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(f"Ollama receipt extraction response was not JSON: {response[:500]}") from error

    receipt_data = _OllamaReceiptDataExtraction.model_validate(payload)
    return ReceiptDataExtraction(
        merchant_name=receipt_data.merchant_name,
        transaction_date=receipt_data.transaction_date,
        currency=receipt_data.currency,
        receipt_number=receipt_data.receipt_number,
        subtotal=receipt_data.subtotal,
        total_tax=receipt_data.total_tax,
        total=receipt_data.total,
        items=[
            ExtractedReceiptItem(
                description=item.description,
                quantity=item.quantity,
                unit_price=item.unit_price,
                amount=item.amount,
                discount_amount=item.discount_amount,
                discount_description=item.discount_description,
                line_type=item.line_type,
                confidence=item.confidence,
            )
            for item in receipt_data.items
        ],
        extraction=ReceiptDataExtractionMetadata(
            pipeline=ReceiptExtractionPipeline.visionkit_ollama,
            model=None,
            confidence=None,
            raw_text=raw_text,
        ),
    )


def _visionkit_ollama_receipt_prompt(lines: list[str]) -> str:
    return (
        "Extract receipt JSON. Amounts are strings without currency. "
        "Dates YYYY-MM-DD. currency ISO. Optional unknown=null. "
        "items are purchase lines; include tax/tip/discount lines only when shown.\n"
        "OCR:\n"
        + "\n".join(lines)
    )


def _receipt_ollama_model() -> str:
    return first_config_value(RECEIPT_OLLAMA_MODEL_ENV_VARS, DEFAULT_RECEIPT_OLLAMA_MODEL) or (
        DEFAULT_RECEIPT_OLLAMA_MODEL
    )


def _ollama_url() -> str:
    return first_config_value(OLLAMA_URL_ENV_VARS, DEFAULT_OLLAMA_URL) or DEFAULT_OLLAMA_URL


def _ollama_timeout_seconds() -> float:
    for config_key in OLLAMA_TIMEOUT_ENV_VARS:
        value = first_config_value((config_key,))
        if value is None:
            continue
        try:
            timeout_seconds = float(value)
        except ValueError as error:
            raise RuntimeError(f"{config_key} must be a number") from error
        if timeout_seconds <= 0:
            raise RuntimeError(f"{config_key} must be greater than 0")
        return timeout_seconds
    return DEFAULT_OLLAMA_TIMEOUT_SECONDS


def _mean_confidence(observations: list[_VisionKitTextObservation]) -> float | None:
    confidences = [
        observation.confidence for observation in observations if observation.confidence is not None
    ]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def upsert_transaction_to_firestore(
    transaction: Transaction,
    *,
    client: FirestoreClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> None:
    if not collection:
        raise ValueError("collection must not be empty")

    firestore_client = client if client is not None else create_firestore_client()
    document = transaction_firestore_document(transaction)
    item_count = len(transaction.receipt.items) if transaction.receipt is not None else 0
    LOGGER.info(
        "Upserting transaction %s to Firestore collection %s with %d receipt item(s)",
        transaction.id,
        collection,
        item_count,
    )
    LOGGER.debug(
        "Firestore document %s/%s top-level fields: %s",
        collection,
        transaction.id,
        sorted(document),
    )
    try:
        firestore_client.collection(collection).document(transaction.id).set(document, merge=True)
    except Exception:
        LOGGER.exception(
            "Firestore upsert failed for transaction %s in collection %s",
            transaction.id,
            collection,
        )
        raise
    LOGGER.info(
        "Firestore upsert completed for transaction %s in collection %s",
        transaction.id,
        collection,
    )


def transaction_firestore_document(transaction: Transaction) -> dict[str, Any]:
    return transaction.model_dump(mode="json", by_alias=True, exclude_none=True)


def populate_transaction_ingestion_metadata(
    transaction: Transaction,
    *,
    ingestion_filename: str,
    ingestion_file_sha256_hex: str,
    ingestion_type: IngestionType,
    ingestion_file_url: str | None = None,
    ingestion_datetime: datetime | None = None,
) -> Transaction:
    transaction.ingestion_datetime = ingestion_datetime or datetime.now(UTC)
    transaction.ingestion_filename = ingestion_filename
    transaction.ingestion_file_url = ingestion_file_url
    transaction.ingestion_file_sha256_hex = ingestion_file_sha256_hex
    transaction.ingestion_type = ingestion_type
    if ingestion_type == IngestionType.receipt_img:
        transaction.record_type = RecordType.receipt_based
    return transaction


def file_url_from_path(path: Path) -> str:
    return path.resolve().as_uri()


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_after_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from error


def transaction_is_on_or_after(transaction: Transaction, after: date | None) -> bool:
    return after is None or transaction.transaction_date >= after


def filter_transactions_on_or_after(
    transactions: list[Transaction],
    after: date | None,
) -> list[Transaction]:
    if after is None:
        return transactions
    return [
        transaction
        for transaction in transactions
        if transaction_is_on_or_after(transaction, after)
    ]


def write_transaction_receipt_items_csv(transaction: Transaction, file: TextIO) -> None:
    write_transactions_receipt_items_csv([transaction], file)


def write_transactions_receipt_items_csv(
    transactions: list[Transaction] | tuple[Transaction, ...], file: TextIO
) -> None:
    writer = csv.DictWriter(file, fieldnames=TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(transaction_receipt_item_rows(transactions))


def transaction_receipt_item_rows(
    transactions: list[Transaction] | tuple[Transaction, ...],
) -> list[dict[str, object | None]]:
    return [
        row for transaction in transactions for row in _transaction_receipt_item_rows(transaction)
    ]


def write_transaction_json(transaction: Transaction, file: TextIO) -> None:
    file.write(transaction.model_dump_json(by_alias=True, indent=2, exclude_none=True))
    file.write("\n")


def write_transactions_json(
    transactions: list[Transaction] | tuple[Transaction, ...], file: TextIO
) -> None:
    file.write(
        json.dumps(
            [
                transaction.model_dump(mode="json", by_alias=True, exclude_none=True)
                for transaction in transactions
            ],
            indent=2,
        )
    )
    file.write("\n")


def write_receipt_items_csv(receipt: Receipt, file: TextIO) -> None:
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(_receipt_item_rows(receipt))


def write_receipt_json(receipt: Receipt, file: TextIO) -> None:
    file.write(receipt.model_dump_json(by_alias=True, indent=2, exclude_none=True))
    file.write("\n")


def _transaction_receipt_item_rows(
    transaction: Transaction,
) -> list[dict[str, object | None]]:
    combined_description = transaction_combined_description(transaction)
    transaction_taxonomy_parts = split_taxonomy_path(effective_transaction_taxonomy(transaction))
    transaction_fields: dict[str, object | None] = {
        "transaction_id": transaction.id,
        "transaction_date": transaction.transaction_date.isoformat(),
        "payee": transaction.payee,
        "transaction_description": transaction.description,
        "combined_description": combined_description,
        "transaction_amount": transaction.amount,
        "transaction_currency": transaction.currency,
        "ingestion_datetime": transaction.ingestion_datetime.isoformat()
        if transaction.ingestion_datetime is not None
        else None,
        "ingestion_filename": transaction.ingestion_filename,
        "ingestion_file_url": transaction.ingestion_file_url,
        "ingestion_file_sha256_hex": transaction.ingestion_file_sha256_hex,
        "ingestion_type": transaction.ingestion_type.value
        if transaction.ingestion_type is not None
        else None,
    }

    if transaction.receipt is not None:
        return _receipt_item_rows(
            transaction.receipt,
            transaction_id=transaction.id,
            transaction_date=transaction.transaction_date.isoformat(),
            payee=transaction.payee,
            transaction_description=transaction.description,
            transaction_amount=transaction.amount,
            transaction_currency=transaction.currency,
            ingestion_datetime=transaction.ingestion_datetime.isoformat()
            if transaction.ingestion_datetime is not None
            else None,
            ingestion_filename=transaction.ingestion_filename,
            ingestion_file_url=transaction.ingestion_file_url,
            ingestion_file_sha256_hex=transaction.ingestion_file_sha256_hex,
            ingestion_type=transaction.ingestion_type.value
            if transaction.ingestion_type is not None
            else None,
            include_brave_search_result=False,
            include_item_category_id=False,
            include_category_allocation=True,
        )

    category_allocations = transaction.category_allocations or []
    rows: list[dict[str, object | None]] = []
    for allocation in category_allocations or [None]:
        row = dict(transaction_fields)
        for index, value in enumerate(transaction_taxonomy_parts, start=1):
            row[f"item_taxonomy_{index}"] = value
        if allocation is not None:
            row["category_allocation.category_id"] = allocation.category_id
            row["category_allocation.amount"] = allocation.amount
        rows.append(row)
    return rows


def _receipt_item_rows(
    receipt: Receipt,
    *,
    transaction_id: str | None = None,
    transaction_date: str | None = None,
    payee: str | None = None,
    transaction_description: str | None = None,
    transaction_amount: str | None = None,
    transaction_currency: str | None = None,
    ingestion_datetime: str | None = None,
    ingestion_filename: str | None = None,
    ingestion_file_url: str | None = None,
    ingestion_file_sha256_hex: str | None = None,
    ingestion_type: str | None = None,
    include_brave_search_result: bool = True,
    include_item_category_id: bool = True,
    include_category_allocation: bool = False,
) -> list[dict[str, object | None]]:
    extraction = receipt.extraction
    rows: list[dict[str, object | None]] = []
    for index, item in enumerate(receipt.items, start=1):
        taxonomy_parts = split_taxonomy_path(effective_receipt_item_taxonomy(item))
        row: dict[str, object | None] = {
            "transaction_id": transaction_id,
            "transaction_date": transaction_date,
            "payee": payee,
            "transaction_description": transaction_description,
            "combined_description": item.description,
            "transaction_amount": transaction_amount,
            "transaction_currency": transaction_currency,
            "ingestion_datetime": ingestion_datetime,
            "ingestion_filename": ingestion_filename,
            "ingestion_file_url": ingestion_file_url,
            "ingestion_file_sha256_hex": ingestion_file_sha256_hex,
            "ingestion_type": ingestion_type,
            "receipt_id": receipt.id,
            "source_document_id": receipt.source_document_id,
            "receipt_number": receipt.receipt_number,
            "receipt_subtotal": receipt.subtotal,
            "receipt_total": receipt.total,
            "extraction_model": extraction.model if extraction is not None else None,
            "extraction_confidence": extraction.confidence if extraction is not None else None,
            "item_index": index,
            "item_id": item.id,
            "item_description": item.description,
            "item_raw_description": item.raw_description,
            "item_quantity": item.quantity,
            "item_unit_price": item.unit_price,
            "item_amount": item.amount,
            "item_discount_amount": item.discount_amount,
            "item_discount_description": item.discount_description,
            "item_net_amount": item.net_amount,
            "item_line_type": item.line_type.value if item.line_type is not None else None,
            "item_confidence": item.confidence,
        }
        for taxonomy_index in range(1, MAX_TAXONOMY_LEVELS + 1):
            row[f"item_taxonomy_{taxonomy_index}"] = taxonomy_parts[taxonomy_index - 1]
        if include_brave_search_result:
            row["item_brave_search_result"] = item.brave_search_result
        if include_item_category_id:
            row["item_category_id"] = item.category_id
        if include_category_allocation:
            row["category_allocation.category_id"] = item.category_id
            row["category_allocation.amount"] = _category_allocation_amount(
                item.net_amount,
                transaction_amount=transaction_amount,
            )
        rows.append(row)
    return rows


def _category_allocation_amount(
    item_amount: str | None,
    *,
    transaction_amount: str | None,
) -> str | None:
    if item_amount is None or transaction_amount is None:
        return item_amount

    try:
        item_decimal = Decimal(item_amount)
        transaction_decimal = Decimal(transaction_amount)
    except InvalidOperation:
        return item_amount

    if transaction_decimal < 0:
        return format(-item_decimal, "f")
    if transaction_decimal > 0:
        return format(item_decimal, "f")
    return format(Decimal("0"), "f")


if __name__ == "__main__":
    main()
