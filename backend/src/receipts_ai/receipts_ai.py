from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TextIO

from receipts_ai.document_intelligence import analyze_receipt_file
from receipts_ai.models.transaction import Receipt
from receipts_ai.receipt_extraction import receipt_from_document_intelligence_result

CSV_FIELDNAMES: tuple[str, ...] = (
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
    "item_line_type",
    "item_category_id",
    "item_confidence",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a receipt with Azure Document Intelligence."
    )
    parser.add_argument("receipt", type=Path, help="Path to a receipt image or PDF.")
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
    args = parser.parse_args()

    result = analyze_receipt_file(args.receipt)
    receipt = receipt_from_document_intelligence_result(result)
    _write_receipt(receipt, output_format=args.format, output_path=args.output)


def _write_receipt(
    receipt: Receipt, *, output_format: str, output_path: Path | None = None
) -> None:
    if output_path is None:
        _write_receipt_to_file(receipt, output_format=output_format, file=sys.stdout)
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        _write_receipt_to_file(receipt, output_format=output_format, file=file)


def _write_receipt_to_file(receipt: Receipt, *, output_format: str, file: TextIO) -> None:
    if output_format == "csv":
        write_receipt_items_csv(receipt, file)
        return

    if output_format == "json":
        write_receipt_json(receipt, file)
        return

    raise ValueError(f"unsupported output format: {output_format}")


def write_receipt_items_csv(receipt: Receipt, file: TextIO) -> None:
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(_receipt_item_rows(receipt))


def write_receipt_json(receipt: Receipt, file: TextIO) -> None:
    file.write(receipt.model_dump_json(by_alias=True, indent=2, exclude_none=True))
    file.write("\n")


def _receipt_item_rows(receipt: Receipt) -> list[dict[str, object | None]]:
    extraction = receipt.extraction
    return [
        {
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
            "item_brave_search_result": item.brave_search_result,
            "item_quantity": item.quantity,
            "item_unit_price": item.unit_price,
            "item_amount": item.amount,
            "item_line_type": item.line_type.value if item.line_type is not None else None,
            "item_category_id": item.category_id,
            "item_confidence": item.confidence,
        }
        for index, item in enumerate(receipt.items, start=1)
    ]


if __name__ == "__main__":
    main()
