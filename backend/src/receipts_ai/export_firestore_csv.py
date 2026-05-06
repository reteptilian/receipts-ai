from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TextIO

from receipts_ai.firestore_transactions import (
    FirestoreTransactionClient,
    stream_transactions_from_firestore,
)
from receipts_ai.ingest_receipts import (
    DEFAULT_FIRESTORE_COLLECTION,
    write_transactions_receipt_items_csv,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all Firestore transaction receipt items to CSV."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write CSV output to a file instead of stdout.",
    )
    parser.add_argument(
        "--firestore-collection",
        default=DEFAULT_FIRESTORE_COLLECTION,
        help=f"Firestore collection to export from. Defaults to {DEFAULT_FIRESTORE_COLLECTION}.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        help="Show logs at this level.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(name)s:%(message)s")

    export_firestore_receipt_items_csv(
        output_path=args.output,
        collection=args.firestore_collection,
    )


def export_firestore_receipt_items_csv(
    *,
    output_path: Path | None = None,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> None:
    if output_path is None:
        _export_firestore_receipt_items_csv_to_file(
            sys.stdout,
            client=client,
            collection=collection,
        )
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        _export_firestore_receipt_items_csv_to_file(
            file,
            client=client,
            collection=collection,
        )


def _export_firestore_receipt_items_csv_to_file(
    file: TextIO,
    *,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> None:
    transactions = list(stream_transactions_from_firestore(client=client, collection=collection))
    write_transactions_receipt_items_csv(transactions, file)
    LOGGER.info(
        "Exported %d Firestore transaction(s) from collection %s",
        len(transactions),
        collection,
    )


if __name__ == "__main__":
    main()
