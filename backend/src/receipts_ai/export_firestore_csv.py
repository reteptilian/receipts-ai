from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

from receipts_ai.ingest_receipts import (
    DEFAULT_FIRESTORE_COLLECTION,
    create_firestore_client,
    write_transactions_receipt_items_csv,
)
from receipts_ai.models.transaction import Transaction

LOGGER = logging.getLogger(__name__)


class FirestoreDocumentSnapshot(Protocol):
    id: str

    def to_dict(self) -> dict[str, Any] | None: ...


class FirestoreCollectionStream(Protocol):
    def stream(self) -> Iterable[FirestoreDocumentSnapshot]: ...


class FirestoreExportClient(Protocol):
    def collection(self, collection_path: str) -> FirestoreCollectionStream: ...


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
    client: FirestoreExportClient | None = None,
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
    client: FirestoreExportClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> None:
    transactions = list(stream_transactions_from_firestore(client=client, collection=collection))
    write_transactions_receipt_items_csv(transactions, file)
    LOGGER.info(
        "Exported %d Firestore transaction(s) from collection %s",
        len(transactions),
        collection,
    )


def stream_transactions_from_firestore(
    *,
    client: FirestoreExportClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> Iterable[Transaction]:
    if not collection:
        raise ValueError("collection must not be empty")

    firestore_client = (
        client
        if client is not None
        else cast(FirestoreExportClient, cast(object, create_firestore_client()))
    )
    LOGGER.info("Streaming transactions from Firestore collection %s", collection)
    for snapshot in firestore_client.collection(collection).stream():
        document = snapshot.to_dict()
        if document is None:
            LOGGER.warning("Skipping Firestore document %s because it has no data", snapshot.id)
            continue
        yield Transaction.model_validate(document)


if __name__ == "__main__":
    main()
