from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Protocol, cast

from receipts_ai.firestore_client import DEFAULT_FIRESTORE_COLLECTION, create_firestore_client
from receipts_ai.models.transaction import Transaction

__all__ = (
    "FirestoreTransactionClient",
    "stream_transactions_from_firestore",
    "transactions_from_firestore",
)

LOGGER = logging.getLogger(__name__)


class FirestoreDocumentSnapshot(Protocol):
    id: str

    def to_dict(self) -> dict[str, Any] | None: ...


class FirestoreCollectionStream(Protocol):
    def stream(self) -> Iterable[FirestoreDocumentSnapshot]: ...


class FirestoreTransactionClient(Protocol):
    def collection(self, collection_path: str) -> FirestoreCollectionStream: ...


def transactions_from_firestore(
    *,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> list[Transaction]:
    """Download all transaction records from Firestore."""
    transactions = list(stream_transactions_from_firestore(client=client, collection=collection))
    LOGGER.info(
        "Downloaded %d Firestore transaction(s) from collection %s",
        len(transactions),
        collection,
    )
    return transactions


def stream_transactions_from_firestore(
    *,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> Iterable[Transaction]:
    if not collection:
        raise ValueError("collection must not be empty")

    firestore_client = (
        client
        if client is not None
        else cast(FirestoreTransactionClient, cast(object, create_firestore_client()))
    )
    LOGGER.info("Streaming transactions from Firestore collection %s", collection)
    for snapshot in firestore_client.collection(collection).stream():
        document = snapshot.to_dict()
        if document is None:
            LOGGER.warning("Skipping Firestore document %s because it has no data", snapshot.id)
            continue
        yield Transaction.model_validate(document)
