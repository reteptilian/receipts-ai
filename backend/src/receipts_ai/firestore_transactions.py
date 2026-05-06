from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from receipts_ai.firestore_client import DEFAULT_FIRESTORE_COLLECTION, create_firestore_client
from receipts_ai.models.transaction import (
    ReceiptItemUserOverrides,
    Transaction,
    TransactionUserOverrides,
)

__all__ = (
    "FirestoreTransactionClient",
    "set_receipt_item_user_overrides",
    "set_transaction_user_overrides",
    "stream_transactions_from_firestore",
    "transactions_from_firestore",
)

LOGGER = logging.getLogger(__name__)


class FirestoreDocumentSnapshot(Protocol):
    id: str

    def to_dict(self) -> dict[str, Any] | None: ...


class FirestoreDocumentReference(Protocol):
    def get(self) -> FirestoreDocumentSnapshot: ...

    def set(self, document_data: dict[str, Any], *, merge: bool = False) -> object: ...


class FirestoreCollectionReference(Protocol):
    def document(self, document_id: str) -> FirestoreDocumentReference: ...

    def stream(self) -> Iterable[FirestoreDocumentSnapshot]: ...


class FirestoreTransactionClient(Protocol):
    def collection(self, collection_path: str) -> FirestoreCollectionReference: ...


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


def set_transaction_user_overrides(
    transaction_id: str,
    user_overrides: TransactionUserOverrides | Mapping[str, Any],
    *,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
    updated_at: datetime | None = None,
) -> None:
    """Persist user edits for top-level editable transaction fields."""
    if not transaction_id:
        raise ValueError("transaction_id must not be empty")
    if not collection:
        raise ValueError("collection must not be empty")

    firestore_client = _firestore_transaction_client(client)
    document = {
        "userOverrides": _transaction_user_overrides_document(user_overrides),
        "updatedAt": _json_datetime(updated_at or datetime.now(UTC)),
    }
    LOGGER.info(
        "Persisting transaction user overrides for %s in collection %s",
        transaction_id,
        collection,
    )
    firestore_client.collection(collection).document(transaction_id).set(document, merge=True)


def set_receipt_item_user_overrides(
    transaction_id: str,
    user_overrides: ReceiptItemUserOverrides | Mapping[str, Any],
    *,
    item_index: int | None = None,
    receipt_item_id: str | None = None,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
    updated_at: datetime | None = None,
) -> None:
    """Persist user edits for editable fields on one receipt item.

    Firestore arrays cannot be merged by element, so this reads the transaction,
    updates the selected item, and writes the receipt items array back.
    """
    if not transaction_id:
        raise ValueError("transaction_id must not be empty")
    if not collection:
        raise ValueError("collection must not be empty")
    if (item_index is None) == (receipt_item_id is None):
        raise ValueError("provide exactly one of item_index or receipt_item_id")
    if item_index is not None and item_index < 0:
        raise ValueError("item_index must be non-negative")
    if receipt_item_id == "":
        raise ValueError("receipt_item_id must not be empty")

    firestore_client = _firestore_transaction_client(client)
    document_reference = firestore_client.collection(collection).document(transaction_id)
    snapshot = document_reference.get()
    document = snapshot.to_dict()
    if document is None:
        raise ValueError(f"transaction {transaction_id} does not exist in collection {collection}")

    transaction = Transaction.model_validate(document)
    if transaction.receipt is None:
        raise ValueError(f"transaction {transaction_id} does not contain receipt items")

    resolved_index = (
        item_index
        if item_index is not None
        else _receipt_item_index_by_id(transaction, cast(str, receipt_item_id))
    )
    if resolved_index >= len(transaction.receipt.items):
        raise ValueError(
            f"receipt item index {resolved_index} is out of range for transaction {transaction_id}"
        )

    transaction.receipt.items[resolved_index].user_overrides = _receipt_item_user_overrides(
        user_overrides
    )
    items_document = [
        item.model_dump(mode="json", by_alias=True, exclude_none=True)
        for item in transaction.receipt.items
    ]
    LOGGER.info(
        "Persisting receipt item user overrides for transaction %s item %d in collection %s",
        transaction_id,
        resolved_index,
        collection,
    )
    document_reference.set(
        {
            "receipt": {"items": items_document},
            "updatedAt": _json_datetime(updated_at or datetime.now(UTC)),
        },
        merge=True,
    )


def _firestore_transaction_client(
    client: FirestoreTransactionClient | None,
) -> FirestoreTransactionClient:
    return (
        client
        if client is not None
        else cast(FirestoreTransactionClient, cast(object, create_firestore_client()))
    )


def _transaction_user_overrides_document(
    user_overrides: TransactionUserOverrides | Mapping[str, Any],
) -> dict[str, Any]:
    return _transaction_user_overrides(user_overrides).model_dump(
        mode="json", by_alias=True, exclude_none=True
    )


def _transaction_user_overrides(
    user_overrides: TransactionUserOverrides | Mapping[str, Any],
) -> TransactionUserOverrides:
    if isinstance(user_overrides, TransactionUserOverrides):
        return user_overrides
    return TransactionUserOverrides.model_validate(user_overrides)


def _receipt_item_user_overrides(
    user_overrides: ReceiptItemUserOverrides | Mapping[str, Any],
) -> ReceiptItemUserOverrides:
    if isinstance(user_overrides, ReceiptItemUserOverrides):
        return user_overrides
    return ReceiptItemUserOverrides.model_validate(user_overrides)


def _receipt_item_index_by_id(transaction: Transaction, receipt_item_id: str) -> int:
    if transaction.receipt is None:
        raise ValueError(f"transaction {transaction.id} does not contain receipt items")

    for index, item in enumerate(transaction.receipt.items):
        if item.id == receipt_item_id:
            return index
    raise ValueError(
        f"receipt item {receipt_item_id} does not exist on transaction {transaction.id}"
    )


def _json_datetime(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
