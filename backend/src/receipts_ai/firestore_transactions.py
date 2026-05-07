from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from receipts_ai.firestore_client import DEFAULT_FIRESTORE_COLLECTION, create_firestore_client
from receipts_ai.models.transaction import (
    MatchSource,
    MatchStatus,
    ReceiptItem,
    ReceiptItemUserOverrides,
    RecordType,
    Source,
    Transaction,
    TransactionUserOverrides,
)

__all__ = (
    "FirestoreTransactionClient",
    "link_bank_statement_transaction_to_receipt",
    "save_transaction_review_edits",
    "set_receipt_item_user_overrides",
    "set_transaction_user_overrides",
    "stream_transactions_from_firestore",
    "transactions_from_firestore",
    "unlink_bank_statement_transaction_from_receipt",
)

LOGGER = logging.getLogger(__name__)


class FirestoreDocumentSnapshot(Protocol):
    id: str

    def to_dict(self) -> dict[str, Any] | None: ...


class FirestoreDocumentReference(Protocol):
    def get(self) -> FirestoreDocumentSnapshot: ...

    def set(self, document_data: dict[str, Any], *, merge: bool = False) -> object: ...


class FirestoreWriteBatch(Protocol):
    def set(
        self,
        reference: FirestoreDocumentReference,
        document_data: dict[str, Any],
        *,
        merge: bool = False,
    ) -> object: ...

    def commit(self) -> object: ...


class FirestoreCollectionReference(Protocol):
    def document(self, document_id: str) -> FirestoreDocumentReference: ...

    def stream(self) -> Iterable[FirestoreDocumentSnapshot]: ...


class FirestoreTransactionClient(Protocol):
    def collection(self, collection_path: str) -> FirestoreCollectionReference: ...

    def batch(self) -> FirestoreWriteBatch: ...


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


def save_transaction_review_edits(
    transaction_id: str,
    transaction_user_overrides: TransactionUserOverrides | Mapping[str, Any],
    *,
    receipt_transaction_id: str | None = None,
    receipt_items: Sequence[ReceiptItem] | None = None,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
    updated_at: datetime | None = None,
) -> None:
    """Persist a complete transaction review draft.

    The transaction-level edits are written to the selected/display transaction.
    Receipt item edits may belong to a linked receipt transaction, so they are
    addressed separately and committed in the same Firestore batch.
    """
    if not transaction_id:
        raise ValueError("transaction_id must not be empty")
    if not collection:
        raise ValueError("collection must not be empty")
    if receipt_transaction_id == "":
        raise ValueError("receipt_transaction_id must not be empty")
    if receipt_items is not None and receipt_transaction_id is None:
        raise ValueError("receipt_transaction_id is required when receipt_items are provided")

    firestore_client = _firestore_transaction_client(client)
    collection_reference = firestore_client.collection(collection)
    timestamp = _json_datetime(updated_at or datetime.now(UTC))
    transaction_ref = collection_reference.document(transaction_id)
    batch = firestore_client.batch()
    batch.set(
        transaction_ref,
        {
            "userOverrides": _transaction_user_overrides_document(transaction_user_overrides),
            "updatedAt": timestamp,
        },
        merge=True,
    )

    if receipt_items is not None:
        receipt_ref = collection_reference.document(cast(str, receipt_transaction_id))
        items_document = [
            item.model_dump(mode="json", by_alias=True, exclude_none=True) for item in receipt_items
        ]
        batch.set(
            receipt_ref,
            {
                "receipt": {"items": items_document},
                "updatedAt": timestamp,
            },
            merge=True,
        )

    LOGGER.info(
        "Persisting transaction review edits for %s in collection %s",
        transaction_id,
        collection,
    )
    batch.commit()


def link_bank_statement_transaction_to_receipt(
    bank_statement_transaction_id: str,
    receipt_based_transaction_id: str,
    *,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
    updated_at: datetime | None = None,
) -> None:
    """Link a bank statement transaction (BST) to a receipt-based transaction (RBT)."""
    if not collection:
        raise ValueError("collection must not be empty")
    if not bank_statement_transaction_id:
        raise ValueError("bank_statement_transaction_id must not be empty")
    if not receipt_based_transaction_id:
        raise ValueError("receipt_based_transaction_id must not be empty")
    if bank_statement_transaction_id == receipt_based_transaction_id:
        raise ValueError("BST and RBT transaction ids must be different")

    firestore_client = _firestore_transaction_client(client)
    bst_ref = firestore_client.collection(collection).document(bank_statement_transaction_id)
    rbt_ref = firestore_client.collection(collection).document(receipt_based_transaction_id)
    bst = _transaction_from_reference(bst_ref, collection=collection)
    rbt = _transaction_from_reference(rbt_ref, collection=collection)

    if not _is_bank_statement_transaction(bst):
        raise ValueError(f"transaction {bank_statement_transaction_id} is not a BST")
    if not _is_receipt_based_transaction(rbt):
        raise ValueError(f"transaction {receipt_based_transaction_id} is not an RBT")

    timestamp = _json_datetime(updated_at or datetime.now(UTC))
    bst_ref.set(
        {
            "recordType": RecordType.bank_statement.value,
            "linkedReceiptBasedTransactionId": receipt_based_transaction_id,
            "linkedTransactionIds": _linked_transaction_ids(bst, receipt_based_transaction_id),
            "matchStatus": MatchStatus.confirmed.value,
            "matchSource": MatchSource.user.value,
            "updatedAt": timestamp,
        },
        merge=True,
    )
    rbt_ref.set(
        {
            "recordType": RecordType.receipt_based.value,
            "linkedTransactionIds": _linked_transaction_ids(rbt, bank_statement_transaction_id),
            "matchStatus": MatchStatus.confirmed.value,
            "matchSource": MatchSource.user.value,
            "updatedAt": timestamp,
        },
        merge=True,
    )


def unlink_bank_statement_transaction_from_receipt(
    bank_statement_transaction_id: str,
    *,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
    updated_at: datetime | None = None,
) -> None:
    """Remove the RBT link from a bank statement transaction."""
    if not collection:
        raise ValueError("collection must not be empty")
    if not bank_statement_transaction_id:
        raise ValueError("bank_statement_transaction_id must not be empty")

    firestore_client = _firestore_transaction_client(client)
    bst_ref = firestore_client.collection(collection).document(bank_statement_transaction_id)
    bst = _transaction_from_reference(bst_ref, collection=collection)
    if not _is_bank_statement_transaction(bst):
        raise ValueError(f"transaction {bank_statement_transaction_id} is not a BST")

    receipt_based_transaction_id = bst.linked_receipt_based_transaction_id
    timestamp = _json_datetime(updated_at or datetime.now(UTC))
    bst_ref.set(
        {
            "linkedReceiptBasedTransactionId": None,
            "linkedTransactionIds": _linked_transaction_ids_without(
                bst, receipt_based_transaction_id
            ),
            "matchStatus": MatchStatus.unmatched.value,
            "updatedAt": timestamp,
        },
        merge=True,
    )
    if receipt_based_transaction_id is None:
        return

    rbt_ref = firestore_client.collection(collection).document(receipt_based_transaction_id)
    rbt_snapshot = rbt_ref.get()
    rbt_document = rbt_snapshot.to_dict()
    if rbt_document is None:
        return
    rbt = Transaction.model_validate(rbt_document)
    rbt_ref.set(
        {
            "linkedTransactionIds": _linked_transaction_ids_without(
                rbt, bank_statement_transaction_id
            ),
            "matchStatus": MatchStatus.unmatched.value,
            "updatedAt": timestamp,
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


def _transaction_from_reference(
    document_reference: FirestoreDocumentReference, *, collection: str
) -> Transaction:
    snapshot = document_reference.get()
    document = snapshot.to_dict()
    if document is None:
        raise ValueError(f"transaction {snapshot.id} does not exist in collection {collection}")
    return Transaction.model_validate(document)


def _is_bank_statement_transaction(transaction: Transaction) -> bool:
    return (
        transaction.record_type == RecordType.bank_statement
        or transaction.source == Source.bank_statement
    )


def _is_receipt_based_transaction(transaction: Transaction) -> bool:
    return (
        transaction.record_type == RecordType.receipt_based or transaction.source == Source.receipt
    )


def _linked_transaction_ids(transaction: Transaction, linked_transaction_id: str) -> list[str]:
    return sorted({*(transaction.linked_transaction_ids or []), linked_transaction_id})


def _linked_transaction_ids_without(
    transaction: Transaction, linked_transaction_id: str | None
) -> list[str]:
    if linked_transaction_id is None:
        return list(transaction.linked_transaction_ids or [])
    return [
        transaction_id
        for transaction_id in transaction.linked_transaction_ids or []
        if transaction_id != linked_transaction_id
    ]


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
