from __future__ import annotations

import csv
import sys
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from typing import Any, TypedDict, Unpack

import pytest

import receipts_ai
from receipts_ai import export_firestore_csv
from receipts_ai.export_firestore_csv import export_firestore_receipt_items_csv
from receipts_ai.firestore_transactions import (
    set_receipt_item_user_overrides,
    set_transaction_user_overrides,
    stream_transactions_from_firestore,
    transactions_from_firestore,
)
from receipts_ai.ingest_receipts import transaction_firestore_document
from receipts_ai.models.transaction import (
    CategoryAllocation,
    IngestionType,
    LineType,
    Receipt,
    ReceiptItemUserOverrides,
    Source,
    Transaction,
    TransactionUserOverrides,
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
    taxonomy1: str | None
    taxonomy2: str | None
    taxonomy3: str | None
    taxonomy4: str | None
    taxonomy5: str | None
    taxonomy6: str | None
    taxonomy7: str | None
    taxonomy8: str | None
    taxonomy9: str | None
    confidence: float | None


def ReceiptItem(**kwargs: Unpack[ReceiptItemKwargs]) -> GeneratedReceiptItem:  # noqa: N802
    if "amount" in kwargs and "net_amount" not in kwargs:
        kwargs["net_amount"] = kwargs["amount"]
    return GeneratedReceiptItem(**kwargs)


class FakeDocumentSnapshot:
    def __init__(self, document_id: str, document: dict[str, Any] | None) -> None:
        self.id = document_id
        self._document = document

    def to_dict(self) -> dict[str, Any] | None:
        return self._document


class FakeDocumentReference:
    def __init__(self, collection: FakeCollection, document_id: str) -> None:
        self.collection = collection
        self.document_id = document_id

    def get(self) -> FakeDocumentSnapshot:
        return FakeDocumentSnapshot(self.document_id, self.collection.documents[self.document_id])

    def set(self, document_data: dict[str, Any], *, merge: bool = False) -> None:
        self.collection.set_calls.append((self.document_id, document_data, merge))


class FakeCollection:
    def __init__(self, snapshots: list[FakeDocumentSnapshot]) -> None:
        self.snapshots = snapshots
        self.documents = {snapshot.id: snapshot.to_dict() for snapshot in snapshots}
        self.set_calls: list[tuple[str, dict[str, Any], bool]] = []

    def document(self, document_id: str) -> FakeDocumentReference:
        return FakeDocumentReference(self, document_id)

    def stream(self) -> list[FakeDocumentSnapshot]:
        return self.snapshots


class FakeFirestoreClient:
    def __init__(self, snapshots: list[FakeDocumentSnapshot]) -> None:
        self.snapshots = snapshots
        self.collections: list[str] = []
        self.collection_references: dict[str, FakeCollection] = {}

    def collection(self, collection_path: str) -> FakeCollection:
        self.collections.append(collection_path)
        if collection_path not in self.collection_references:
            self.collection_references[collection_path] = FakeCollection(self.snapshots)
        return self.collection_references[collection_path]


def test_streams_transactions_from_firestore_documents():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        ingestion_datetime=datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC),
        ingestion_filename="receipt.pdf",
        ingestion_file_url="file:///tmp/receipt.pdf",
        ingestion_file_sha256_hex="0" * 64,
        ingestion_type=IngestionType.receipt_img,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    client = FakeFirestoreClient(
        [FakeDocumentSnapshot("receipt_1", transaction_firestore_document(transaction))]
    )

    transactions = list(
        stream_transactions_from_firestore(client=client, collection="test-transactions")
    )

    assert client.collections == ["test-transactions"]
    assert transactions == [transaction]


def test_downloads_all_transactions_from_firestore_documents():
    first = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
    )
    second = Transaction(
        id="statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 28),
        description="ACH CREDIT PAYROLL",
        amount="100.00",
        currency="USD",
    )
    client = FakeFirestoreClient(
        [
            FakeDocumentSnapshot("receipt_1", transaction_firestore_document(first)),
            FakeDocumentSnapshot("missing", None),
            FakeDocumentSnapshot("statement_1", transaction_firestore_document(second)),
        ]
    )

    transactions = transactions_from_firestore(client=client, collection="test-transactions")

    assert client.collections == ["test-transactions"]
    assert transactions == [first, second]
    assert receipts_ai.transactions_from_firestore is transactions_from_firestore


def test_sets_transaction_user_overrides_in_firestore():
    updated_at = datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC)
    client = FakeFirestoreClient([])

    set_transaction_user_overrides(
        "statement_1",
        TransactionUserOverrides(
            description="Costco",
            category_allocations=[],
        ),
        client=client,
        collection="test-transactions",
        updated_at=updated_at,
    )

    collection = client.collection_references["test-transactions"]
    assert collection.set_calls == [
        (
            "statement_1",
            {
                "userOverrides": {
                    "description": "Costco",
                    "categoryAllocations": [],
                },
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        )
    ]


def test_sets_receipt_item_user_overrides_in_firestore_by_index():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-10.00",
        currency="USD",
        receipt=Receipt(
            total="10.00",
            items=[
                ReceiptItem(id="coffee", description="Coffee", amount="7.00"),
                ReceiptItem(id="bagel", description="Bagel", amount="3.00"),
            ],
        ),
    )
    client = FakeFirestoreClient(
        [FakeDocumentSnapshot("receipt_1", transaction_firestore_document(transaction))]
    )

    set_receipt_item_user_overrides(
        "receipt_1",
        ReceiptItemUserOverrides(
            amount="4.00",
            net_amount="4.00",
            category_id="Food & Dining > Bakeries",
        ),
        item_index=1,
        client=client,
        collection="test-transactions",
        updated_at=datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC),
    )

    collection = client.collection_references["test-transactions"]
    assert collection.set_calls == [
        (
            "receipt_1",
            {
                "receipt": {
                    "items": [
                        {
                            "id": "coffee",
                            "description": "Coffee",
                            "amount": "7.00",
                            "netAmount": "7.00",
                            "lineType": "item",
                        },
                        {
                            "id": "bagel",
                            "description": "Bagel",
                            "amount": "3.00",
                            "netAmount": "3.00",
                            "lineType": "item",
                            "userOverrides": {
                                "amount": "4.00",
                                "netAmount": "4.00",
                                "categoryId": "Food & Dining > Bakeries",
                            },
                        },
                    ]
                },
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        )
    ]


def test_sets_receipt_item_user_overrides_in_firestore_by_item_id():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(id="coffee", description="Coffee", amount="7.00")]),
    )
    client = FakeFirestoreClient(
        [FakeDocumentSnapshot("receipt_1", transaction_firestore_document(transaction))]
    )

    set_receipt_item_user_overrides(
        "receipt_1",
        {"categoryId": "Food & Dining > Coffee Shops"},
        receipt_item_id="coffee",
        client=client,
        collection="test-transactions",
        updated_at=datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC),
    )

    collection = client.collection_references["test-transactions"]
    assert collection.set_calls[0][1]["receipt"]["items"][0]["userOverrides"] == {
        "categoryId": "Food & Dining > Coffee Shops"
    }


def test_export_firestore_receipt_items_csv_writes_all_transaction_rows(tmp_path: Path):
    first = Transaction(
        id="receipt_1",
        source=Source.receipt,
        ingestion_datetime=datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC),
        ingestion_filename="receipt.pdf",
        ingestion_file_url="file:///tmp/receipt.pdf",
        ingestion_file_sha256_hex="0" * 64,
        ingestion_type=IngestionType.receipt_img,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-10.00",
        currency="USD",
        receipt=Receipt(
            total="10.00",
            items=[
                ReceiptItem(
                    description="Coffee",
                    amount="7.00",
                    net_amount="6.50",
                    category_id="Food & Dining > Coffee Shops",
                ),
                ReceiptItem(
                    description="Bagel",
                    amount="3.00",
                    category_id="Food & Dining > Bakeries",
                ),
            ],
        ),
    )
    second = Transaction(
        id="receipt_2",
        source=Source.receipt,
        transaction_date=date(2026, 4, 28),
        payee="Grocery",
        amount="-4.49",
        currency="USD",
        receipt=Receipt(
            items=[
                ReceiptItem(
                    description="Saltines",
                    amount="4.49",
                    category_id="Groceries",
                )
            ]
        ),
    )
    client = FakeFirestoreClient(
        [
            FakeDocumentSnapshot("receipt_1", transaction_firestore_document(first)),
            FakeDocumentSnapshot("receipt_2", transaction_firestore_document(second)),
        ]
    )
    output_path = tmp_path / "firestore-export.csv"

    export_firestore_receipt_items_csv(
        output_path=output_path,
        client=client,
        collection="test-transactions",
    )

    rows = list(csv.DictReader(StringIO(output_path.read_text(encoding="utf-8"))))
    assert "item_brave_search_result" not in rows[0]
    assert "item_category_id" not in rows[0]
    assert "transaction_description" in rows[0]
    assert "combined_description" in rows[0]
    assert "category_allocation.category_id" in rows[0]
    assert "category_allocation.amount" in rows[0]
    assert [row["transaction_id"] for row in rows] == ["receipt_1", "receipt_1", "receipt_2"]
    assert [row["ingestion_datetime"] for row in rows] == [
        "2026-05-06T07:08:09+00:00",
        "2026-05-06T07:08:09+00:00",
        "",
    ]
    assert [row["ingestion_filename"] for row in rows] == ["receipt.pdf", "receipt.pdf", ""]
    assert [row["ingestion_file_url"] for row in rows] == [
        "file:///tmp/receipt.pdf",
        "file:///tmp/receipt.pdf",
        "",
    ]
    assert [row["ingestion_file_sha256_hex"] for row in rows] == [
        "0" * 64,
        "0" * 64,
        "",
    ]
    assert [row["ingestion_type"] for row in rows] == ["receipt_img", "receipt_img", ""]
    assert [row["item_index"] for row in rows] == ["1", "2", "1"]
    assert [row["item_description"] for row in rows] == ["Coffee", "Bagel", "Saltines"]
    assert [row["transaction_description"] for row in rows] == ["", "", ""]
    assert [row["combined_description"] for row in rows] == ["Coffee", "Bagel", "Saltines"]
    assert [row["transaction_amount"] for row in rows] == ["-10.00", "-10.00", "-4.49"]
    assert [row["category_allocation.category_id"] for row in rows] == [
        "Food & Dining > Coffee Shops",
        "Food & Dining > Bakeries",
        "Groceries",
    ]
    assert [row["category_allocation.amount"] for row in rows] == ["-6.50", "-3.00", "-4.49"]


def test_export_firestore_csv_unpivots_transactions_without_receipt_items(
    tmp_path: Path,
):
    transaction = Transaction(
        id="statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="Grocery",
        description="POS PURCHASE GROCERY #123",
        amount="-12.00",
        currency="USD",
        category_allocations=[
            CategoryAllocation(category_id="Groceries", amount="-7.00"),
            CategoryAllocation(category_id="Household", amount="-5.00"),
        ],
    )
    client = FakeFirestoreClient(
        [FakeDocumentSnapshot("statement_1", transaction_firestore_document(transaction))]
    )
    output_path = tmp_path / "firestore-export.csv"

    export_firestore_receipt_items_csv(
        output_path=output_path,
        client=client,
        collection="test-transactions",
    )

    rows = list(csv.DictReader(StringIO(output_path.read_text(encoding="utf-8"))))
    assert len(rows) == 2
    assert [row["transaction_id"] for row in rows] == ["statement_1", "statement_1"]
    assert [row["transaction_description"] for row in rows] == [
        "POS PURCHASE GROCERY #123",
        "POS PURCHASE GROCERY #123",
    ]
    assert [row["combined_description"] for row in rows] == [
        "POS PURCHASE GROCERY #123",
        "POS PURCHASE GROCERY #123",
    ]
    assert [row["category_allocation.category_id"] for row in rows] == [
        "Groceries",
        "Household",
    ]
    assert [row["category_allocation.amount"] for row in rows] == ["-7.00", "-5.00"]
    assert [row["item_description"] for row in rows] == ["", ""]
    assert [row["item_taxonomy_1"] for row in rows] == ["", ""]


def test_export_firestore_csv_uses_payee_for_credit_card_combined_description(
    tmp_path: Path,
):
    transaction = Transaction(
        id="statement_1",
        source=Source.bank_statement,
        account_id="5555444433331111",
        transaction_date=date(2026, 4, 27),
        payee="COSTCO WHSE #123",
        description="POS PURCHASE COSTCO WHSE #123",
        amount="-42.19",
        currency="USD",
    )
    client = FakeFirestoreClient(
        [FakeDocumentSnapshot("statement_1", transaction_firestore_document(transaction))]
    )
    output_path = tmp_path / "firestore-export.csv"

    export_firestore_receipt_items_csv(
        output_path=output_path,
        client=client,
        collection="test-transactions",
    )

    rows = list(csv.DictReader(StringIO(output_path.read_text(encoding="utf-8"))))
    assert len(rows) == 1
    assert rows[0]["transaction_description"] == "POS PURCHASE COSTCO WHSE #123"
    assert rows[0]["combined_description"] == "COSTCO WHSE #123"


def test_export_firestore_csv_writes_transaction_row_without_category_allocations(
    tmp_path: Path,
):
    transaction = Transaction(
        id="statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        category_allocations=[],
    )
    client = FakeFirestoreClient(
        [FakeDocumentSnapshot("statement_1", transaction_firestore_document(transaction))]
    )
    output_path = tmp_path / "firestore-export.csv"

    export_firestore_receipt_items_csv(
        output_path=output_path,
        client=client,
        collection="test-transactions",
    )

    rows = list(csv.DictReader(StringIO(output_path.read_text(encoding="utf-8"))))
    assert len(rows) == 1
    assert rows[0]["transaction_id"] == "statement_1"
    assert rows[0]["transaction_amount"] == "-7.00"
    assert rows[0]["combined_description"] == ""
    assert rows[0]["category_allocation.category_id"] == ""
    assert rows[0]["category_allocation.amount"] == ""
    assert rows[0]["item_description"] == ""


def test_export_skips_missing_firestore_documents(tmp_path: Path):
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    client = FakeFirestoreClient(
        [
            FakeDocumentSnapshot("missing", None),
            FakeDocumentSnapshot("receipt_1", transaction_firestore_document(transaction)),
        ]
    )

    output_path = tmp_path / "firestore-export.csv"
    export_firestore_receipt_items_csv(
        output_path=output_path,
        client=client,
        collection="test-transactions",
    )

    rows = list(csv.DictReader(StringIO(output_path.read_text(encoding="utf-8"))))
    assert len(rows) == 1
    assert rows[0]["transaction_id"] == "receipt_1"


def test_main_exports_firestore_csv_to_output_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    output_path = tmp_path / "firestore-export.csv"
    calls: list[tuple[Path | None, str]] = []

    def fake_export_firestore_receipt_items_csv(
        *,
        output_path: Path | None = None,
        collection: str,
    ) -> None:
        calls.append((output_path, collection))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-export-firestore-csv",
            "--firestore-collection",
            "processed-transactions",
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        export_firestore_csv,
        "export_firestore_receipt_items_csv",
        fake_export_firestore_receipt_items_csv,
    )

    export_firestore_csv.main()

    assert calls == [(output_path, "processed-transactions")]
