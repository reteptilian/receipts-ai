from __future__ import annotations

import csv
import sys
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from receipts_ai import export_firestore_csv
from receipts_ai.export_firestore_csv import (
    export_firestore_receipt_items_csv,
    stream_transactions_from_firestore,
)
from receipts_ai.ingest_receipts import transaction_firestore_document
from receipts_ai.models.transaction import (
    CategoryAllocation,
    Receipt,
    ReceiptItem,
    Source,
    Transaction,
)


class FakeDocumentSnapshot:
    def __init__(self, document_id: str, document: dict[str, Any] | None) -> None:
        self.id = document_id
        self._document = document

    def to_dict(self) -> dict[str, Any] | None:
        return self._document


class FakeCollection:
    def __init__(self, snapshots: list[FakeDocumentSnapshot]) -> None:
        self.snapshots = snapshots

    def stream(self) -> list[FakeDocumentSnapshot]:
        return self.snapshots


class FakeFirestoreClient:
    def __init__(self, snapshots: list[FakeDocumentSnapshot]) -> None:
        self.snapshots = snapshots
        self.collections: list[str] = []

    def collection(self, collection_path: str) -> FakeCollection:
        self.collections.append(collection_path)
        return FakeCollection(self.snapshots)


def test_streams_transactions_from_firestore_documents():
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
        [FakeDocumentSnapshot("receipt_1", transaction_firestore_document(transaction))]
    )

    transactions = list(
        stream_transactions_from_firestore(client=client, collection="test-transactions")
    )

    assert client.collections == ["test-transactions"]
    assert transactions == [transaction]


def test_export_firestore_receipt_items_csv_writes_all_transaction_rows(tmp_path: Path):
    first = Transaction(
        id="receipt_1",
        source=Source.receipt,
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
    assert [row["category_allocation.amount"] for row in rows] == ["6.50", "3.00", "4.49"]


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
