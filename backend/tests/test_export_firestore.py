from __future__ import annotations

import csv
import sys
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from typing import Any, TypedDict, Unpack

import pytest

import receipts_ai
from receipts_ai import export_firestore
from receipts_ai.export_firestore import export_firestore_receipt_items_csv
from receipts_ai.firestore_transactions import (
    link_bank_statement_transaction_to_receipt,
    save_transaction_review_edits,
    set_receipt_item_user_overrides,
    set_transaction_user_overrides,
    stream_transactions_from_firestore,
    transactions_from_firestore,
    unlink_bank_statement_transaction_from_receipt,
)
from receipts_ai.ingest_receipts import transaction_firestore_document
from receipts_ai.models.transaction import (
    CategoryAllocation,
    IngestionType,
    LineType,
    Receipt,
    ReceiptItemUserOverrides,
    RecordType,
    Source,
    Transaction,
    TransactionUserOverrides,
    UserCategoryAllocation,
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


class FakeWriteBatch:
    def __init__(self) -> None:
        self.set_calls: list[tuple[str, dict[str, Any], bool]] = []
        self.commit_calls = 0

    def set(
        self,
        reference: Any,
        document_data: dict[str, Any],
        *,
        merge: bool = False,
    ) -> None:
        self.set_calls.append((reference.document_id, document_data, merge))

    def commit(self) -> None:
        self.commit_calls += 1


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
        self.write_batch = FakeWriteBatch()

    def collection(self, collection_path: str) -> FakeCollection:
        self.collections.append(collection_path)
        if collection_path not in self.collection_references:
            self.collection_references[collection_path] = FakeCollection(self.snapshots)
        return self.collection_references[collection_path]

    def batch(self) -> FakeWriteBatch:
        return self.write_batch


class FakeWorksheet:
    _next_id = 1

    def __init__(self, title: str, rows: int, cols: int) -> None:
        self.title = title
        self.rows = rows
        self.cols = cols
        self.id = FakeWorksheet._next_id
        FakeWorksheet._next_id += 1
        self.values: list[list[object]] = []
        self.clear_calls = 0
        self.freeze_calls: list[dict[str, int | None]] = []

    def clear(self) -> None:
        self.clear_calls += 1
        self.values = []

    def resize(self, *, rows: int | None = None, cols: int | None = None) -> None:
        if rows is not None:
            self.rows = rows
        if cols is not None:
            self.cols = cols

    def update(
        self,
        values: list[list[object]],
        *,
        range_name: str,
        value_input_option: str,
    ) -> None:
        assert range_name == "A1"
        assert value_input_option == "USER_ENTERED"
        self.values = values

    def freeze(self, *, rows: int | None = None, cols: int | None = None) -> None:
        self.freeze_calls.append({"rows": rows, "cols": cols})


class FakeSpreadsheet:
    def __init__(self) -> None:
        self.worksheets_by_title: dict[str, FakeWorksheet] = {
            "Sheet1": FakeWorksheet(title="Sheet1", rows=1000, cols=26)
        }
        self.deleted_worksheet_titles: list[str] = []
        self.batch_update_calls: list[dict[str, Any]] = []

    def worksheet(self, title: str) -> FakeWorksheet:
        import gspread

        if title not in self.worksheets_by_title:
            raise gspread.WorksheetNotFound(title)
        return self.worksheets_by_title[title]

    def add_worksheet(self, *, title: str, rows: int, cols: int) -> FakeWorksheet:
        worksheet = FakeWorksheet(title=title, rows=rows, cols=cols)
        self.worksheets_by_title[title] = worksheet
        return worksheet

    def del_worksheet(self, worksheet: FakeWorksheet) -> None:
        self.deleted_worksheet_titles.append(worksheet.title)
        del self.worksheets_by_title[worksheet.title]

    def batch_update(self, body: dict[str, Any]) -> None:
        self.batch_update_calls.append(body)


class FakeGspreadClient:
    def __init__(self) -> None:
        self.created_titles: list[str] = []
        self.opened_titles: list[str] = []
        self.opened_keys: list[str] = []
        self.opened_urls: list[str] = []
        self.spreadsheet = FakeSpreadsheet()

    def open(self, title: str) -> FakeSpreadsheet:
        import gspread

        self.opened_titles.append(title)
        raise gspread.SpreadsheetNotFound(title)

    def create(self, title: str) -> FakeSpreadsheet:
        self.created_titles.append(title)
        return self.spreadsheet

    def open_by_key(self, key: str) -> FakeSpreadsheet:
        self.opened_keys.append(key)
        return self.spreadsheet

    def open_by_url(self, url: str) -> FakeSpreadsheet:
        self.opened_urls.append(url)
        return self.spreadsheet


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
        {
            "description": "Costco",
            "transactionDate": "2026-04-28",
            "categoryAllocations": [],
        },
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
                    "transactionDate": "2026-04-28",
                    "categoryAllocations": [],
                },
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        )
    ]


def test_saves_transaction_review_edits_in_firestore_batch():
    updated_at = datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC)
    client = FakeFirestoreClient([])
    item = ReceiptItem(
        id="item_1",
        description="Latte",
        amount="7.00",
    )
    item.user_overrides = ReceiptItemUserOverrides(amount="7.50")

    save_transaction_review_edits(
        "statement_1",
        TransactionUserOverrides(
            payee="Coffee Shop",
            category_allocations=[
                UserCategoryAllocation(category_id="Food & Dining > Coffee", amount="-7.50")
            ],
        ),
        receipt_transaction_id="receipt_1",
        receipt_items=[item],
        client=client,
        collection="test-transactions",
        updated_at=updated_at,
    )

    assert client.collections == ["test-transactions"]
    assert client.write_batch.commit_calls == 1
    assert client.write_batch.set_calls == [
        (
            "statement_1",
            {
                "userOverrides": {
                    "payee": "Coffee Shop",
                    "categoryAllocations": [
                        {
                            "categoryId": "Food & Dining > Coffee",
                            "amount": "-7.50",
                            "source": "user",
                        }
                    ],
                },
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        ),
        (
            "receipt_1",
            {
                "receipt": {
                    "items": [
                        {
                            "id": "item_1",
                            "description": "Latte",
                            "amount": "7.00",
                            "netAmount": "7.00",
                            "lineType": "item",
                            "userOverrides": {"amount": "7.50"},
                        }
                    ]
                },
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        ),
    ]


def test_links_bank_statement_transaction_to_receipt_in_firestore():
    updated_at = datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC)
    bst = Transaction(
        id="statement_1",
        source=Source.bank_statement,
        record_type=RecordType.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
    )
    rbt = Transaction(
        id="receipt_1",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    client = FakeFirestoreClient(
        [
            FakeDocumentSnapshot("statement_1", transaction_firestore_document(bst)),
            FakeDocumentSnapshot("receipt_1", transaction_firestore_document(rbt)),
        ]
    )

    link_bank_statement_transaction_to_receipt(
        "statement_1",
        "receipt_1",
        client=client,
        collection="test-transactions",
        updated_at=updated_at,
    )

    collection = client.collection_references["test-transactions"]
    assert collection.set_calls == [
        (
            "statement_1",
            {
                "recordType": "bank_statement",
                "linkedReceiptBasedTransactionId": "receipt_1",
                "linkedTransactionIds": ["receipt_1"],
                "matchStatus": "confirmed",
                "matchSource": "user",
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        ),
        (
            "receipt_1",
            {
                "recordType": "receipt_based",
                "linkedTransactionIds": ["statement_1"],
                "matchStatus": "confirmed",
                "matchSource": "user",
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        ),
    ]


def test_unlinks_bank_statement_transaction_from_receipt_in_firestore():
    updated_at = datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC)
    bst = Transaction(
        id="statement_1",
        source=Source.bank_statement,
        record_type=RecordType.bank_statement,
        linked_receipt_based_transaction_id="receipt_1",
        linked_transaction_ids=["receipt_1"],
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
    )
    rbt = Transaction(
        id="receipt_1",
        source=Source.receipt,
        record_type=RecordType.receipt_based,
        linked_transaction_ids=["statement_1"],
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    client = FakeFirestoreClient(
        [
            FakeDocumentSnapshot("statement_1", transaction_firestore_document(bst)),
            FakeDocumentSnapshot("receipt_1", transaction_firestore_document(rbt)),
        ]
    )

    unlink_bank_statement_transaction_from_receipt(
        "statement_1",
        client=client,
        collection="test-transactions",
        updated_at=updated_at,
    )

    collection = client.collection_references["test-transactions"]
    assert collection.set_calls == [
        (
            "statement_1",
            {
                "linkedReceiptBasedTransactionId": None,
                "linkedTransactionIds": [],
                "matchStatus": "unmatched",
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        ),
        (
            "receipt_1",
            {
                "linkedTransactionIds": [],
                "matchStatus": "unmatched",
                "updatedAt": "2026-05-06T07:08:09Z",
            },
            True,
        ),
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
                    taxonomy1="Food & Dining",
                    taxonomy2="Coffee Shops",
                    taxonomy3="Prepared Drinks",
                    taxonomy4="Coffee",
                    taxonomy5="Hot Drinks",
                    taxonomy6="Latte",
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


def test_export_firestore_receipt_items_csv_unpivots_transactions_without_receipt_items(
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


def test_export_firestore_receipt_items_csv_uses_payee_for_credit_card_combined_description(
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


def test_export_firestore_receipt_items_csv_writes_transaction_row_without_category_allocations(
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
    assert rows[0]["combined_description"] == "Coffee Shop"
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


def test_export_firestore_receipt_items_google_sheet_writes_records_and_budget_pivot():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-10.00",
        currency="USD",
        receipt=Receipt(
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
            ]
        ),
    )
    firestore_client = FakeFirestoreClient(
        [FakeDocumentSnapshot("receipt_1", transaction_firestore_document(transaction))]
    )
    gspread_client = FakeGspreadClient()

    export_firestore.export_firestore_receipt_items_google_sheet(
        spreadsheet_title="Receipt Budget",
        gspread_client=gspread_client,
        client=firestore_client,
        collection="test-transactions",
    )

    assert gspread_client.opened_titles == ["Receipt Budget"]
    assert gspread_client.created_titles == ["Receipt Budget"]
    spreadsheet = gspread_client.spreadsheet
    records = spreadsheet.worksheet("records")
    budget = spreadsheet.worksheet("budget")
    stuff = spreadsheet.worksheet("stuff")
    assert spreadsheet.deleted_worksheet_titles == ["Sheet1"]
    assert "Sheet1" not in spreadsheet.worksheets_by_title
    header = records.values[0]
    category_column = header.index("category_allocation.category_id")
    description_column = header.index("combined_description")
    amount_column = header.index("category_allocation.amount")
    item_net_amount_column = header.index("item_net_amount")
    stuff_row_columns = [
        header.index(fieldname)
        for fieldname in (
            "item_taxonomy_1",
            "item_taxonomy_2",
            "item_taxonomy_3",
            "item_taxonomy_4",
            "item_taxonomy_5",
            "item_taxonomy_6",
            "combined_description",
        )
    ]
    assert records.values[1][description_column] == "Coffee"
    assert records.values[1][category_column] == "Food & Dining > Coffee Shops"
    assert records.values[1][amount_column] == "-6.50"
    assert records.values[1][item_net_amount_column] == "6.50"
    assert records.values[2][description_column] == "Bagel"
    assert records.freeze_calls == [{"rows": 1, "cols": None}]

    budget_pivot = spreadsheet.batch_update_calls[0]["requests"][0]["updateCells"]["rows"][0][
        "values"
    ][0]["pivotTable"]
    assert budget_pivot["source"]["sheetId"] == records.id
    assert budget_pivot["source"]["endRowIndex"] == 3
    assert [row["sourceColumnOffset"] for row in budget_pivot["rows"]] == [
        category_column,
        description_column,
    ]
    assert budget_pivot["values"] == [
        {
            "sourceColumnOffset": amount_column,
            "summarizeFunction": "SUM",
            "name": "Sum of category_allocation.amount",
        }
    ]
    assert spreadsheet.batch_update_calls[0]["requests"][0]["updateCells"]["start"] == {
        "sheetId": budget.id,
        "rowIndex": 0,
        "columnIndex": 0,
    }

    stuff_pivot = spreadsheet.batch_update_calls[1]["requests"][0]["updateCells"]["rows"][0][
        "values"
    ][0]["pivotTable"]
    assert stuff_pivot["source"]["sheetId"] == records.id
    assert stuff_pivot["source"]["endRowIndex"] == 3
    assert [row["sourceColumnOffset"] for row in stuff_pivot["rows"]] == stuff_row_columns
    assert stuff_pivot["values"] == [
        {
            "sourceColumnOffset": item_net_amount_column,
            "summarizeFunction": "SUM",
            "name": "Sum of item_net_amount",
        }
    ]
    assert spreadsheet.batch_update_calls[1]["requests"][0]["updateCells"]["start"] == {
        "sheetId": stuff.id,
        "rowIndex": 0,
        "columnIndex": 0,
    }


def test_export_firestore_receipt_items_google_sheet_opens_existing_sheet_by_id():
    client = FakeFirestoreClient([])
    gspread_client = FakeGspreadClient()

    export_firestore.export_firestore_receipt_items_google_sheet(
        spreadsheet_id="spreadsheet-key",
        gspread_client=gspread_client,
        client=client,
        collection="test-transactions",
    )

    assert gspread_client.opened_keys == ["spreadsheet-key"]
    assert gspread_client.created_titles == []
    assert gspread_client.spreadsheet.worksheet("records").values[0][0] == "transaction_id"


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
            "receipts-ai-export-firestore",
            "--firestore-collection",
            "processed-transactions",
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(
        export_firestore,
        "export_firestore_receipt_items_csv",
        fake_export_firestore_receipt_items_csv,
    )

    export_firestore.main()

    assert calls == [(output_path, "processed-transactions")]


def test_main_exports_firestore_google_sheet(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []

    def fake_export_firestore_receipt_items_google_sheet(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-export-firestore",
            "--firestore-collection",
            "processed-transactions",
            "--google-sheet-title",
            "Receipt Budget",
            "--google-oauth-credentials",
            "/tmp/credentials.json",
            "--google-oauth-authorized-user",
            "/tmp/authorized_user.json",
        ],
    )
    monkeypatch.setattr(
        export_firestore,
        "export_firestore_receipt_items_google_sheet",
        fake_export_firestore_receipt_items_google_sheet,
    )

    export_firestore.main()

    assert calls == [
        {
            "spreadsheet_title": "Receipt Budget",
            "spreadsheet_id": None,
            "spreadsheet_url": None,
            "oauth_credentials_path": Path("/tmp/credentials.json"),
            "oauth_authorized_user_path": Path("/tmp/authorized_user.json"),
            "collection": "processed-transactions",
        }
    ]
