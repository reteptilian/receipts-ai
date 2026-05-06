from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from typing import TypedDict, Unpack

import pytest

from receipts_ai import firestore_client, ingest_receipts
from receipts_ai.ingest_receipts import (
    CSV_FIELDNAMES,
    main,
    transaction_firestore_document,
    upsert_transaction_to_firestore,
    write_receipt_items_csv,
    write_receipt_json,
    write_transaction_json,
    write_transaction_receipt_items_csv,
    write_transactions_json,
)
from receipts_ai.models.transaction import (
    ExtractionMetadata,
    IngestionType,
    LineType,
    Receipt,
    Source,
    Transaction,
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


def test_writes_one_csv_row_per_receipt_item():
    receipt = Receipt(
        subtotal="10.00",
        total="11.00",
        extraction=ExtractionMetadata(model="prebuilt-receipt", confidence=0.97),
        items=[
            ReceiptItem(
                description="Coffee",
                raw_description="COF",
                brave_search_result="Coffee product search result",
                quantity=2,
                unit_price="3.50",
                amount="7.00",
                discount_amount="-1.00",
                discount_description="/1779212",
                net_amount="6.00",
                category_id="Food & Dining > Fast Food & Coffee",
                taxonomy1="Food, Beverages & Tobacco",
                taxonomy2="Beverages",
            ),
            ReceiptItem(description="Bagel", amount="3.00", confidence=0.91),
        ],
    )
    output = StringIO()

    write_receipt_items_csv(receipt, output)

    header, *_ = output.getvalue().splitlines()
    assert header == ",".join(CSV_FIELDNAMES)

    rows = list(csv.DictReader(StringIO(output.getvalue())))
    assert rows == [
        {
            "transaction_id": "",
            "transaction_date": "",
            "payee": "",
            "transaction_description": "",
            "combined_description": "Coffee",
            "transaction_amount": "",
            "transaction_currency": "",
            "ingestion_datetime": "",
            "ingestion_filename": "",
            "ingestion_file_url": "",
            "ingestion_file_sha256_hex": "",
            "ingestion_type": "",
            "receipt_id": "",
            "source_document_id": "",
            "receipt_number": "",
            "receipt_subtotal": "10.00",
            "receipt_total": "11.00",
            "extraction_model": "prebuilt-receipt",
            "extraction_confidence": "0.97",
            "item_index": "1",
            "item_id": "",
            "item_description": "Coffee",
            "item_raw_description": "COF",
            "item_brave_search_result": "Coffee product search result",
            "item_quantity": "2.0",
            "item_unit_price": "3.50",
            "item_amount": "7.00",
            "item_discount_amount": "-1.00",
            "item_discount_description": "/1779212",
            "item_net_amount": "6.00",
            "item_line_type": "item",
            "item_category_id": "Food & Dining > Fast Food & Coffee",
            "item_taxonomy_1": "Food, Beverages & Tobacco",
            "item_taxonomy_2": "Beverages",
            "item_taxonomy_3": "",
            "item_taxonomy_4": "",
            "item_taxonomy_5": "",
            "item_taxonomy_6": "",
            "item_taxonomy_7": "",
            "item_taxonomy_8": "",
            "item_taxonomy_9": "",
            "item_confidence": "",
        },
        {
            "transaction_id": "",
            "transaction_date": "",
            "payee": "",
            "transaction_description": "",
            "combined_description": "Bagel",
            "transaction_amount": "",
            "transaction_currency": "",
            "ingestion_datetime": "",
            "ingestion_filename": "",
            "ingestion_file_url": "",
            "ingestion_file_sha256_hex": "",
            "ingestion_type": "",
            "receipt_id": "",
            "source_document_id": "",
            "receipt_number": "",
            "receipt_subtotal": "10.00",
            "receipt_total": "11.00",
            "extraction_model": "prebuilt-receipt",
            "extraction_confidence": "0.97",
            "item_index": "2",
            "item_id": "",
            "item_description": "Bagel",
            "item_raw_description": "",
            "item_brave_search_result": "",
            "item_quantity": "",
            "item_unit_price": "",
            "item_amount": "3.00",
            "item_discount_amount": "",
            "item_discount_description": "",
            "item_net_amount": "3.00",
            "item_line_type": "item",
            "item_category_id": "",
            "item_taxonomy_1": "",
            "item_taxonomy_2": "",
            "item_taxonomy_3": "",
            "item_taxonomy_4": "",
            "item_taxonomy_5": "",
            "item_taxonomy_6": "",
            "item_taxonomy_7": "",
            "item_taxonomy_8": "",
            "item_taxonomy_9": "",
            "item_confidence": "0.91",
        },
    ]


def test_writes_transaction_fields_on_each_csv_receipt_item_row():
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
        description="",
        amount="-11.00",
        currency="USD",
        receipt=Receipt(
            subtotal="10.00",
            total="11.00",
            items=[
                ReceiptItem(
                    description="Coffee",
                    amount="7.00",
                    category_id="Food & Dining > Fast Food & Coffee",
                )
            ],
        ),
    )
    output = StringIO()

    write_transaction_receipt_items_csv(transaction, output)

    rows = list(csv.DictReader(StringIO(output.getvalue())))
    assert rows[0]["transaction_id"] == "receipt_1"
    assert rows[0]["transaction_date"] == "2026-04-27"
    assert rows[0]["payee"] == "Coffee Shop"
    assert rows[0]["transaction_description"] == ""
    assert rows[0]["combined_description"] == "Coffee"
    assert rows[0]["transaction_amount"] == "-11.00"
    assert rows[0]["transaction_currency"] == "USD"
    assert rows[0]["ingestion_datetime"] == "2026-05-06T07:08:09+00:00"
    assert rows[0]["ingestion_filename"] == "receipt.pdf"
    assert rows[0]["ingestion_file_url"] == "file:///tmp/receipt.pdf"
    assert rows[0]["ingestion_file_sha256_hex"] == "0" * 64
    assert rows[0]["ingestion_type"] == "receipt_img"
    assert rows[0]["item_description"] == "Coffee"
    assert "item_category_id" not in rows[0]
    assert rows[0]["category_allocation.category_id"] == ("Food & Dining > Fast Food & Coffee")
    assert rows[0]["category_allocation.amount"] == "-7.00"


def test_json_output_preserves_nested_receipt_struct():
    receipt = Receipt(
        total="7.00",
        extraction=ExtractionMetadata(model="prebuilt-receipt", raw_text="Coffee 7.00"),
        items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")],
    )
    output = StringIO()

    write_receipt_json(receipt, output)

    assert '"total": "7.00"' in output.getvalue()
    assert '"rawText": "Coffee 7.00"' in output.getvalue()
    assert '"rawDescription": "COF"' in output.getvalue()
    assert '"items": [' in output.getvalue()


def test_transaction_json_output_wraps_nested_receipt_struct():
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
        receipt=Receipt(
            total="7.00",
            items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")],
        ),
    )
    output = StringIO()

    write_transaction_json(transaction, output)

    assert '"transactionDate": "2026-04-27"' in output.getvalue()
    assert '"payee": "Coffee Shop"' in output.getvalue()
    assert '"receipt": {' in output.getvalue()
    assert '"rawDescription": "COF"' in output.getvalue()


def test_transactions_json_output_writes_json_array():
    transactions = [
        Transaction(
            id="receipt_1",
            source=Source.receipt,
            transaction_date=date(2026, 4, 27),
            payee="Coffee Shop",
            amount="-7.00",
            currency="USD",
            receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
        ),
        Transaction(
            id="receipt_2",
            source=Source.receipt,
            transaction_date=date(2026, 4, 28),
            payee="Bakery",
            amount="-5.00",
            currency="USD",
            receipt=Receipt(items=[ReceiptItem(description="Bagel", amount="5.00")]),
        ),
    ]
    output = StringIO()

    write_transactions_json(transactions, output)

    payload = json.loads(output.getvalue())
    assert [transaction["id"] for transaction in payload] == ["receipt_1", "receipt_2"]
    assert payload[0]["receipt"]["items"][0]["description"] == "Coffee"
    assert payload[1]["receipt"]["items"][0]["description"] == "Bagel"


def test_transaction_firestore_document_uses_json_safe_aliases():
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
        receipt=Receipt(
            total="7.00",
            items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")],
        ),
    )

    document = transaction_firestore_document(transaction)

    assert document["transactionDate"] == "2026-04-27"
    assert document["source"] == "receipt"
    assert document["ingestionDatetime"] == "2026-05-06T07:08:09Z"
    assert document["ingestionFilename"] == "receipt.pdf"
    assert document["ingestionFileUrl"] == "file:///tmp/receipt.pdf"
    assert document["ingestionFileSha256Hex"] == "0" * 64
    assert document["ingestionType"] == "receipt_img"
    assert document["receipt"] == {
        "total": "7.00",
        "items": [
            {
                "description": "Coffee",
                "rawDescription": "COF",
                "amount": "7.00",
                "netAmount": "7.00",
                "lineType": "item",
            }
        ],
    }


def test_upsert_transaction_to_firestore_merges_transaction_document(
    caplog: pytest.LogCaptureFixture,
):
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    calls: list[tuple[str, str, dict[str, object], bool]] = []

    class FakeDocument:
        def __init__(self, collection: str, document_id: str) -> None:
            self.collection = collection
            self.document_id = document_id

        def set(self, document_data: dict[str, object], *, merge: bool = False) -> None:
            calls.append((self.collection, self.document_id, document_data, merge))

    class FakeCollection:
        def __init__(self, collection: str) -> None:
            self.collection = collection

        def document(self, document_id: str) -> FakeDocument:
            return FakeDocument(self.collection, document_id)

    class FakeFirestoreClient:
        def collection(self, collection_path: str) -> FakeCollection:
            return FakeCollection(collection_path)

    with caplog.at_level(logging.INFO, logger=ingest_receipts.__name__):
        upsert_transaction_to_firestore(
            transaction, client=FakeFirestoreClient(), collection="test-transactions"
        )

    assert calls == [
        (
            "test-transactions",
            "receipt_1",
            transaction_firestore_document(transaction),
            True,
        )
    ]
    assert (
        "Upserting transaction receipt_1 to Firestore collection test-transactions" in caplog.text
    )
    assert (
        "Firestore upsert completed for transaction receipt_1 in collection test-transactions"
        in caplog.text
    )


def test_create_firestore_client_sets_emulator_env_from_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    sentinel_client = object()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FIRESTORE_EMULATOR_HOST", raising=False)
    (tmp_path / ".receipts_ai.config").write_text(
        "\n".join(
            (
                "FIRESTORE_EMULATOR_HOST=127.0.0.1:8080",
                "FIREBASE_PROJECT_ID=receipts-ai-local",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        firestore_client,
        "_create_firestore_emulator_client",
        lambda: sentinel_client,
    )

    client = firestore_client.create_firestore_client()

    assert client is sentinel_client
    assert os.environ["FIRESTORE_EMULATOR_HOST"] == "127.0.0.1:8080"


def test_json_output_includes_receipt_item_category():
    receipt = Receipt(
        total="4.49",
        items=[
            ReceiptItem(
                description="Saltines",
                amount="4.49",
                category_id="Food & Dining > Groceries",
            )
        ],
    )
    output = StringIO()

    write_receipt_json(receipt, output)

    assert '"categoryId": "Food & Dining > Groceries"' in output.getvalue()


def test_main_can_enrich_items_with_brave_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(
        items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")]
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=receipt,
    )
    calls: list[Transaction] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--brave-search",
            "--brave-search-delay-seconds",
            "1.1",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(
        ingest_receipts,
        "analyze_receipt_file",
        fake_analyze_receipt_file,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        calls.append(transaction_to_enrich)
        assert request_delay_seconds == 1.1
        assert transaction_to_enrich.receipt is not None
        transaction_to_enrich.receipt.items[0].brave_search_result = "search payload"
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction,
    ) -> Transaction:
        assert transaction_to_clean.receipt is not None
        assert transaction_to_clean.receipt.items[0].brave_search_result == "search payload"
        transaction_to_clean.receipt.items[0].description = "Coffee"
        return transaction_to_clean

    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )

    main()

    assert calls == [transaction]
    assert receipt.items[0].brave_search_result == "search payload"


def test_main_after_filters_receipts_before_enrichment_and_upsert(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    calls: list[str] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        _ = transaction_to_enrich, request_delay_seconds
        calls.append("enrich")
        return transaction_to_enrich

    def fake_upsert_transaction_to_firestore(
        transaction_to_upsert: Transaction, *, collection: str
    ) -> None:
        _ = transaction_to_upsert, collection
        calls.append("upsert")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--after",
            "2026-04-28",
            "--brave-search",
            "--upsert-firestore",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "upsert_transaction_to_firestore",
        fake_upsert_transaction_to_firestore,
    )

    main()

    assert calls == []
    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert rows == []


def test_main_can_use_openai_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    calls: list[tuple[Path, str, object | None]] = []

    def fake_transaction_from_openai_receipt(
        path: Path, *, model: str, cache: object | None = None
    ) -> Transaction:
        calls.append((path, model, cache))
        return transaction

    def fail_analyze_receipt_file(_path: Path) -> object:
        raise AssertionError("azure pipeline should not run")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--pipeline",
            "openai",
            "--openai-model",
            "gpt-test",
            "--format",
            "json",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fail_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_openai_receipt",
        fake_transaction_from_openai_receipt,
    )

    main()

    assert calls == [(receipt_path, "gpt-test", None)]


def test_main_processes_multiple_receipts_as_combined_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_1_path = tmp_path / "receipt-1.pdf"
    receipt_2_path = tmp_path / "receipt-2.pdf"
    receipt_1_path.write_bytes(b"receipt 1")
    receipt_2_path.write_bytes(b"receipt 2")
    analyze_calls: list[Path] = []

    def fake_analyze_receipt_file(path: Path) -> Path:
        analyze_calls.append(path)
        return path

    def fake_transaction_from_document_intelligence_result(result: object) -> Transaction:
        assert isinstance(result, Path)
        path = result
        receipt_number = "001" if path == receipt_1_path else "002"
        amount = "1.00" if path == receipt_1_path else "2.00"
        return Transaction(
            id=f"transaction_{receipt_number}",
            source=Source.receipt,
            transaction_date=date(2026, 4, 27),
            payee=f"Store {receipt_number}",
            amount=f"-{amount}",
            currency="USD",
            receipt=Receipt(
                receipt_number=receipt_number,
                items=[
                    ReceiptItem(
                        description=f"Item {receipt_number}",
                        amount=amount,
                    )
                ],
            ),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["receipts-ai-ingest-receipts", str(receipt_1_path), str(receipt_2_path)],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert analyze_calls == [receipt_1_path, receipt_2_path]
    assert [row["transaction_id"] for row in rows] == ["transaction_001", "transaction_002"]
    assert [row["ingestion_file_sha256_hex"] for row in rows] == [
        hashlib.sha256(b"receipt 1").hexdigest(),
        hashlib.sha256(b"receipt 2").hexdigest(),
    ]
    assert [row["item_description"] for row in rows] == ["Item 001", "Item 002"]


def test_main_wraps_brave_search_client_when_cache_file_is_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    cache_path = tmp_path / "api-cache.json"
    receipt = Receipt(
        items=[ReceiptItem(description="Coffee", raw_description="COF", amount="7.00")]
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=receipt,
    )

    class FakeBraveClient:
        def search(self, query: str) -> dict[str, object]:
            return {"query": query}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_analyze_receipt_file(path: Path, *, cache: object) -> dict[str, Path]:
        assert cache.__class__.__name__ == "JsonCallCache"
        return {"result": path}

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction,
        *,
        client: object,
        request_delay_seconds: float | None = None,
    ) -> Transaction:
        assert request_delay_seconds is None
        assert client.__class__.__name__ == "CachedBraveSearchClient"
        assert transaction_to_enrich.receipt is not None
        transaction_to_enrich.receipt.items[0].brave_search_result = "search payload"
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction, *, client: object
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedCategoryModelClient"
        assert transaction_to_clean.receipt is not None
        assert transaction_to_clean.receipt.items[0].brave_search_result == "search payload"
        return transaction_to_clean

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--brave-search",
            "--cache-file",
            str(cache_path),
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(ingest_receipts, "create_brave_search_client", lambda: FakeBraveClient())
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )

    main()

    assert receipt.items[0].brave_search_result == "search payload"


def test_main_wraps_ollama_client_when_cache_file_is_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    cache_path = tmp_path / "api-cache.json"
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )

    class FakeBraveClient:
        def search(self, query: str) -> dict[str, object]:
            return {"query": query}

    class FakeCategoryClient:
        def complete(self, prompt: str) -> str:
            return "Food & Dining" if "top level" in prompt else "Fast Food & Coffee"

    def fake_analyze_receipt_file(path: Path, *, cache: object) -> dict[str, Path]:
        assert cache.__class__.__name__ == "JsonCallCache"
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction,
        *,
        client: object,
        request_delay_seconds: float | None = None,
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedBraveSearchClient"
        assert request_delay_seconds is None
        return transaction_to_enrich

    def fake_categorize_receipt_items(
        transaction_to_categorize: Transaction, *, client: object
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedCategoryModelClient"
        assert transaction_to_categorize.receipt is not None
        transaction_to_categorize.receipt.items[
            0
        ].category_id = "Food & Dining > Fast Food & Coffee"
        return transaction_to_categorize

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction, *, client: object
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedCategoryModelClient"
        assert transaction_to_clean.receipt is not None
        transaction_to_clean.receipt.items[0].description = "Coffee"
        return transaction_to_clean

    def fake_classify_receipt_items_by_product_taxonomy(
        transaction_to_classify: Transaction, *, client: object
    ) -> Transaction:
        assert client.__class__.__name__ == "CachedCategoryModelClient"
        assert transaction_to_classify.receipt is not None
        transaction_to_classify.receipt.items[0].taxonomy1 = "Food, Beverages & Tobacco"
        return transaction_to_classify

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--categorize-items",
            "--cache-file",
            str(cache_path),
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(ingest_receipts, "create_brave_search_client", lambda: FakeBraveClient())
    monkeypatch.setattr(
        ingest_receipts, "create_ollama_category_client", lambda: FakeCategoryClient()
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(ingest_receipts, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy",
        fake_classify_receipt_items_by_product_taxonomy,
    )

    main()

    assert transaction.receipt is not None
    assert transaction.receipt.items[0].category_id == "Food & Dining > Fast Food & Coffee"
    assert transaction.receipt.items[0].taxonomy1 == "Food, Beverages & Tobacco"


def test_main_can_categorize_items_after_brave_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(
        items=[ReceiptItem(description="Saltines", raw_description="NBSC SALTINE", amount="4.49")]
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=receipt,
    )
    calls: list[str] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        assert request_delay_seconds == 0.25
        calls.append("brave")
        assert transaction_to_enrich.receipt is not None
        transaction_to_enrich.receipt.items[0].brave_search_result = "search payload"
        return transaction_to_enrich

    def fake_categorize_receipt_items(transaction_to_categorize: Transaction) -> Transaction:
        calls.append("categorize")
        assert transaction_to_categorize.receipt is not None
        assert transaction_to_categorize.receipt.items[0].brave_search_result == "search payload"
        assert transaction_to_categorize.receipt.items[0].description == "Nabisco Saltine Crackers"
        transaction_to_categorize.receipt.items[0].category_id = "Food & Dining > Groceries"
        return transaction_to_categorize

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction,
    ) -> Transaction:
        calls.append("clean")
        assert transaction_to_clean.receipt is not None
        assert transaction_to_clean.receipt.items[0].brave_search_result == "search payload"
        transaction_to_clean.receipt.items[0].description = "Nabisco Saltine Crackers"
        return transaction_to_clean

    def fake_classify_receipt_items_by_product_taxonomy(
        transaction_to_classify: Transaction,
    ) -> Transaction:
        calls.append("taxonomy")
        assert transaction_to_classify.receipt is not None
        assert transaction_to_classify.receipt.items[0].brave_search_result == "search payload"
        transaction_to_classify.receipt.items[0].taxonomy1 = "Food, Beverages & Tobacco"
        return transaction_to_classify

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--categorize-items",
            "--brave-search-delay-seconds",
            "0.25",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(ingest_receipts, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy",
        fake_classify_receipt_items_by_product_taxonomy,
    )

    main()

    assert calls == ["brave", "clean", "categorize", "taxonomy"]
    assert receipt.items[0].category_id == "Food & Dining > Groceries"
    assert receipt.items[0].taxonomy1 == "Food, Beverages & Tobacco"


def test_main_categorizes_items_with_flattened_budget_categories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(items=[ReceiptItem(description="Saltines", amount="4.49")])
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=receipt,
    )

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        _ = request_delay_seconds
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(transaction_to_clean: Transaction) -> Transaction:
        return transaction_to_clean

    def fake_categorize_receipt_items(transaction_to_categorize: Transaction) -> Transaction:
        assert transaction_to_categorize.receipt is not None
        transaction_to_categorize.receipt.items[0].category_id = "Food & Dining > Groceries"
        return transaction_to_categorize

    def fake_classify_receipt_items_by_product_taxonomy(
        transaction_to_classify: Transaction,
    ) -> Transaction:
        return transaction_to_classify

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--categorize-items",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(ingest_receipts, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy",
        fake_classify_receipt_items_by_product_taxonomy,
    )

    main()

    assert receipt.items[0].category_id == "Food & Dining > Groceries"


def test_main_can_use_vector_product_taxonomy_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    receipt = Receipt(items=[ReceiptItem(description="Apple AirPods Pro 3", amount="249.00")])
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Apple",
        amount="-249.00",
        currency="USD",
        receipt=receipt,
    )
    calls: list[str] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        _ = request_delay_seconds
        calls.append("brave")
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(transaction_to_clean: Transaction) -> Transaction:
        calls.append("clean")
        return transaction_to_clean

    def fake_categorize_receipt_items(transaction_to_categorize: Transaction) -> Transaction:
        calls.append("categorize")
        assert transaction_to_categorize.receipt is not None
        transaction_to_categorize.receipt.items[0].category_id = "Shopping & Retail > Electronics"
        return transaction_to_categorize

    def fake_classify_receipt_items_by_product_taxonomy_vector_search(
        transaction_to_classify: Transaction,
    ) -> Transaction:
        calls.append("vector-taxonomy")
        assert transaction_to_classify.receipt is not None
        transaction_to_classify.receipt.items[0].taxonomy1 = "Electronics"
        transaction_to_classify.receipt.items[0].taxonomy2 = "Audio"
        transaction_to_classify.receipt.items[0].taxonomy3 = "Headphones"
        return transaction_to_classify

    def fail_greedy_taxonomy(_transaction: Transaction) -> Transaction:
        raise AssertionError("greedy taxonomy should not be called")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--categorize-items",
            "--product-taxonomy-method",
            "vector",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(ingest_receipts, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy",
        fail_greedy_taxonomy,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "classify_receipt_items_by_product_taxonomy_vector_search",
        fake_classify_receipt_items_by_product_taxonomy_vector_search,
    )

    main()

    assert calls == ["brave", "clean", "categorize", "vector-taxonomy"]
    assert receipt.items[0].category_id == "Shopping & Retail > Electronics"
    assert receipt.items[0].taxonomy1 == "Electronics"
    assert receipt.items[0].taxonomy2 == "Audio"
    assert receipt.items[0].taxonomy3 == "Headphones"


def test_main_can_upsert_processed_transaction_to_firestore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_path.write_bytes(b"receipt")
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[ReceiptItem(description="Coffee", amount="7.00")]),
    )
    upserts: list[tuple[Transaction, str]] = []

    def fake_analyze_receipt_file(path: Path) -> dict[str, Path]:
        return {"result": path}

    def fake_transaction_from_document_intelligence_result(_result: object) -> Transaction:
        return transaction

    def fake_enrich_receipt_items_with_brave_search(
        transaction_to_enrich: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        assert request_delay_seconds is None
        assert transaction_to_enrich.receipt is not None
        transaction_to_enrich.receipt.items[0].brave_search_result = "search payload"
        return transaction_to_enrich

    def fake_clean_receipt_item_descriptions(
        transaction_to_clean: Transaction,
    ) -> Transaction:
        assert transaction_to_clean.receipt is not None
        transaction_to_clean.receipt.items[0].description = "Clean Coffee"
        return transaction_to_clean

    def fake_upsert_transaction_to_firestore(
        transaction_to_upsert: Transaction, *, collection: str
    ) -> None:
        upserts.append((transaction_to_upsert, collection))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-receipts",
            "--brave-search",
            "--upsert-firestore",
            "--firestore-collection",
            "processed-transactions",
            str(receipt_path),
        ],
    )
    monkeypatch.setattr(ingest_receipts, "analyze_receipt_file", fake_analyze_receipt_file)
    monkeypatch.setattr(
        ingest_receipts,
        "transaction_from_document_intelligence_result",
        fake_transaction_from_document_intelligence_result,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "clean_receipt_item_descriptions",
        fake_clean_receipt_item_descriptions,
    )
    monkeypatch.setattr(
        ingest_receipts,
        "upsert_transaction_to_firestore",
        fake_upsert_transaction_to_firestore,
    )

    main()

    assert upserts == [(transaction, "processed-transactions")]
    assert transaction.receipt is not None
    assert transaction.receipt.items[0].description == "Clean Coffee"
    assert transaction.receipt.items[0].brave_search_result == "search payload"
