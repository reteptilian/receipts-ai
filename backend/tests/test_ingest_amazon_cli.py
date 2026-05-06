from __future__ import annotations

import csv
import hashlib
import sys
import zipfile
from datetime import date
from io import StringIO
from pathlib import Path

import pytest

from receipts_ai import ingest_amazon
from receipts_ai.ingest_amazon import (
    main,
    transactions_from_amazon_export_zip,
    transactions_from_amazon_orders_csv,
)
from receipts_ai.models.transaction import Source, Transaction

SAMPLE_ORDER_CSV = """ASIN,Billing Address,Carrier Name & Tracking Number,Currency,Gift Message,Gift Recipient Contact,Gift Sender Name,Item Serial Number,Order Date,Order ID,Order Status,Original Quantity,Payment Method Type,Product Condition,Product Name,Purchase Order Number,Ship Date,Shipment Item Subtotal,Shipment Item Subtotal Tax,Shipment Status,Shipping Address,Shipping Charge,Shipping Option,Total Amount,Total Discounts,Unit Price,Unit Price Tax,Website
B07XYYBHFT,REDACTED,REDACTED,USD,Not Available,Not Available,Not Available,Not Available,2026-05-01T23:57:58Z,114-9364152-5237852,Closed,1,Visa - 4637,New,"Gaiam Yoga Block - Supportive Latex-Free EVA Foam Soft Non-Slip Surface for Yoga, Pilates, Meditation (Black), 1 EA",Not Applicable,2026-05-02T20:34:44.221Z,31.57,3.32,Shipped,REDACTED,0,rush,10.6,0,9.59,1.01,Amazon.com
B01AVDVHTI,REDACTED,REDACTED,USD,Not Available,Not Available,Not Available,Not Available,2026-05-01T23:57:58Z,114-9364152-5237852,Closed,1,Visa - 4637,New,"Fit Simplify Resistance Loop Exercise Bands with Instruction Guide and Carry Bag, Set of 5",Not Applicable,2026-05-02T20:34:44.221Z,31.57,3.32,Shipped,REDACTED,0,rush,11.03,0,9.98,1.05,Amazon.com
B0DXKGB45T,REDACTED,REDACTED,USD,Not Available,Not Available,Not Available,Not Available,2026-05-01T23:57:58Z,114-9364152-5237852,Closed,1,Visa - 4637,New,Nike Women's Flex Classic 6 Pack Headband,Not Applicable,2026-05-02T20:34:44.221Z,31.57,3.32,Shipped,REDACTED,0,rush,13.26,0,12,1.26,Amazon.com
"""


def test_parses_amazon_orders_csv_as_itemized_transactions():
    transactions = transactions_from_amazon_orders_csv(SAMPLE_ORDER_CSV, source="Your Orders.csv")

    assert len(transactions) == 1
    transaction = transactions[0]
    assert transaction.id.startswith("amazon_order_")
    assert transaction.source == Source.amazon_order
    assert transaction.ingestion_datetime is not None
    assert transaction.ingestion_datetime.date() == date.today()
    assert transaction.ingestion_filename == "Your Orders.csv"
    assert transaction.ingestion_file_url is None
    assert (
        transaction.ingestion_file_sha256_hex
        == hashlib.sha256(SAMPLE_ORDER_CSV.encode("utf-8")).hexdigest()
    )
    assert transaction.ingestion_type == "amazon"
    assert transaction.external_id == "114-9364152-5237852"
    assert transaction.account_id == "amazon:Visa - 4637"
    assert transaction.transaction_date == date(2026, 5, 1)
    assert transaction.posted_date == date(2026, 5, 2)
    assert transaction.payee == "Amazon.com"
    assert transaction.amount == "-34.89"
    assert transaction.currency == "USD"
    assert transaction.kind == "expense"
    assert transaction.status == "posted"
    assert transaction.receipt is not None
    assert transaction.receipt.receipt_number == "114-9364152-5237852"
    assert transaction.receipt.total == "34.89"
    assert transaction.receipt.source_document_id == "Your Orders.csv"
    assert [item.description for item in transaction.receipt.items[:3]] == [
        "Gaiam Yoga Block - Supportive Latex-Free EVA Foam Soft Non-Slip Surface for Yoga, Pilates, Meditation (Black), 1 EA",
        "Fit Simplify Resistance Loop Exercise Bands with Instruction Guide and Carry Bag, Set of 5",
        "Nike Women's Flex Classic 6 Pack Headband",
    ]
    assert transaction.receipt.items[0].raw_description == transaction.receipt.items[0].description
    assert transaction.receipt.items[0].amount == "9.59"
    assert transaction.receipt.items[0].unit_price == "9.59"
    assert transaction.receipt.items[0].net_amount == "9.59"
    assert transaction.receipt.items[-1].line_type == "tax"
    assert transaction.receipt.items[-1].amount == "3.32"


def test_parses_combined_amazon_timestamp_cell():
    csv_content = SAMPLE_ORDER_CSV.replace(
        "2026-05-01T23:57:58Z",
        "2024-07-19T19:52:49Z and 2024-07-19T20:23:05+00:00",
    )

    transactions = transactions_from_amazon_orders_csv(csv_content)

    assert transactions[0].transaction_date == date(2024, 7, 19)


def test_skips_amazon_order_history_notice_rows():
    notice_row = (
        "Not Available,REDACTED,REDACTED,USD,Not Available,Not Available,Not Available,"
        'Not Available,"Due to technical limitations, please refer to your Order History '
        "(available in your Amazon account) for certain details about any orders placed prior "
        'to 2002",legacy-notice,Closed,1,Visa - 4637,Not Available,Not Available,'
        "Not Applicable,Not Available,Not Available,Not Available,Not Available,REDACTED,"
        "0,Not Available,0,0,0,0,Amazon.com\n"
    )

    transactions = transactions_from_amazon_orders_csv(SAMPLE_ORDER_CSV + notice_row)

    assert len(transactions) == 1
    assert transactions[0].external_id == "114-9364152-5237852"


def test_ignores_amazon_order_history_notice_ship_dates():
    notice = (
        "Due to technical limitations, please refer to your Order History "
        "(available in your Amazon account) for certain details about any orders placed prior "
        "to 2002"
    )
    csv_content = SAMPLE_ORDER_CSV.replace("2026-05-02T20:34:44.221Z", f'"{notice}"')

    transactions = transactions_from_amazon_orders_csv(csv_content)

    assert len(transactions) == 1
    assert transactions[0].posted_date is None


def test_finds_order_history_csv_by_basename_inside_zip(tmp_path: Path):
    export_path = tmp_path / "Your Orders.zip"
    with zipfile.ZipFile(export_path, "w") as archive:
        archive.writestr("Your Amazon Orders/Order History.csv", SAMPLE_ORDER_CSV)

    transactions = transactions_from_amazon_export_zip(export_path)

    assert len(transactions) == 1
    assert transactions[0].ingestion_filename == "Your Amazon Orders/Order History.csv"
    assert transactions[0].ingestion_file_url == export_path.resolve().as_uri()
    assert (
        transactions[0].ingestion_file_sha256_hex
        == hashlib.sha256(export_path.read_bytes()).hexdigest()
    )
    assert transactions[0].receipt is not None
    assert transactions[0].receipt.source_document_id == (
        f"{export_path}:Your Amazon Orders/Order History.csv"
    )


def test_requires_explicit_orders_csv_name_when_zip_has_multiple_matches(tmp_path: Path):
    export_path = tmp_path / "Your Orders.zip"
    with zipfile.ZipFile(export_path, "w") as archive:
        archive.writestr("one/Order History.csv", SAMPLE_ORDER_CSV)
        archive.writestr("two/Order History.csv", SAMPLE_ORDER_CSV)

    with pytest.raises(ValueError, match="pass --orders-csv-name"):
        transactions_from_amazon_export_zip(export_path)

    transactions = transactions_from_amazon_export_zip(
        export_path, orders_csv_name="two/Order History.csv"
    )

    assert len(transactions) == 1
    assert transactions[0].ingestion_filename == "two/Order History.csv"
    assert transactions[0].ingestion_file_url == export_path.resolve().as_uri()
    assert transactions[0].receipt is not None
    assert transactions[0].receipt.source_document_id == f"{export_path}:two/Order History.csv"


def test_main_writes_receipt_item_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    csv_path = tmp_path / "Order History.csv"
    csv_path.write_text(SAMPLE_ORDER_CSV, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["receipts-ai-ingest-amazon", str(csv_path)])

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert [row["receipt_number"] for row in rows] == [
        "114-9364152-5237852",
        "114-9364152-5237852",
        "114-9364152-5237852",
        "114-9364152-5237852",
    ]
    assert rows[0]["combined_description"].startswith("Gaiam Yoga Block")
    assert rows[0]["ingestion_filename"] == "Order History.csv"
    assert rows[0]["ingestion_file_url"] == csv_path.resolve().as_uri()
    assert (
        rows[0]["ingestion_file_sha256_hex"]
        == hashlib.sha256(SAMPLE_ORDER_CSV.encode("utf-8")).hexdigest()
    )
    assert rows[0]["ingestion_type"] == "amazon"
    assert rows[-1]["item_line_type"] == "tax"
    assert rows[-1]["category_allocation.amount"] == "-3.32"


def test_main_after_filters_amazon_transactions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    csv_path = tmp_path / "Order History.csv"
    older_order_rows = "\n".join(SAMPLE_ORDER_CSV.splitlines()[1:])
    older_order_rows = older_order_rows.replace(
        "2026-05-01T23:57:58Z",
        "2026-04-27T23:57:58Z",
    ).replace(
        "114-9364152-5237852",
        "114-9364152-5237000",
    )
    csv_path.write_text(f"{SAMPLE_ORDER_CSV}{older_order_rows}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["receipts-ai-ingest-amazon", "--after", "2026-05-01", str(csv_path)],
    )

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert {row["receipt_number"] for row in rows} == {"114-9364152-5237852"}
    assert {row["transaction_date"] for row in rows} == {"2026-05-01"}


def test_main_categorizes_without_cleaning_amazon_descriptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    csv_path = tmp_path / "Order History.csv"
    csv_path.write_text(SAMPLE_ORDER_CSV, encoding="utf-8")
    calls: list[str] = []

    def fake_enrich_receipt_items_with_brave_search(
        transaction: Transaction, *, request_delay_seconds: float | None = None
    ) -> Transaction:
        calls.append(f"brave:{request_delay_seconds}")
        assert transaction.receipt is not None
        assert transaction.receipt.items[0].description.startswith("Gaiam Yoga Block")
        transaction.receipt.items[0].brave_search_result = "search payload"
        return transaction

    def fake_categorize_receipt_items(transaction: Transaction) -> Transaction:
        calls.append("categorize")
        assert transaction.receipt is not None
        assert transaction.receipt.items[0].description.startswith("Gaiam Yoga Block")
        transaction.receipt.items[0].category_id = "Shopping > Sporting Goods"
        return transaction

    def fake_classify_receipt_items_by_product_taxonomy(transaction: Transaction) -> Transaction:
        calls.append("taxonomy")
        assert transaction.receipt is not None
        assert transaction.receipt.items[0].description.startswith("Gaiam Yoga Block")
        transaction.receipt.items[0].taxonomy1 = "Sporting Goods"
        return transaction

    monkeypatch.setattr(
        sys,
        "argv",
        ["receipts-ai-ingest-amazon", "--categorize", "--format", "json", str(csv_path)],
    )
    monkeypatch.setattr(
        ingest_amazon,
        "enrich_receipt_items_with_brave_search",
        fake_enrich_receipt_items_with_brave_search,
    )
    monkeypatch.setattr(ingest_amazon, "categorize_receipt_items", fake_categorize_receipt_items)
    monkeypatch.setattr(
        ingest_amazon,
        "classify_receipt_items_by_product_taxonomy",
        fake_classify_receipt_items_by_product_taxonomy,
    )

    main()

    assert calls == ["brave:None", "categorize", "taxonomy"]


def test_main_can_upsert_firestore(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    csv_path = tmp_path / "Order History.csv"
    csv_path.write_text(SAMPLE_ORDER_CSV, encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_upsert_transaction_to_firestore(transaction: Transaction, *, collection: str) -> None:
        calls.append((transaction.external_id or "", collection))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "receipts-ai-ingest-amazon",
            "--upsert-firestore",
            "--firestore-collection",
            "test-transactions",
            str(csv_path),
        ],
    )
    monkeypatch.setattr(
        ingest_amazon,
        "upsert_transaction_to_firestore",
        fake_upsert_transaction_to_firestore,
    )

    main()

    assert calls == [("114-9364152-5237852", "test-transactions")]
