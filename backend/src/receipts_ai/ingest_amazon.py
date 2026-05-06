from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TextIO

from receipts_ai.brave_search import (
    CachedBraveSearchClient,
    create_brave_search_client,
    enrich_receipt_items_with_brave_search,
)
from receipts_ai.cache import JsonCallCache
from receipts_ai.categorization import (
    CachedCategoryModelClient,
    CategoryModelClient,
    categorize_receipt_items,
    classify_receipt_items_by_product_taxonomy,
    classify_receipt_items_by_product_taxonomy_vector_search,
    create_ollama_category_client,
)
from receipts_ai.firestore_client import DEFAULT_FIRESTORE_COLLECTION
from receipts_ai.ingest_receipts import (
    populate_transaction_ingestion_metadata,
    sha256_hex,
    upsert_transaction_to_firestore,
    write_transactions_json,
    write_transactions_receipt_items_csv,
)
from receipts_ai.models.transaction import (
    IngestionType,
    Kind,
    LineType,
    Receipt,
    ReceiptItem,
    Source,
    Status,
    Transaction,
)

DEFAULT_ORDERS_CSV_NAME = "Order History.csv"
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Amazon order history from Your Orders.zip or Order History.csv."
    )
    parser.add_argument(
        "exports",
        metavar="export",
        nargs="+",
        type=Path,
        help="Path to an Amazon Your Orders.zip export or an Order History.csv file.",
    )
    parser.add_argument(
        "--orders-csv-name",
        default=DEFAULT_ORDERS_CSV_NAME,
        help=(
            "CSV member name to read inside zip files. Defaults to finding "
            f"{DEFAULT_ORDERS_CSV_NAME!r} by basename, case-insensitively."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format. CSV writes one row per receipt item; JSON preserves the Transaction struct.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Write output to a file instead of stdout."
    )
    parser.add_argument(
        "--brave-search",
        action="store_true",
        help="Populate each Amazon item braveSearchResult with Brave Search summaries.",
    )
    parser.add_argument(
        "--brave-search-delay-seconds",
        type=float,
        help="Sleep this many seconds between Brave Search requests.",
    )
    parser.add_argument(
        "--categorize",
        "--categorize-items",
        action="store_true",
        dest="categorize_items",
        help="Use Ollama to populate each Amazon item categoryId and product taxonomy.",
    )
    parser.add_argument(
        "--product-taxonomy-method",
        choices=("greedy", "vector"),
        default="greedy",
        help=(
            "Product taxonomy classification method used with --categorize. "
            "'greedy' walks the taxonomy tree with Ollama; 'vector' retrieves nearest "
            "taxonomy embedding paths and asks Ollama to rank them. Defaults to greedy."
        ),
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        help="Cache Brave Search and Ollama responses in this JSON file.",
    )
    parser.add_argument(
        "--upsert-firestore",
        action="store_true",
        help=(
            "Upsert the processed transactions into Cloud Firestore. Set "
            "FIRESTORE_EMULATOR_HOST for a local emulator or "
            "FIREBASE_SERVICE_ACCT_KEY_FILEPATH for production."
        ),
    )
    parser.add_argument(
        "--firestore-collection",
        default=DEFAULT_FIRESTORE_COLLECTION,
        help=f"Firestore collection to upsert into. Defaults to {DEFAULT_FIRESTORE_COLLECTION}.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        help="Show logs at this level.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(name)s:%(message)s")

    cache = JsonCallCache(args.cache_file) if args.cache_file is not None else None
    category_client = (
        CachedCategoryModelClient(cache=cache, client_factory=create_ollama_category_client)
        if cache is not None
        else None
    )

    transactions: list[Transaction] = []
    for export_path in args.exports:
        export_transactions = transactions_from_amazon_export(
            export_path,
            orders_csv_name=args.orders_csv_name,
        )
        for transaction in export_transactions:
            _enrich_and_categorize_transaction(
                transaction,
                brave_search=args.brave_search or args.categorize_items,
                brave_search_delay_seconds=args.brave_search_delay_seconds,
                categorize_items=args.categorize_items,
                product_taxonomy_method=args.product_taxonomy_method,
                cache=cache,
                category_client=category_client,
            )
            if args.upsert_firestore:
                upsert_transaction_to_firestore(transaction, collection=args.firestore_collection)
        transactions.extend(export_transactions)

    _write_transactions(transactions, output_format=args.format, output_path=args.output)


def transactions_from_amazon_export(
    export_path: Path, *, orders_csv_name: str = DEFAULT_ORDERS_CSV_NAME
) -> list[Transaction]:
    if zipfile.is_zipfile(export_path):
        return transactions_from_amazon_export_zip(export_path, orders_csv_name=orders_csv_name)
    return transactions_from_amazon_orders_csv_file(export_path)


def transactions_from_amazon_export_zip(
    export_path: Path, *, orders_csv_name: str = DEFAULT_ORDERS_CSV_NAME
) -> list[Transaction]:
    export_bytes = export_path.read_bytes()
    with zipfile.ZipFile(export_path) as archive:
        member_name = _orders_csv_member_name(archive.namelist(), orders_csv_name=orders_csv_name)
        with archive.open(member_name) as file:
            content = file.read().decode("utf-8-sig")
    return transactions_from_amazon_orders_csv(
        content,
        source=f"{export_path}:{member_name}",
        ingestion_filename=member_name,
        ingestion_file_sha256_hex=sha256_hex(export_bytes),
    )


def transactions_from_amazon_orders_csv_file(orders_csv_path: Path) -> list[Transaction]:
    content = orders_csv_path.read_bytes()
    return transactions_from_amazon_orders_csv(
        content.decode("utf-8-sig"),
        source=str(orders_csv_path),
        ingestion_filename=orders_csv_path.name,
        ingestion_file_sha256_hex=sha256_hex(content),
    )


def transactions_from_amazon_orders_csv(
    content: str,
    *,
    source: str | None = None,
    ingestion_filename: str | None = None,
    ingestion_file_sha256_hex: str | None = None,
) -> list[Transaction]:
    reader = csv.DictReader(content.splitlines())
    rows_by_order_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_index, row in enumerate(reader, start=2):
        if _is_amazon_order_history_notice_row(row):
            LOGGER.warning("Skipping Amazon order history notice row %d in %s", row_index, source)
            continue
        order_id = _required_csv_value(row, "Order ID", row_index=row_index)
        rows_by_order_id[order_id].append(row)

    transactions = [
        _transaction_from_amazon_order_rows(order_id, rows, source=source)
        for order_id, rows in rows_by_order_id.items()
    ]
    for transaction in transactions:
        populate_transaction_ingestion_metadata(
            transaction,
            ingestion_filename=ingestion_filename or _source_filename(source) or DEFAULT_ORDERS_CSV_NAME,
            ingestion_file_sha256_hex=ingestion_file_sha256_hex
            or sha256_hex(content.encode("utf-8")),
            ingestion_type=IngestionType.amazon,
        )
    transactions.sort(
        key=lambda transaction: (transaction.transaction_date, transaction.external_id or "")
    )
    transactions.reverse()
    LOGGER.debug("Parsed %d Amazon order transaction(s) from %s", len(transactions), source)
    return transactions


def _transaction_from_amazon_order_rows(
    order_id: str, rows: list[dict[str, str]], *, source: str | None
) -> Transaction:
    first_row = rows[0]
    order_date = _date_from_amazon_timestamp(
        _required_csv_value(first_row, "Order Date", row_index=0)
    )
    currency = (_clean_value(first_row.get("Currency", "")) or "USD").upper()
    payee = _clean_value(first_row.get("Website", "")) or "Amazon.com"
    total = sum((_money(row.get("Total Amount")) for row in rows), Decimal("0"))
    status = _status_from_amazon_rows(rows)
    items = [
        _receipt_item_from_amazon_row(row, order_id=order_id, index=index)
        for index, row in enumerate(rows, start=1)
    ]
    tax_total = sum(
        (_money(row.get("Unit Price Tax")) * _quantity_decimal(row) for row in rows), Decimal("0")
    )
    if tax_total != 0:
        items.append(
            ReceiptItem(
                id=_amazon_item_id(order_id, "tax", len(items) + 1),
                description="Sales tax",
                amount=_decimal_string(tax_total),
                net_amount=_decimal_string(tax_total),
                line_type=LineType.tax,
            )
        )

    shipping_total = _order_shipping_total(rows)
    if shipping_total != 0:
        items.append(
            ReceiptItem(
                id=_amazon_item_id(order_id, "shipping", len(items) + 1),
                description="Shipping",
                amount=_decimal_string(shipping_total),
                net_amount=_decimal_string(shipping_total),
                line_type=LineType.shipping,
            )
        )

    return Transaction(
        id=_amazon_transaction_id(order_id),
        source=Source.amazon_order,
        external_id=order_id,
        account_id=_amazon_account_id(first_row),
        transaction_date=order_date,
        posted_date=_posted_date(rows),
        payee=payee,
        description=_order_description(items),
        amount=_decimal_string(-total),
        currency=currency,
        kind=Kind.expense if total > 0 else Kind.adjustment,
        status=status,
        receipt=Receipt(
            id=f"amazon_receipt_{order_id}",
            source_document_id=source,
            receipt_number=order_id,
            subtotal=_decimal_string(
                sum((_item_net_subtotal(item) for item in items), Decimal("0"))
            ),
            total=_decimal_string(total),
            items=items,
        ),
    )


def _receipt_item_from_amazon_row(row: dict[str, str], *, order_id: str, index: int) -> ReceiptItem:
    quantity = _quantity_decimal(row)
    unit_price = _money(row.get("Unit Price"))
    amount = unit_price * quantity
    discount_amount = _optional_decimal_string(_money(row.get("Total Discounts")))
    product_name = _required_csv_value(row, "Product Name", row_index=index)
    return ReceiptItem(
        id=_amazon_item_id(order_id, _clean_value(row.get("ASIN", "")) or str(index), index),
        description=product_name,
        raw_description=product_name,
        quantity=float(quantity),
        unit_price=_decimal_string(unit_price),
        amount=_decimal_string(amount),
        discount_amount=discount_amount,
        discount_description="Amazon discount" if discount_amount is not None else None,
        net_amount=_decimal_string(amount + _money(row.get("Total Discounts"))),
        line_type=LineType.item,
    )


def _enrich_and_categorize_transaction(
    transaction: Transaction,
    *,
    brave_search: bool,
    brave_search_delay_seconds: float | None,
    categorize_items: bool,
    product_taxonomy_method: str,
    cache: JsonCallCache | None,
    category_client: CategoryModelClient | None,
) -> None:
    if brave_search:
        if cache is not None:
            enrich_receipt_items_with_brave_search(
                transaction,
                client=CachedBraveSearchClient(
                    cache=cache, client_factory=create_brave_search_client
                ),
                request_delay_seconds=brave_search_delay_seconds,
            )
        else:
            enrich_receipt_items_with_brave_search(
                transaction, request_delay_seconds=brave_search_delay_seconds
            )
    if categorize_items:
        if category_client is not None:
            categorize_receipt_items(transaction, client=category_client)
            _classify_receipt_items_by_product_taxonomy(
                transaction,
                method=product_taxonomy_method,
                client=category_client,
            )
        else:
            categorize_receipt_items(transaction)
            _classify_receipt_items_by_product_taxonomy(
                transaction,
                method=product_taxonomy_method,
            )


def _classify_receipt_items_by_product_taxonomy(
    transaction: Transaction,
    *,
    method: str,
    client: CategoryModelClient | None = None,
) -> Transaction:
    if method == "greedy":
        return (
            classify_receipt_items_by_product_taxonomy(transaction, client=client)
            if client is not None
            else classify_receipt_items_by_product_taxonomy(transaction)
        )
    if method == "vector":
        return (
            classify_receipt_items_by_product_taxonomy_vector_search(transaction, client=client)
            if client is not None
            else classify_receipt_items_by_product_taxonomy_vector_search(transaction)
        )
    raise ValueError(f"unsupported product taxonomy method: {method}")


def _write_transactions(
    transactions: list[Transaction] | tuple[Transaction, ...],
    *,
    output_format: str,
    output_path: Path | None,
) -> None:
    if output_path is None:
        _write_transactions_to_file(transactions, output_format=output_format, file=sys.stdout)
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        _write_transactions_to_file(transactions, output_format=output_format, file=file)


def _write_transactions_to_file(
    transactions: list[Transaction] | tuple[Transaction, ...],
    *,
    output_format: str,
    file: TextIO,
) -> None:
    if output_format == "csv":
        write_transactions_receipt_items_csv(transactions, file)
        return
    if output_format == "json":
        write_transactions_json(transactions, file)
        return
    raise ValueError(f"unsupported output format: {output_format}")


def _orders_csv_member_name(member_names: Iterable[str], *, orders_csv_name: str) -> str:
    exact_matches = [name for name in member_names if name == orders_csv_name]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(f"Amazon export contains multiple {orders_csv_name!r} entries")

    target_basename = Path(orders_csv_name).name.lower()
    basename_matches = [name for name in member_names if Path(name).name.lower() == target_basename]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        matches = ", ".join(sorted(basename_matches))
        raise ValueError(
            f"Amazon export contains multiple {target_basename!r} entries; "
            f"pass --orders-csv-name with one of: {matches}"
        )
    raise ValueError(f"Amazon export does not contain {orders_csv_name!r}")


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip().strip("'").strip('"')
    if not stripped or stripped == "Not Available" or stripped == "Not Applicable":
        return None
    return stripped


def _source_filename(source: str | None) -> str | None:
    if source is None:
        return None
    if ":" in source:
        return source.rsplit(":", maxsplit=1)[1]
    return Path(source).name


def _required_csv_value(row: dict[str, str], fieldname: str, *, row_index: int) -> str:
    value = _clean_value(row.get(fieldname))
    if value is None:
        location = f"row {row_index}" if row_index > 0 else "Amazon order row"
        raise ValueError(f"{location} is missing required {fieldname!r} field")
    return value


def _money(value: str | None) -> Decimal:
    cleaned = _clean_value(value)
    if cleaned is None:
        return Decimal("0")
    try:
        return Decimal(cleaned.replace(",", "").lstrip("+"))
    except InvalidOperation as error:
        raise ValueError(f"invalid Amazon money amount: {value}") from error


def _quantity_decimal(row: dict[str, str]) -> Decimal:
    value = _clean_value(row.get("Original Quantity"))
    if value is None:
        return Decimal("1")
    try:
        quantity = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"invalid Amazon quantity: {value}") from error
    return quantity if quantity > 0 else Decimal("1")


def _decimal_string(value: Decimal) -> str:
    normalized = value.quantize(Decimal("0.01"))
    return format(normalized, "f")


def _optional_decimal_string(value: Decimal) -> str | None:
    return _decimal_string(value) if value != 0 else None


def _date_from_amazon_timestamp(value: str) -> date:
    timestamp = value.strip().split(" and ", maxsplit=1)[0]
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        return datetime.fromisoformat(timestamp).date()
    except ValueError as error:
        raise ValueError(f"invalid Amazon timestamp: {value}") from error


def _optional_date_from_amazon_timestamp(value: str | None) -> date | None:
    cleaned = _clean_value(value)
    if cleaned is None or _is_amazon_order_history_notice(cleaned):
        return None
    return _date_from_amazon_timestamp(cleaned)


def _is_amazon_order_history_notice_row(row: dict[str, str]) -> bool:
    return _is_amazon_order_history_notice(row.get("Order Date"))


def _is_amazon_order_history_notice(value: str | None) -> bool:
    cleaned = _clean_value(value)
    return (
        cleaned is not None
        and "please refer to your order history" in cleaned.lower()
        and "orders placed prior to 2002" in cleaned.lower()
    )


def _posted_date(rows: list[dict[str, str]]) -> date | None:
    ship_dates = {
        ship_date
        for row in rows
        if (ship_date := _optional_date_from_amazon_timestamp(row.get("Ship Date"))) is not None
    }
    if len(ship_dates) == 1:
        ship_date = next(iter(ship_dates))
        order_date = _date_from_amazon_timestamp(
            _required_csv_value(rows[0], "Order Date", row_index=0)
        )
        return ship_date if ship_date != order_date else None
    return None


def _status_from_amazon_rows(rows: list[dict[str, str]]) -> Status:
    statuses = {
        (
            _clean_value(row.get("Order Status")) or _clean_value(row.get("Shipment Status")) or ""
        ).lower()
        for row in rows
    }
    if any("cancel" in status for status in statuses):
        return Status.void
    if any(status == "authorized" for status in statuses):
        return Status.pending
    return Status.posted


def _amazon_account_id(row: dict[str, str]) -> str | None:
    payment_method = _clean_value(row.get("Payment Method Type"))
    return f"amazon:{payment_method}" if payment_method is not None else None


def _order_description(items: list[ReceiptItem]) -> str | None:
    product_items = [item for item in items if item.line_type == LineType.item]
    if not product_items:
        return None
    first_description = product_items[0].description
    remaining_count = len(product_items) - 1
    return (
        first_description
        if remaining_count == 0
        else f"{first_description} + {remaining_count} more"
    )


def _order_shipping_total(rows: list[dict[str, str]]) -> Decimal:
    nonzero_shipping_values = [
        _money(row.get("Shipping Charge"))
        for row in rows
        if _money(row.get("Shipping Charge")) != 0
    ]
    unique_values = set(nonzero_shipping_values)
    if len(unique_values) == 1:
        return next(iter(unique_values))
    return sum(nonzero_shipping_values, Decimal("0"))


def _item_net_subtotal(item: ReceiptItem) -> Decimal:
    if item.line_type != LineType.item:
        return Decimal("0")
    return Decimal(item.net_amount or item.amount)


def _amazon_transaction_id(order_id: str) -> str:
    digest = hashlib.sha256(order_id.encode("utf-8")).hexdigest()[:24]
    return f"amazon_order_{digest}"


def _amazon_item_id(order_id: str, item_key: str, index: int) -> str:
    digest = hashlib.sha256(f"{order_id}|{item_key}|{index}".encode()).hexdigest()[:24]
    return f"amazon_item_{digest}"


if __name__ == "__main__":
    main()
