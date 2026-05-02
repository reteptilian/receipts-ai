from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO, cast

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
    clean_receipt_item_descriptions,
    create_ollama_category_client,
)
from receipts_ai.document_intelligence import analyze_receipt_file
from receipts_ai.models.transaction import Receipt, Transaction
from receipts_ai.openai_receipt_extraction import (
    DEFAULT_OPENAI_MODEL,
    OPENAI_MODEL_ENV_VAR,
    transaction_from_openai_receipt,
)
from receipts_ai.receipt_extraction import transaction_from_document_intelligence_result

if TYPE_CHECKING:
    from firebase_admin import App

CSV_FIELDNAMES: tuple[str, ...] = (
    "transaction_id",
    "transaction_date",
    "payee",
    "transaction_amount",
    "transaction_currency",
    "receipt_id",
    "source_document_id",
    "receipt_number",
    "receipt_subtotal",
    "receipt_total",
    "extraction_model",
    "extraction_confidence",
    "item_index",
    "item_id",
    "item_description",
    "item_raw_description",
    "item_brave_search_result",
    "item_quantity",
    "item_unit_price",
    "item_amount",
    "item_discount_amount",
    "item_discount_description",
    "item_net_amount",
    "item_line_type",
    "item_category_id",
    "item_taxonomy_1",
    "item_taxonomy_2",
    "item_taxonomy_3",
    "item_taxonomy_4",
    "item_taxonomy_5",
    "item_taxonomy_6",
    "item_taxonomy_7",
    "item_taxonomy_8",
    "item_taxonomy_9",
    "item_confidence",
)

DEFAULT_FIRESTORE_COLLECTION = "transactions"
DEFAULT_FIREBASE_EMULATOR_PROJECT_ID = "receipts-ai-local"
FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR = "FIREBASE_SERVICE_ACCT_KEY_FILEPATH"
FIRESTORE_EMULATOR_HOST_ENV_VAR = "FIRESTORE_EMULATOR_HOST"
FIRESTORE_PROJECT_ID_ENV_VARS = (
    "FIREBASE_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
)
LOGGER = logging.getLogger(__name__)


class FirestoreDocumentReference(Protocol):
    def set(self, document_data: dict[str, Any], *, merge: bool = False) -> object: ...


class FirestoreCollectionReference(Protocol):
    def document(self, document_id: str) -> FirestoreDocumentReference: ...


class FirestoreClient(Protocol):
    def collection(self, collection_path: str) -> FirestoreCollectionReference: ...


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one or more receipts with Azure Document Intelligence."
    )
    parser.add_argument(
        "receipts",
        metavar="receipt",
        nargs="+",
        type=Path,
        help="Path to a receipt image or PDF. Provide multiple paths to process them together.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("azure", "openai"),
        default="azure",
        help=(
            "Receipt extraction pipeline to use. 'azure' keeps the existing Azure Document "
            "Intelligence pipeline; 'openai' sends the receipt directly to OpenAI."
        ),
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv(OPENAI_MODEL_ENV_VAR, DEFAULT_OPENAI_MODEL),
        help=(
            "OpenAI model for --pipeline openai. Can also be set with "
            f"{OPENAI_MODEL_ENV_VAR}. Defaults to {DEFAULT_OPENAI_MODEL}."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format. CSV writes one row per receipt item; JSON preserves the nested struct.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write output to a file instead of stdout.",
    )
    parser.add_argument(
        "--brave-search",
        action="store_true",
        help=(
            "Populate each receipt item braveSearchResult with Brave Search summaries "
            "and clean item descriptions with Ollama."
        ),
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
        help=(
            "Use Ollama to populate each receipt item categoryId and product taxonomy "
            "from Brave Search results."
        ),
    )
    parser.add_argument(
        "--flatten-budget-categories",
        action="store_true",
        help=(
            "With --categorize, ask Ollama to choose once from flattened "
            "'Top > Subcategory' budget category paths."
        ),
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
        help="Cache Azure Document Intelligence, OpenAI, Brave Search, and Ollama responses in this JSON file.",
    )
    parser.add_argument(
        "--upsert-firestore",
        action="store_true",
        help=(
            "Upsert the processed transaction into Cloud Firestore. Set "
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
    for receipt_path in args.receipts:
        transaction = _process_receipt(
            receipt_path, pipeline=args.pipeline, openai_model=args.openai_model, cache=cache
        )
        if transaction.receipt is None:
            raise ValueError(f"transaction from {receipt_path} does not contain a receipt")

        if args.brave_search or args.categorize_items:
            if cache is not None:
                enrich_receipt_items_with_brave_search(
                    transaction,
                    client=CachedBraveSearchClient(
                        cache=cache, client_factory=create_brave_search_client
                    ),
                    request_delay_seconds=args.brave_search_delay_seconds,
                )
            else:
                enrich_receipt_items_with_brave_search(
                    transaction, request_delay_seconds=args.brave_search_delay_seconds
                )
            if category_client is not None:
                clean_receipt_item_descriptions(transaction, client=category_client)
            else:
                clean_receipt_item_descriptions(transaction)
        if args.categorize_items:
            if category_client is not None:
                if args.flatten_budget_categories:
                    categorize_receipt_items(
                        transaction,
                        client=category_client,
                        use_flattened_categories=True,
                    )
                else:
                    categorize_receipt_items(transaction, client=category_client)
                _classify_receipt_items_by_product_taxonomy(
                    transaction,
                    method=args.product_taxonomy_method,
                    client=category_client,
                )
            else:
                if args.flatten_budget_categories:
                    categorize_receipt_items(transaction, use_flattened_categories=True)
                else:
                    categorize_receipt_items(transaction)
                _classify_receipt_items_by_product_taxonomy(
                    transaction,
                    method=args.product_taxonomy_method,
                )
        if args.upsert_firestore:
            upsert_transaction_to_firestore(transaction, collection=args.firestore_collection)
        transactions.append(transaction)

    _write_transactions(transactions, output_format=args.format, output_path=args.output)


def _write_transactions(
    transactions: list[Transaction] | tuple[Transaction, ...],
    *,
    output_format: str,
    output_path: Path | None = None,
) -> None:
    if output_path is None:
        _write_transactions_to_file(transactions, output_format=output_format, file=sys.stdout)
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        _write_transactions_to_file(
            transactions,
            output_format=output_format,
            file=file,
        )


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
        if len(transactions) == 1:
            write_transaction_json(transactions[0], file)
        else:
            write_transactions_json(transactions, file)
        return

    raise ValueError(f"unsupported output format: {output_format}")


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
            classify_receipt_items_by_product_taxonomy_vector_search(
                transaction,
                client=client,
            )
            if client is not None
            else classify_receipt_items_by_product_taxonomy_vector_search(transaction)
        )
    raise ValueError(f"unsupported product taxonomy method: {method}")


def _process_receipt(
    receipt_path: Path,
    *,
    pipeline: str,
    openai_model: str,
    cache: JsonCallCache | None,
) -> Transaction:
    if pipeline == "azure":
        result = (
            analyze_receipt_file(receipt_path, cache=cache)
            if cache is not None
            else analyze_receipt_file(receipt_path)
        )
        return transaction_from_document_intelligence_result(result)

    if pipeline == "openai":
        return transaction_from_openai_receipt(receipt_path, model=openai_model, cache=cache)

    raise ValueError(f"unsupported receipt pipeline: {pipeline}")


def create_firestore_client() -> FirestoreClient:
    emulator_host = os.getenv(FIRESTORE_EMULATOR_HOST_ENV_VAR)
    service_account_key_filepath = os.getenv(FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR)

    if emulator_host:
        LOGGER.info(
            "Creating Firestore client for emulator at %s using project %s",
            emulator_host,
            _firestore_project_id(),
        )
        return _create_firestore_emulator_client()

    if service_account_key_filepath:
        LOGGER.info(
            "Creating Firestore client from service account file %s",
            service_account_key_filepath,
        )
        return _create_firestore_service_account_client(Path(service_account_key_filepath))

    raise RuntimeError(
        "Set FIRESTORE_EMULATOR_HOST to use the local Firestore emulator or "
        "FIREBASE_SERVICE_ACCT_KEY_FILEPATH to use production Cloud Firestore."
    )


def upsert_transaction_to_firestore(
    transaction: Transaction,
    *,
    client: FirestoreClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> None:
    if not collection:
        raise ValueError("collection must not be empty")

    firestore_client = client if client is not None else create_firestore_client()
    document = transaction_firestore_document(transaction)
    item_count = len(transaction.receipt.items) if transaction.receipt is not None else 0
    LOGGER.info(
        "Upserting transaction %s to Firestore collection %s with %d receipt item(s)",
        transaction.id,
        collection,
        item_count,
    )
    LOGGER.debug(
        "Firestore document %s/%s top-level fields: %s",
        collection,
        transaction.id,
        sorted(document),
    )
    try:
        firestore_client.collection(collection).document(transaction.id).set(document, merge=True)
    except Exception:
        LOGGER.exception(
            "Firestore upsert failed for transaction %s in collection %s",
            transaction.id,
            collection,
        )
        raise
    LOGGER.info(
        "Firestore upsert completed for transaction %s in collection %s",
        transaction.id,
        collection,
    )


def transaction_firestore_document(transaction: Transaction) -> dict[str, Any]:
    return transaction.model_dump(mode="json", by_alias=True, exclude_none=True)


def _create_firestore_emulator_client() -> FirestoreClient:
    from firebase_admin import firestore
    from google.auth.credentials import AnonymousCredentials

    # from google.cloud import firestore

    project_id = _firestore_project_id()
    cred = AnonymousCredentials()
    LOGGER.debug("Initializing Firebase app for Firestore emulator with project %s", project_id)
    app = _firebase_app(
        name="receipts-ai-firestore-emulator", options={"projectId": project_id}, credential=cred
    )
    return cast(FirestoreClient, cast(object, firestore.client(app=app)))


def _create_firestore_service_account_client(service_account_key_filepath: Path) -> FirestoreClient:
    from firebase_admin import credentials, firestore

    if not service_account_key_filepath.is_file():
        raise RuntimeError(
            f"{FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR} must point to a service account JSON file"
        )

    LOGGER.debug("Initializing Firebase app with service account credentials")
    credential = credentials.Certificate(str(service_account_key_filepath))
    app = _firebase_app(name="receipts-ai-firestore-service-account", credential=credential)
    return cast(FirestoreClient, cast(object, firestore.client(app=app)))


def _firebase_app(
    *,
    name: str,
    credential: Any | None = None,
    options: dict[str, str] | None = None,
) -> App:
    import firebase_admin

    try:
        return cast("App", firebase_admin.get_app(name))
    except ValueError:
        initialize_app = cast(  # type: ignore[reportUnknownMemberType]
            "Callable[[Any | None, dict[str, str] | None, str], App]",
            firebase_admin.initialize_app,
        )
        return initialize_app(credential, options, name)


def _firestore_project_id() -> str:
    for env_var in FIRESTORE_PROJECT_ID_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value
    return DEFAULT_FIREBASE_EMULATOR_PROJECT_ID


def write_transaction_receipt_items_csv(transaction: Transaction, file: TextIO) -> None:
    if transaction.receipt is None:
        raise ValueError("transaction does not contain a receipt")

    write_transactions_receipt_items_csv([transaction], file)


def write_transactions_receipt_items_csv(
    transactions: list[Transaction] | tuple[Transaction, ...], file: TextIO
) -> None:
    for transaction in transactions:
        if transaction.receipt is None:
            raise ValueError(f"transaction {transaction.id} does not contain a receipt")

    writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for transaction in transactions:
        writer.writerows(_transaction_receipt_item_rows(transaction))


def write_transaction_json(transaction: Transaction, file: TextIO) -> None:
    file.write(transaction.model_dump_json(by_alias=True, indent=2, exclude_none=True))
    file.write("\n")


def write_transactions_json(
    transactions: list[Transaction] | tuple[Transaction, ...], file: TextIO
) -> None:
    file.write(
        json.dumps(
            [
                transaction.model_dump(mode="json", by_alias=True, exclude_none=True)
                for transaction in transactions
            ],
            indent=2,
        )
    )
    file.write("\n")


def write_receipt_items_csv(receipt: Receipt, file: TextIO) -> None:
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(_receipt_item_rows(receipt))


def write_receipt_json(receipt: Receipt, file: TextIO) -> None:
    file.write(receipt.model_dump_json(by_alias=True, indent=2, exclude_none=True))
    file.write("\n")


def _transaction_receipt_item_rows(
    transaction: Transaction,
) -> list[dict[str, object | None]]:
    if transaction.receipt is None:
        raise ValueError("transaction does not contain a receipt")

    return _receipt_item_rows(
        transaction.receipt,
        transaction_id=transaction.id,
        transaction_date=transaction.transaction_date.isoformat(),
        payee=transaction.payee,
        transaction_amount=transaction.amount,
        transaction_currency=transaction.currency,
    )


def _receipt_item_rows(
    receipt: Receipt,
    *,
    transaction_id: str | None = None,
    transaction_date: str | None = None,
    payee: str | None = None,
    transaction_amount: str | None = None,
    transaction_currency: str | None = None,
) -> list[dict[str, object | None]]:
    extraction = receipt.extraction
    return [
        {
            "transaction_id": transaction_id,
            "transaction_date": transaction_date,
            "payee": payee,
            "transaction_amount": transaction_amount,
            "transaction_currency": transaction_currency,
            "receipt_id": receipt.id,
            "source_document_id": receipt.source_document_id,
            "receipt_number": receipt.receipt_number,
            "receipt_subtotal": receipt.subtotal,
            "receipt_total": receipt.total,
            "extraction_model": extraction.model if extraction is not None else None,
            "extraction_confidence": extraction.confidence if extraction is not None else None,
            "item_index": index,
            "item_id": item.id,
            "item_description": item.description,
            "item_raw_description": item.raw_description,
            "item_brave_search_result": item.brave_search_result,
            "item_quantity": item.quantity,
            "item_unit_price": item.unit_price,
            "item_amount": item.amount,
            "item_discount_amount": item.discount_amount,
            "item_discount_description": item.discount_description,
            "item_net_amount": item.net_amount,
            "item_line_type": item.line_type.value if item.line_type is not None else None,
            "item_category_id": item.category_id,
            "item_taxonomy_1": item.taxonomy1,
            "item_taxonomy_2": item.taxonomy2,
            "item_taxonomy_3": item.taxonomy3,
            "item_taxonomy_4": item.taxonomy4,
            "item_taxonomy_5": item.taxonomy5,
            "item_taxonomy_6": item.taxonomy6,
            "item_taxonomy_7": item.taxonomy7,
            "item_taxonomy_8": item.taxonomy8,
            "item_taxonomy_9": item.taxonomy9,
            "item_confidence": item.confidence,
        }
        for index, item in enumerate(receipt.items, start=1)
    ]


if __name__ == "__main__":
    main()
