from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections.abc import Iterable
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TextIO

from receipts_ai.brave_search import (
    CachedBraveSearchClient,
    create_brave_search_client,
    enrich_transactions_with_brave_search,
)
from receipts_ai.cache import JsonCallCache
from receipts_ai.categorization import (
    CachedCategoryModelClient,
    categorize_transactions,
    create_ollama_category_client,
)
from receipts_ai.ingest_receipts import (
    DEFAULT_FIRESTORE_COLLECTION,
    upsert_transaction_to_firestore,
)
from receipts_ai.models.transaction import Kind, Source, Status, Transaction

CSV_FIELDNAMES: tuple[str, ...] = (
    "transaction_id",
    "source",
    "external_id",
    "account_id",
    "transaction_date",
    "posted_date",
    "payee",
    "description",
    "brave_search_result",
    "amount",
    "currency",
    "kind",
    "status",
    "linked_transaction_ids",
    "category_allocations",
    "notes",
    "created_at",
    "updated_at",
)

LOGGER = logging.getLogger(__name__)
OFX_AGGREGATE_TAGS = ("STMTRS", "CCSTMTRS")
OFX_TRANSACTION_PATTERN = re.compile(r"<STMTTRN>\s*(.*?)\s*</STMTTRN>", re.IGNORECASE | re.DOTALL)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest one or more statement files."
    )
    parser.add_argument(
        "statements",
        metavar="statement",
        nargs="+",
        type=Path,
        help="Path to a statement file. Provide multiple paths to process them together.",
    )
    parser.add_argument(
        "--statement-format",
        choices=("ofx", "fidelity-csv"),
        default="ofx",
        help="Input statement format. Defaults to OFX/QFX.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format. CSV writes one row per transaction; JSON preserves the Transaction struct.",
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
        help="Populate each transaction braveSearchResult with Brave Search summaries.",
    )
    parser.add_argument(
        "--brave-search-delay-seconds",
        type=float,
        help="Sleep this many seconds between Brave Search requests.",
    )
    parser.add_argument(
        "--categorize",
        action="store_true",
        dest="categorize_transactions",
        help="Use Ollama to populate each transaction categoryAllocations.",
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
    for statement_path in args.statements:
        statement_transactions = transactions_from_file(
            statement_path, statement_format=args.statement_format
        )
        if args.brave_search or args.categorize_transactions:
            if cache is not None:
                enrich_transactions_with_brave_search(
                    statement_transactions,
                    client=CachedBraveSearchClient(
                        cache=cache, client_factory=create_brave_search_client
                    ),
                    request_delay_seconds=args.brave_search_delay_seconds,
                )
            else:
                enrich_transactions_with_brave_search(
                    statement_transactions,
                    request_delay_seconds=args.brave_search_delay_seconds,
                )
        if args.categorize_transactions:
            if category_client is not None:
                categorize_transactions(statement_transactions, client=category_client)
            else:
                categorize_transactions(statement_transactions)
        if args.upsert_firestore:
            for transaction in statement_transactions:
                upsert_transaction_to_firestore(
                    transaction, collection=args.firestore_collection
                )
        transactions.extend(statement_transactions)

    _write_transactions(transactions, output_format=args.format, output_path=args.output)


def transactions_from_ofx_file(statement_path: Path) -> list[Transaction]:
    return transactions_from_ofx(statement_path.read_text(encoding="utf-8-sig"), source=statement_path)


def transactions_from_file(statement_path: Path, *, statement_format: str) -> list[Transaction]:
    if statement_format == "ofx":
        return transactions_from_ofx_file(statement_path)
    if statement_format == "fidelity-csv":
        return transactions_from_fidelity_csv_file(statement_path)
    raise ValueError(f"unsupported statement format: {statement_format}")


def transactions_from_ofx(ofx: str, *, source: Path | str | None = None) -> list[Transaction]:
    transactions: list[Transaction] = []
    for statement_index, (statement_block, is_credit_card) in enumerate(
        _statement_blocks(ofx), start=1
    ):
        currency = (_tag_value(statement_block, "CURDEF") or "USD").upper()
        account_id = _account_id(statement_block)
        transaction_blocks = OFX_TRANSACTION_PATTERN.findall(statement_block)
        LOGGER.debug(
            "Parsed %d OFX transaction block(s) from statement %d in %s",
            len(transaction_blocks),
            statement_index,
            source,
        )
        for transaction_block in transaction_blocks:
            transactions.append(
                _transaction_from_block(
                    transaction_block,
                    account_id=account_id,
                    currency=currency,
                    is_credit_card=is_credit_card,
                )
            )
    return transactions


def transactions_from_fidelity_csv_file(statement_path: Path) -> list[Transaction]:
    return transactions_from_fidelity_csv(
        statement_path.read_text(encoding="utf-8-sig"), source=statement_path
    )


def transactions_from_fidelity_csv(
    content: str, *, source: Path | str | None = None
) -> list[Transaction]:
    rows = list(csv.reader(content.splitlines()))
    header_index = _fidelity_header_index(rows)
    fieldnames = rows[header_index]
    transactions: list[Transaction] = []
    for row_index, row_values in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if not any(value.strip() for value in row_values):
            continue
        if len(row_values) != len(fieldnames):
            LOGGER.debug("Skipping non-data Fidelity CSV row %d in %s", row_index, source)
            continue

        row = dict(zip(fieldnames, row_values, strict=True))
        if not _looks_like_fidelity_transaction_row(row):
            LOGGER.debug("Skipping non-transaction Fidelity CSV row %d in %s", row_index, source)
            continue

        transactions.append(_transaction_from_fidelity_row(row, row_index=row_index))

    LOGGER.debug("Parsed %d Fidelity CSV transaction row(s) from %s", len(transactions), source)
    return transactions


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
        _write_transactions_to_file(transactions, output_format=output_format, file=file)


def _write_transactions_to_file(
    transactions: list[Transaction] | tuple[Transaction, ...],
    *,
    output_format: str,
    file: TextIO,
) -> None:
    if output_format == "csv":
        write_transactions_csv(transactions, file)
        return
    if output_format == "json":
        if len(transactions) == 1:
            write_transaction_json(transactions[0], file)
        else:
            write_transactions_json(transactions, file)
        return
    raise ValueError(f"unsupported output format: {output_format}")


def write_transactions_csv(
    transactions: list[Transaction] | tuple[Transaction, ...], file: TextIO
) -> None:
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(_transaction_rows(transactions))


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


def _transaction_rows(transactions: Iterable[Transaction]) -> list[dict[str, object | None]]:
    rows: list[dict[str, object | None]] = []
    for transaction in transactions:
        document = transaction.model_dump(mode="json", by_alias=True, exclude_none=True)
        rows.append(
            {
                "transaction_id": transaction.id,
                "source": transaction.source.value,
                "external_id": transaction.external_id,
                "account_id": transaction.account_id,
                "transaction_date": transaction.transaction_date.isoformat(),
                "posted_date": transaction.posted_date.isoformat()
                if transaction.posted_date is not None
                else None,
                "payee": transaction.payee,
                "description": transaction.description,
                "brave_search_result": transaction.brave_search_result,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "kind": transaction.kind.value if transaction.kind is not None else None,
                "status": transaction.status.value if transaction.status is not None else None,
                "linked_transaction_ids": json.dumps(document.get("linkedTransactionIds", [])),
                "category_allocations": json.dumps(document.get("categoryAllocations", [])),
                "notes": transaction.notes,
                "created_at": document.get("createdAt"),
                "updated_at": document.get("updatedAt"),
            }
        )
    return rows


def _statement_blocks(ofx: str) -> list[tuple[str, bool]]:
    blocks: list[tuple[str, bool]] = []
    for tag in OFX_AGGREGATE_TAGS:
        blocks.extend(
            (block, tag.upper() == "CCSTMTRS")
            for block in re.findall(
                rf"<{tag}>\s*(.*?)\s*</{tag}>", ofx, flags=re.IGNORECASE | re.DOTALL
            )
        )
    return blocks or [(ofx, False)]


def _transaction_from_block(
    transaction_block: str,
    *,
    account_id: str | None,
    currency: str,
    is_credit_card: bool,
) -> Transaction:
    transaction_type = (_tag_value(transaction_block, "TRNTYPE") or "").upper()
    amount = _amount(_required_tag_value(transaction_block, "TRNAMT"))
    transaction_date = _ofx_date(_required_tag_value(transaction_block, "DTPOSTED"))
    posted_date = _optional_ofx_date(_tag_value(transaction_block, "DTAVAIL"))
    fitid = _tag_value(transaction_block, "FITID")
    name = _tag_value(transaction_block, "NAME")
    memo = _tag_value(transaction_block, "MEMO")
    description = _description(name=name, memo=memo)
    return Transaction(
        id=_transaction_id(account_id=account_id, fitid=fitid, transaction_block=transaction_block),
        source=Source.bank_statement,
        external_id=fitid,
        account_id=account_id,
        transaction_date=transaction_date,
        posted_date=posted_date if posted_date != transaction_date else None,
        payee=name if is_credit_card else None,
        description=_credit_card_description(name=name, memo=memo)
        if is_credit_card
        else description,
        amount=amount,
        currency=currency,
        kind=_kind(transaction_type=transaction_type, amount=amount),
        status=Status.posted,
    )


def _transaction_from_fidelity_row(row: dict[str, str], *, row_index: int) -> Transaction:
    transaction_date = _fidelity_date(_required_csv_value(row, "Run Date", row_index=row_index))
    action = _required_csv_value(row, "Action", row_index=row_index)
    amount = _amount(_required_csv_value(row, "Amount ($)", row_index=row_index))
    description = _fidelity_description(action=action, description=row.get("Description"))
    external_id = _fidelity_external_id(row)

    return Transaction(
        id=_transaction_id(
            account_id=None,
            fitid=external_id,
            transaction_block="|".join(row.get(fieldname, "") for fieldname in row),
        ),
        source=Source.bank_statement,
        external_id=external_id,
        transaction_date=transaction_date,
        description=description,
        amount=amount,
        currency="USD",
        kind=_kind(transaction_type="", amount=amount),
        status=Status.posted,
    )


def _tag_value(block: str, tag: str) -> str | None:
    xml_match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if xml_match:
        return _clean_value(xml_match.group(1))

    sgml_match = re.search(
        rf"<{tag}>\s*([^<\r\n]*(?:[^\S\r\n][^<\r\n]*)?)",
        block,
        flags=re.IGNORECASE,
    )
    if sgml_match:
        return _clean_value(sgml_match.group(1))
    return None


def _required_tag_value(block: str, tag: str) -> str:
    value = _tag_value(block, tag)
    if value is None:
        raise ValueError(f"OFX transaction is missing required <{tag}> field")
    return value


def _required_csv_value(row: dict[str, str], fieldname: str, *, row_index: int) -> str:
    value = _clean_value(row.get(fieldname, ""))
    if value is None:
        raise ValueError(f"Fidelity CSV row {row_index} is missing required {fieldname!r} field")
    return value


def _clean_value(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


def _account_id(statement_block: str) -> str | None:
    bank_id = _tag_value(statement_block, "BANKID")
    account_id = _tag_value(statement_block, "ACCTID")
    account_type = _tag_value(statement_block, "ACCTTYPE")
    if account_id is None:
        return None

    parts = [part for part in (bank_id, account_id, account_type) if part]
    return ":".join(parts)


def _amount(value: str) -> str:
    try:
        decimal = Decimal(value.replace(",", "").lstrip("+"))
    except InvalidOperation as error:
        raise ValueError(f"invalid statement transaction amount: {value}") from error
    return format(decimal, "f")


def _ofx_date(value: str) -> date:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})", value.strip())
    if not match:
        raise ValueError(f"invalid OFX date: {value}")
    year, month, day = (int(part) for part in match.groups())
    return date(year, month, day)


def _fidelity_date(value: str) -> date:
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value.strip())
    if not match:
        raise ValueError(f"invalid Fidelity CSV date: {value}")
    month, day, year = (int(part) for part in match.groups())
    return date(year, month, day)


def _optional_ofx_date(value: str | None) -> date | None:
    return _ofx_date(value) if value is not None else None


def _description(*, name: str | None, memo: str | None) -> str | None:
    if memo and memo != name:
        return memo
    return name


def _credit_card_description(*, name: str | None, memo: str | None) -> str | None:
    if memo and memo != name:
        return memo
    return None


def _fidelity_description(*, action: str, description: str | None) -> str:
    cleaned_description = _clean_value(description or "")
    if cleaned_description is None or cleaned_description == "No Description":
        return action
    return f"{action} {cleaned_description}"


def _kind(*, transaction_type: str, amount: str) -> Kind:
    if transaction_type in {"XFER"}:
        return Kind.transfer
    if transaction_type in {"ADJUSTMENT"}:
        return Kind.adjustment

    decimal = Decimal(amount)
    if decimal < 0:
        return Kind.expense
    if decimal > 0:
        return Kind.income
    return Kind.adjustment


def _transaction_id(
    *,
    account_id: str | None,
    fitid: str | None,
    transaction_block: str,
) -> str:
    stable_source = "|".join(part for part in (account_id, fitid) if part)
    if not stable_source:
        stable_source = transaction_block
    digest = hashlib.sha256(stable_source.encode("utf-8")).hexdigest()[:24]
    return f"bank_statement_{digest}"


def _fidelity_header_index(rows: list[list[str]]) -> int:
    for index, row in enumerate(rows):
        if row and row[0].strip() == "Run Date":
            return index
    raise ValueError("Fidelity CSV is missing the Run Date header row")


def _looks_like_fidelity_transaction_row(row: dict[str, str]) -> bool:
    return (
        _clean_value(row.get("Run Date", "")) is not None
        and _clean_value(row.get("Action", "")) is not None
        and _clean_value(row.get("Amount ($)", "")) is not None
    )


def _fidelity_external_id(row: dict[str, str]) -> str:
    digest = hashlib.sha256(
        "|".join(
            (
                row.get("Run Date", "").strip(),
                row.get("Action", "").strip(),
                row.get("Symbol", "").strip(),
                row.get("Description", "").strip(),
                row.get("Amount ($)", "").strip(),
                row.get("Cash Balance ($)", "").strip(),
                row.get("Settlement Date", "").strip(),
            )
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"fidelity_csv_{digest}"

if __name__ == "__main__":
    main()
