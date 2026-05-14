from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

from receipts_ai.budget_categories import (
    BUDGET_CATEGORY_GROUP_EXPENSE,
    BUDGET_CATEGORY_GROUP_INCOME,
    BUDGET_CATEGORY_GROUP_INTERNAL_TRANSFER,
    BudgetCategoryCatalog,
    load_budget_category_catalog,
)
from receipts_ai.config import add_config_file_argument, configure_config_file
from receipts_ai.firestore_client import DEFAULT_FIRESTORE_COLLECTION
from receipts_ai.firestore_transactions import (
    FirestoreTransactionClient,
    stream_transactions_from_firestore,
)
from receipts_ai.ingest_receipts import (
    TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES,
    transaction_receipt_item_rows,
    write_transactions_receipt_items_csv,
)

LOGGER = logging.getLogger(__name__)
EXPENSE_RECORDS_SHEET_TITLE = "expense transactions"
EXPENSE_BUDGET_SHEET_TITLE = "expense budget"
INCOME_RECORDS_SHEET_TITLE = "income transactions"
INCOME_BUDGET_SHEET_TITLE = "income budget"
INTERNAL_TRANSFER_RECORDS_SHEET_TITLE = "internal transfer transactions"
INTERNAL_TRANSFER_BUDGET_SHEET_TITLE = "internal transfer budget"
UNCATEGORIZED_RECORDS_SHEET_TITLE = "uncategorized transactions"
STUFF_SHEET_TITLE = "stuff"
DEFAULT_GOOGLE_SHEET_TITLE = "Sheet1"
SHEET_CATEGORY_GROUP_FIELDNAME = "category_allocation.group"
SHEET_FIELDNAMES: tuple[str, ...] = (
    *TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES,
    SHEET_CATEGORY_GROUP_FIELDNAME,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all Firestore transaction receipt items to CSV."
    )
    add_config_file_argument(parser)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write CSV output to a file instead of stdout.",
    )
    parser.add_argument(
        "--firestore-collection",
        default=DEFAULT_FIRESTORE_COLLECTION,
        help=f"Firestore collection to export from. Defaults to {DEFAULT_FIRESTORE_COLLECTION}.",
    )
    parser.add_argument(
        "--google-sheet-title",
        help="Create or update a Google Sheet with this title using OAuth authentication.",
    )
    parser.add_argument(
        "--google-sheet-id",
        help="Update an existing Google Sheet by spreadsheet ID using OAuth authentication.",
    )
    parser.add_argument(
        "--google-sheet-url",
        help="Update an existing Google Sheet by URL using OAuth authentication.",
    )
    parser.add_argument(
        "--google-oauth-credentials",
        type=Path,
        help="Path to OAuth client credentials JSON. Defaults to gspread's config path.",
    )
    parser.add_argument(
        "--google-oauth-authorized-user",
        type=Path,
        help="Path to OAuth authorized user JSON. Defaults to gspread's config path.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        help="Show logs at this level.",
    )
    args = parser.parse_args()
    configure_config_file(args.config_file)
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(name)s:%(message)s")

    if args.google_sheet_title or args.google_sheet_id or args.google_sheet_url:
        export_firestore_receipt_items_google_sheet(
            spreadsheet_title=args.google_sheet_title,
            spreadsheet_id=args.google_sheet_id,
            spreadsheet_url=args.google_sheet_url,
            oauth_credentials_path=args.google_oauth_credentials,
            oauth_authorized_user_path=args.google_oauth_authorized_user,
            collection=args.firestore_collection,
        )
        if args.output is None:
            return

    export_firestore_receipt_items_csv(
        output_path=args.output, collection=args.firestore_collection
    )


def export_firestore_receipt_items_csv(
    *,
    output_path: Path | None = None,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> None:
    if output_path is None:
        _export_firestore_receipt_items_csv_to_file(
            sys.stdout,
            client=client,
            collection=collection,
        )
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        _export_firestore_receipt_items_csv_to_file(
            file,
            client=client,
            collection=collection,
        )


def _export_firestore_receipt_items_csv_to_file(
    file: TextIO,
    *,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
) -> None:
    transactions = list(stream_transactions_from_firestore(client=client, collection=collection))
    write_transactions_receipt_items_csv(transactions, file)
    LOGGER.info(
        "Exported %d Firestore transaction(s) from collection %s",
        len(transactions),
        collection,
    )


def export_firestore_receipt_items_google_sheet(
    *,
    spreadsheet_title: str | None = None,
    spreadsheet_id: str | None = None,
    spreadsheet_url: str | None = None,
    oauth_credentials_path: Path | None = None,
    oauth_authorized_user_path: Path | None = None,
    gspread_client: Any | None = None,
    client: FirestoreTransactionClient | None = None,
    collection: str = DEFAULT_FIRESTORE_COLLECTION,
    category_catalog: BudgetCategoryCatalog | None = None,
) -> None:
    if (
        sum(value is not None for value in (spreadsheet_title, spreadsheet_id, spreadsheet_url))
        != 1
    ):
        raise ValueError(
            "Pass exactly one of spreadsheet_title, spreadsheet_id, or spreadsheet_url"
        )

    transactions = list(stream_transactions_from_firestore(client=client, collection=collection))
    rows = transaction_receipt_item_rows(transactions)
    catalog = category_catalog or load_budget_category_catalog()
    sheet_client = gspread_client or _create_gspread_oauth_client(
        oauth_credentials_path=oauth_credentials_path,
        oauth_authorized_user_path=oauth_authorized_user_path,
    )
    try:
        spreadsheet = _open_or_create_spreadsheet(
            sheet_client,
            spreadsheet_title=spreadsheet_title,
            spreadsheet_id=spreadsheet_id,
            spreadsheet_url=spreadsheet_url,
        )
        _write_export_spreadsheet(spreadsheet, rows, category_catalog=catalog)
    except Exception as exc:
        if gspread_client is not None or not _is_invalid_grant_refresh_error(exc):
            raise
        LOGGER.warning(
            "Google OAuth refresh token has expired or been revoked; "
            "starting browser reauthorization flow."
        )
        sheet_client = _create_gspread_oauth_client(
            oauth_credentials_path=oauth_credentials_path,
            oauth_authorized_user_path=oauth_authorized_user_path,
            force_reauthorize=True,
        )
        spreadsheet = _open_or_create_spreadsheet(
            sheet_client,
            spreadsheet_title=spreadsheet_title,
            spreadsheet_id=spreadsheet_id,
            spreadsheet_url=spreadsheet_url,
        )
        _write_export_spreadsheet(spreadsheet, rows, category_catalog=catalog)
    LOGGER.info(
        "Exported %d Firestore transaction(s) from collection %s to Google Sheet %s",
        len(transactions),
        collection,
        spreadsheet_title or spreadsheet_id or spreadsheet_url,
    )


def _create_gspread_oauth_client(
    *,
    oauth_credentials_path: Path | None,
    oauth_authorized_user_path: Path | None,
    force_reauthorize: bool = False,
) -> Any:
    import gspread
    import gspread.auth

    kwargs: dict[str, Any] = {}
    if oauth_credentials_path is not None:
        kwargs["credentials_filename"] = oauth_credentials_path
    if oauth_authorized_user_path is not None:
        kwargs["authorized_user_filename"] = oauth_authorized_user_path
    if force_reauthorize:
        authorized_user_filename = (
            oauth_authorized_user_path or gspread.auth.DEFAULT_AUTHORIZED_USER_FILENAME
        )
        _archive_authorized_user_file(Path(authorized_user_filename))
    return gspread.oauth(**kwargs)


def _archive_authorized_user_file(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    archived_path = path.with_name(f"{path.name}.revoked-{timestamp}")
    path.replace(archived_path)
    LOGGER.info("Archived revoked Google OAuth authorized user file to %s", archived_path)


def _is_invalid_grant_refresh_error(exc: BaseException) -> bool:
    from google.auth.exceptions import RefreshError

    if not isinstance(exc, RefreshError):
        return False
    for arg in exc.args:
        if isinstance(arg, dict):
            error_payload = cast("Mapping[str, object]", arg)
            if error_payload.get("error") == "invalid_grant":
                return True
    return "invalid_grant" in str(exc)


def _open_or_create_spreadsheet(
    sheet_client: Any,
    *,
    spreadsheet_title: str | None,
    spreadsheet_id: str | None,
    spreadsheet_url: str | None,
) -> Any:
    if spreadsheet_id is not None:
        return sheet_client.open_by_key(spreadsheet_id)
    if spreadsheet_url is not None:
        return sheet_client.open_by_url(spreadsheet_url)

    import gspread

    assert spreadsheet_title is not None
    try:
        return sheet_client.open(spreadsheet_title)
    except gspread.SpreadsheetNotFound:
        return sheet_client.create(spreadsheet_title)


def _write_export_spreadsheet(
    spreadsheet: Any,
    rows: list[dict[str, object | None]],
    *,
    category_catalog: BudgetCategoryCatalog,
) -> None:
    grouped_rows = _sheet_rows_by_category_group(rows, category_catalog)
    expense_records = _write_records_worksheet(
        spreadsheet,
        EXPENSE_RECORDS_SHEET_TITLE,
        grouped_rows[BUDGET_CATEGORY_GROUP_EXPENSE],
    )
    expense_budget = _reset_worksheet(spreadsheet, EXPENSE_BUDGET_SHEET_TITLE, rows=1000, cols=26)
    income_records = _write_records_worksheet(
        spreadsheet,
        INCOME_RECORDS_SHEET_TITLE,
        grouped_rows[BUDGET_CATEGORY_GROUP_INCOME],
    )
    income_budget = _reset_worksheet(spreadsheet, INCOME_BUDGET_SHEET_TITLE, rows=1000, cols=26)
    internal_transfer_records = _write_records_worksheet(
        spreadsheet,
        INTERNAL_TRANSFER_RECORDS_SHEET_TITLE,
        grouped_rows[BUDGET_CATEGORY_GROUP_INTERNAL_TRANSFER],
    )
    internal_transfer_budget = _reset_worksheet(
        spreadsheet, INTERNAL_TRANSFER_BUDGET_SHEET_TITLE, rows=1000, cols=26
    )
    _write_records_worksheet(spreadsheet, UNCATEGORIZED_RECORDS_SHEET_TITLE, grouped_rows[None])
    stuff = _reset_worksheet(spreadsheet, STUFF_SHEET_TITLE, rows=1000, cols=26)
    _add_budget_pivot_table(
        spreadsheet,
        records_sheet=expense_records,
        budget_sheet=expense_budget,
        row_count=len(grouped_rows[BUDGET_CATEGORY_GROUP_EXPENSE]) + 1,
    )
    _add_budget_pivot_table(
        spreadsheet,
        records_sheet=income_records,
        budget_sheet=income_budget,
        row_count=len(grouped_rows[BUDGET_CATEGORY_GROUP_INCOME]) + 1,
    )
    _add_budget_pivot_table(
        spreadsheet,
        records_sheet=internal_transfer_records,
        budget_sheet=internal_transfer_budget,
        row_count=len(grouped_rows[BUDGET_CATEGORY_GROUP_INTERNAL_TRANSFER]) + 1,
    )
    _add_stuff_pivot_table(
        spreadsheet,
        records_sheet=expense_records,
        stuff_sheet=stuff,
        row_count=len(grouped_rows[BUDGET_CATEGORY_GROUP_EXPENSE]) + 1,
    )
    _delete_default_google_sheet(spreadsheet)


def _write_records_worksheet(
    spreadsheet: Any,
    title: str,
    rows: list[dict[str, object | None]],
) -> Any:
    worksheet = _reset_worksheet(
        spreadsheet,
        title,
        rows=max(len(rows) + 1, 1),
        cols=len(SHEET_FIELDNAMES),
    )
    worksheet.update(_sheet_values(rows), range_name="A1", value_input_option="USER_ENTERED")
    worksheet.freeze(rows=1)
    return worksheet


def _reset_worksheet(spreadsheet: Any, title: str, *, rows: int, cols: int) -> Any:
    import gspread

    try:
        worksheet = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)

    worksheet.clear()
    worksheet.resize(rows=rows, cols=cols)
    return worksheet


def _delete_default_google_sheet(spreadsheet: Any) -> None:
    import gspread

    try:
        worksheet = spreadsheet.worksheet(DEFAULT_GOOGLE_SHEET_TITLE)
    except gspread.WorksheetNotFound:
        return

    spreadsheet.del_worksheet(worksheet)


def _sheet_values(rows: list[dict[str, object | None]]) -> list[list[Any]]:
    return [list(SHEET_FIELDNAMES)] + [
        ["" if row.get(fieldname) is None else row[fieldname] for fieldname in SHEET_FIELDNAMES]
        for row in rows
    ]


def _sheet_rows_by_category_group(
    rows: list[dict[str, object | None]],
    category_catalog: BudgetCategoryCatalog,
) -> dict[str | None, list[dict[str, object | None]]]:
    grouped_rows: dict[str | None, list[dict[str, object | None]]] = {
        BUDGET_CATEGORY_GROUP_EXPENSE: [],
        BUDGET_CATEGORY_GROUP_INCOME: [],
        BUDGET_CATEGORY_GROUP_INTERNAL_TRANSFER: [],
        None: [],
    }
    for row in rows:
        row_copy = dict(row)
        category_id = row.get("category_allocation.category_id")
        group = category_catalog.category_group(str(category_id) if category_id else None)
        row_copy[SHEET_CATEGORY_GROUP_FIELDNAME] = group
        grouped_rows.setdefault(group, []).append(row_copy)
    return grouped_rows


def _add_budget_pivot_table(
    spreadsheet: Any,
    *,
    records_sheet: Any,
    budget_sheet: Any,
    row_count: int,
) -> None:
    category_column = TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES.index(
        "category_allocation.category_id"
    )
    description_column = TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES.index("combined_description")
    amount_column = TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES.index("category_allocation.amount")
    spreadsheet.batch_update(
        {
            "requests": [
                {
                    "updateCells": {
                        "start": {
                            "sheetId": budget_sheet.id,
                            "rowIndex": 0,
                            "columnIndex": 0,
                        },
                        "rows": [
                            {
                                "values": [
                                    {
                                        "pivotTable": {
                                            "source": {
                                                "sheetId": records_sheet.id,
                                                "startRowIndex": 0,
                                                "startColumnIndex": 0,
                                                "endRowIndex": row_count,
                                                "endColumnIndex": len(SHEET_FIELDNAMES),
                                            },
                                            "rows": [
                                                {
                                                    "sourceColumnOffset": category_column,
                                                    "showTotals": True,
                                                    "sortOrder": "ASCENDING",
                                                },
                                                {
                                                    "sourceColumnOffset": description_column,
                                                    "showTotals": True,
                                                    "sortOrder": "ASCENDING",
                                                },
                                            ],
                                            "values": [
                                                {
                                                    "sourceColumnOffset": amount_column,
                                                    "summarizeFunction": "SUM",
                                                    "name": "Sum of category_allocation.amount",
                                                }
                                            ],
                                            "valueLayout": "HORIZONTAL",
                                        }
                                    }
                                ]
                            }
                        ],
                        "fields": "pivotTable",
                    }
                }
            ]
        }
    )


def _add_stuff_pivot_table(
    spreadsheet: Any,
    *,
    records_sheet: Any,
    stuff_sheet: Any,
    row_count: int,
) -> None:
    row_columns = [
        TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES.index(fieldname)
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
    amount_column = TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES.index("item_net_amount")
    spreadsheet.batch_update(
        {
            "requests": [
                {
                    "updateCells": {
                        "start": {
                            "sheetId": stuff_sheet.id,
                            "rowIndex": 0,
                            "columnIndex": 0,
                        },
                        "rows": [
                            {
                                "values": [
                                    {
                                        "pivotTable": {
                                            "source": {
                                                "sheetId": records_sheet.id,
                                                "startRowIndex": 0,
                                                "startColumnIndex": 0,
                                                "endRowIndex": row_count,
                                                "endColumnIndex": len(SHEET_FIELDNAMES),
                                            },
                                            "rows": [
                                                {
                                                    "sourceColumnOffset": column,
                                                    "showTotals": True,
                                                    "sortOrder": "ASCENDING",
                                                }
                                                for column in row_columns
                                            ],
                                            "values": [
                                                {
                                                    "sourceColumnOffset": amount_column,
                                                    "summarizeFunction": "SUM",
                                                    "name": "Sum of item_net_amount",
                                                }
                                            ],
                                            "valueLayout": "HORIZONTAL",
                                        }
                                    }
                                ]
                            }
                        ],
                        "fields": "pivotTable",
                    }
                }
            ]
        }
    )


if __name__ == "__main__":
    main()
