from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

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
RECORDS_SHEET_TITLE = "records"
BUDGET_SHEET_TITLE = "budget"
DEFAULT_GOOGLE_SHEET_TITLE = "Sheet1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all Firestore transaction receipt items to CSV."
    )
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
    sheet_client = gspread_client or _create_gspread_oauth_client(
        oauth_credentials_path=oauth_credentials_path,
        oauth_authorized_user_path=oauth_authorized_user_path,
    )
    spreadsheet = _open_or_create_spreadsheet(
        sheet_client,
        spreadsheet_title=spreadsheet_title,
        spreadsheet_id=spreadsheet_id,
        spreadsheet_url=spreadsheet_url,
    )
    _write_export_spreadsheet(spreadsheet, rows)
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
) -> Any:
    import gspread

    kwargs: dict[str, Any] = {}
    if oauth_credentials_path is not None:
        kwargs["credentials_filename"] = oauth_credentials_path
    if oauth_authorized_user_path is not None:
        kwargs["authorized_user_filename"] = oauth_authorized_user_path
    return gspread.oauth(**kwargs)


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
) -> None:
    records = _reset_worksheet(
        spreadsheet,
        RECORDS_SHEET_TITLE,
        rows=max(len(rows) + 1, 1),
        cols=len(TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES),
    )
    budget = _reset_worksheet(spreadsheet, BUDGET_SHEET_TITLE, rows=1000, cols=26)
    values = _sheet_values(rows)
    records.update(values, range_name="A1", value_input_option="USER_ENTERED")
    records.freeze(rows=1)
    _add_budget_pivot_table(
        spreadsheet, records_sheet=records, budget_sheet=budget, row_count=len(values)
    )
    _delete_default_google_sheet(spreadsheet)


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
    return [list(TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES)] + [
        [
            "" if row.get(fieldname) is None else row[fieldname]
            for fieldname in TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES
        ]
        for row in rows
    ]


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
                                                "endColumnIndex": len(
                                                    TRANSACTION_RECEIPT_ITEMS_CSV_FIELDNAMES
                                                ),
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


if __name__ == "__main__":
    main()
