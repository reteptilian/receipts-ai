from __future__ import annotations

# pyright: reportPrivateUsage=false
from receipts_ai.categorization import load_budget_category_choices
from receipts_ai.firestore_transactions import (
    link_bank_statement_transaction_to_receipt,
    save_transaction_review_edits,
    transactions_from_firestore,
    unlink_bank_statement_transaction_from_receipt,
)

from receipts_ai_cli.screens.transaction_review import ReceiptItemsScreen, TransactionReviewScreen
from receipts_ai_cli.screens.transactions import ReceiptsAIApp
from receipts_ai_cli.transaction_helpers import TransactionLoader, _open_file_in_external_viewer

__all__ = [
    "ReceiptItemsScreen",
    "ReceiptsAIApp",
    "TransactionLoader",
    "TransactionReviewScreen",
    "link_bank_statement_transaction_to_receipt",
    "load_budget_category_choices",
    "save_transaction_review_edits",
    "transactions_from_firestore",
    "unlink_bank_statement_transaction_from_receipt",
    "_open_file_in_external_viewer",
    "main",
]


def main() -> None:
    ReceiptsAIApp().run()
