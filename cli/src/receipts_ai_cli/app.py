from __future__ import annotations

import argparse
import logging
from pathlib import Path

from receipts_ai.budget_categories import load_budget_category_options
from receipts_ai.categorization import load_budget_category_choices

# pyright: reportPrivateUsage=false
from receipts_ai.config import add_config_file_argument, config_value, configure_config_file
from receipts_ai.firestore_client import (
    FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR,
    FIRESTORE_EMULATOR_HOST_ENV_VAR,
)
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
    "load_budget_category_options",
    "save_transaction_review_edits",
    "transactions_from_firestore",
    "unlink_bank_statement_transaction_from_receipt",
    "_open_file_in_external_viewer",
    "main",
]

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Textual CLI for reviewing receipts-ai transactions."
    )
    add_config_file_argument(parser)
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write log output to this file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Set the application log level. Defaults to INFO.",
    )
    return parser.parse_args()


def _configure_logging(*, log_level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.NullHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(log_file, encoding="utf-8")]

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    if log_file is not None:
        LOGGER.info("Writing logs to %s", log_file)


def _log_firestore_configuration() -> None:
    emulator_host = config_value(FIRESTORE_EMULATOR_HOST_ENV_VAR)
    service_account_key_filepath = config_value(FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR)

    if emulator_host:
        LOGGER.info("Configured to use Firestore emulator at %s", emulator_host)
        return

    if service_account_key_filepath:
        LOGGER.info(
            "Configured to use Cloud Firestore with service account file %s",
            service_account_key_filepath,
        )
        return

    LOGGER.warning(
        "No Firestore configuration found. Set %s for the emulator or %s for Cloud Firestore.",
        FIRESTORE_EMULATOR_HOST_ENV_VAR,
        FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR,
    )


def main() -> None:
    args = _parse_args()
    configure_config_file(args.config_file)
    _configure_logging(log_level=args.log_level, log_file=args.log_file)
    _log_firestore_configuration()
    ReceiptsAIApp().run()
