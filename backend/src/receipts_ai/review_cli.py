from __future__ import annotations

import argparse
import sys
from pathlib import Path

from receipts_ai.cache import SqliteCallCache
from receipts_ai.review_service import import_receipt_for_review, write_training_jsonl
from receipts_ai.review_store import DEFAULT_REVIEW_DB_PATH, ReceiptReviewStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and compare receipt extraction output.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_REVIEW_DB_PATH,
        help=f"Review SQLite database. Defaults to {DEFAULT_REVIEW_DB_PATH}.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    app_parser = subparsers.add_parser("app", help="Launch the Streamlit review UI.")
    app_parser.add_argument(
        "--server-port",
        type=int,
        help="Optional Streamlit server port.",
    )

    import_parser = subparsers.add_parser("import", help="Run a baseline pipeline for receipts.")
    import_parser.add_argument("receipts", metavar="receipt", nargs="+", type=Path)
    import_parser.add_argument("--pipeline", choices=("azure", "visionkit_ollama"), default="azure")
    import_parser.add_argument("--cache-file", type=Path)
    import_parser.add_argument("--force", action="store_true")

    export_parser = subparsers.add_parser(
        "export-training",
        help="Export reviewed receipts as JSONL examples for OCR-text-to-JSON SFT.",
    )
    export_parser.add_argument("-o", "--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "app":
        _run_streamlit_app(args.db, server_port=args.server_port)
        return

    store = ReceiptReviewStore(args.db)
    if args.command == "import":
        cache = SqliteCallCache(args.cache_file) if args.cache_file is not None else None
        for receipt_path in args.receipts:
            extraction = import_receipt_for_review(
                receipt_path,
                store=store,
                pipeline=args.pipeline,
                cache=cache,
                force=args.force,
            )
            print(
                f"{receipt_path}: stored extraction {extraction.id} "
                f"for {extraction.receipt_sha256_hex}"
            )
        return

    if args.command == "export-training":
        with args.output.open("w", encoding="utf-8") as file:
            write_training_jsonl(store, file)
        print(f"wrote {args.output}")
        return


def _run_streamlit_app(db_path: Path, *, server_port: int | None) -> None:
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("review_streamlit_app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--db",
        str(db_path),
    ]
    if server_port is not None:
        sys.argv[3:3] = ["--server.port", str(server_port)]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
