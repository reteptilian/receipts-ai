from __future__ import annotations

import argparse
from pathlib import Path

from receipts_ai.document_intelligence import analyze_receipt_file, pretty_print_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a receipt with Azure Document Intelligence."
    )
    parser.add_argument("receipt", type=Path, help="Path to a receipt image or PDF.")
    args = parser.parse_args()

    result = analyze_receipt_file(args.receipt)
    pretty_print_analysis(result)


if __name__ == "__main__":
    main()
