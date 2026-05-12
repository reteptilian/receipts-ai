#!/bin/bash

# Make any subsequent failure exit the script
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $SCRIPT_DIR/utilities.sh

verify_in_backend

# export everything
uv run receipts-ai-export-firestore --config-file ${CONFIG_FILE}  --log-level DEBUG  --output ~/temp/export.csv --google-sheet-title spending_report  

echo "Complete!"