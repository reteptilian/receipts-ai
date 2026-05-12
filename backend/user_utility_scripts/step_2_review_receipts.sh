#!/bin/bash

# Make any subsequent failure exit the script
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $SCRIPT_DIR/utilities.sh

verify_in_backend

echo "Review all receipts"

echo "Hit control-c to exit the server"

uv run receipts-ai-review --db ~/data/financial/receipts-review.sqlite app --cache-file ${PIPELINE_CACHE_FILE}  2> /tmp/receipts-ai-review.error.log

echo "Complete!"