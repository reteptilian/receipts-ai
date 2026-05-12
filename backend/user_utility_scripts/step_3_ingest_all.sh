#!/bin/bash

# Make any subsequent failure exit the script
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $SCRIPT_DIR/utilities.sh

verify_in_backend

echo "Ingesting:"
echo "$(ls ${SMALL_RECEIPT_SET}/*)"

# REceipts
########### WARNING: RUNNING ONLY SMALL SET OF RECEIPTS
uv run receipts-ai-ingest-receipts --config-file ${CONFIG_FILE} --review-db ${REVIEW_DB} --brave-search  --log-level DEBUG \
  --brave-search-delay-seconds 1.5 --categorize --cache-file ${PIPELINE_CACHE_FILE} \
  --after ${RECORDS_START_DATE} --product-taxonomy-method vector  --upsert-firestore ${SMALL_RECEIPT_SET}/* 

# Amazon
uv run receipts-ai-ingest-amazon   --config-file ${CONFIG_FILE}  --brave-search  --log-level DEBUG --brave-search-delay-seconds 1.5 \
  --cache-file ${PIPELINE_CACHE_FILE}  --after ${RECORDS_START_DATE} --categorize --upsert-firestore \
  ${FINANCIAL_DATA_DIR}/amazon/*.zip


# qfx/ofx
uv run receipts-ai-ingest-statements  --config-file ${CONFIG_FILE}   --brave-search  --log-level DEBUG --brave-search-delay-seconds 1.5  \
  --cache-file ${PIPELINE_CACHE_FILE} --after ${RECORDS_START_DATE} --categorize --upsert-firestore  ${FINANCIAL_DATA_DIR}/ofx_qfx/*.?fx

# fidelity csv
uv run receipts-ai-ingest-statements  --config-file ${CONFIG_FILE}   --brave-search  --log-level DEBUG --brave-search-delay-seconds 1.5  --cache-file ${PIPELINE_CACHE_FILE}  \
  --after ${RECORDS_START_DATE} --categorize --statement-format fidelity-csv --upsert-firestore  ${FINANCIAL_DATA_DIR}/fidelity_csv/*.csv


echo "Complete!"

echo "Review data with: uv run receipts-ai-cli --config-file ${CONFIG_FILE}  --log-level DEBUG --log-file /tmp/receipts-ai.log"