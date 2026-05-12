#!/bin/bash

CODEBASE="/Users/esbensen/code/experimental/receipts_ai"

PIPELINE_CACHE_FILE="${HOME}/data/financial/pipeline.cache.sqlite"
REVIEW_DB="${HOME}/data/financial/receipts-review.sqlite"
SMALL_RECEIPT_SET="${HOME}/data/financial/receipts_small_set"
FINANCIAL_DATA_DIR="${HOME}/data/financial/data"
RECORDS_START_DATE="2026-04-01"

if [[ -n "$RECEIPTS_ENV" ]]; then
    echo "Please set RECEIPTS_ENV to dev or qa."
fi

case "$RECEIPTS_ENV" in
  "dev"|"qa")
    echo "Using environment: $RECEIPTS_ENV"
    ;;
  *)
    echo "RECEIPTS_ENV must be set to dev or qa"
    exit 1
    ;;
esac


CONFIG_FILE="${HOME}/.receipts_ai.${RECEIPTS_ENV}.config"



verify_cwd() {
  local expected="$1"
  if [[ "$PWD" != "$expected" ]]; then
    echo "Incorrect working directory."
    echo "Current directory: $PWD"
    echo "Expected directory: $expected"
    exit 1
  fi
}
verify_in_backend() {
   verify_cwd "${CODEBASE}/backend"
}
