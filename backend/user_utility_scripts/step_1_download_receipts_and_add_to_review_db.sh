#!/bin/bash

# Make any subsequent failure exit the script
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $SCRIPT_DIR/utilities.sh

verify_in_backend

# Download receipt images
# Before using this script, fill in your Google Drive folder url.
uv run receipts-ai-download-google-drive-folder GOOGLE_DRIVE_FOLDER_URL ${HOME}/data/financial/receipts  \
  --log-level DEBUG --skip-existing-by-hash  --google-oauth-credentials ${HOME}/data/financial/creds/credentials.json \
  --google-oauth-authorized-user ${HOME}/data/financial/creds/authorized_user.json

# Load receipt images into review db

########### WARNING: Using receipts_small_set instead of full downloaded set
uv run receipts-ai-review  --db ${REVIEW_DB} import --log-level DEBUG --pipeline azure \
  --cache-file ${PIPELINE_CACHE_FILE} ${HOME}/data/financial/receipts_small_set/*

echo "Complete!"