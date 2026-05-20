from __future__ import annotations

from pathlib import Path

import pytest

from receipts_ai.config import CONFIG_FILE_ENV_VAR
from receipts_ai.firestore_client import (
    FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR,
    FIRESTORE_EMULATOR_HOST_ENV_VAR,
)


@pytest.fixture(autouse=True)
def isolate_user_receipts_ai_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home_path = tmp_path / ".isolated-home"
    home_path.mkdir()

    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.delenv(CONFIG_FILE_ENV_VAR, raising=False)
    monkeypatch.delenv(FIRESTORE_EMULATOR_HOST_ENV_VAR, raising=False)
    monkeypatch.delenv(FIREBASE_SERVICE_ACCT_KEY_FILEPATH_ENV_VAR, raising=False)
