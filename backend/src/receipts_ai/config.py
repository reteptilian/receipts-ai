from __future__ import annotations

import os
from pathlib import Path

CONFIG_FILENAME = ".receipts_ai.config"


def config_value(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value:
        return value
    return _config_file_values(str(Path.home())).get(key, default)


def first_config_value(keys: tuple[str, ...], default: str | None = None) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    config_values = _config_file_values(str(Path.home()))
    for key in keys:
        value = config_values.get(key)
        if value:
            return value
    return default


def _config_file_values(home: str) -> dict[str, str]:
    config_path = Path(home) / CONFIG_FILENAME
    if not config_path.is_file():
        return {}

    values: dict[str, str] = {}
    for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator or not key.strip():
            raise RuntimeError(
                f"{config_path}:{line_number} must use KEY=VALUE config variable format"
            )
        values[key.strip()] = value.strip()
    return values
