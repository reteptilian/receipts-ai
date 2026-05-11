from __future__ import annotations

import argparse
import os
from pathlib import Path

CONFIG_FILENAME = ".receipts_ai.config"
CONFIG_FILE_ENV_VAR = "RECEIPTS_AI_CONFIG_FILE"


def add_config_file_argument(
    parser: argparse.ArgumentParser, *, suppress_default: bool = False
) -> None:
    default = argparse.SUPPRESS if suppress_default else None
    parser.add_argument(
        "--config-file",
        type=Path,
        default=default,
        help=(
            "Path to a receipts-ai config file. Defaults to "
            f"~/{CONFIG_FILENAME}. Can also be set with {CONFIG_FILE_ENV_VAR}."
        ),
    )


def configure_config_file(config_file: Path | str | None) -> None:
    if config_file is not None:
        os.environ[CONFIG_FILE_ENV_VAR] = str(config_file)


def config_value(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value:
        return value
    return _config_file_values().get(key, default)


def first_config_value(keys: tuple[str, ...], default: str | None = None) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    config_values = _config_file_values()
    for key in keys:
        value = config_values.get(key)
        if value:
            return value
    return default


def _config_file_path() -> Path:
    config_file = os.environ.get(CONFIG_FILE_ENV_VAR)
    if config_file:
        return Path(config_file).expanduser()
    return Path.home() / CONFIG_FILENAME


def _config_file_values() -> dict[str, str]:
    config_path = _config_file_path()
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
        values[key.strip()] = _normalize_config_value(_strip_inline_comment(value))
    return values


def _normalize_config_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _strip_inline_comment(value: str) -> str:
    in_quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
            continue
        if char == "#" and in_quote is None:
            return value[:index]
    return value
