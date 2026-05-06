from __future__ import annotations

from pathlib import Path

import pytest

from receipts_ai.config import config_value, first_config_value


def test_config_value_reads_home_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".receipts_ai.config").write_text(
        "\n".join(
            (
                "# local receipts-ai config",
                "OPENAI_API_KEY=config-key",
                "OPENAI_MODEL = gpt-test",
            )
        ),
        encoding="utf-8",
    )

    assert config_value("OPENAI_API_KEY") == "config-key"
    assert config_value("OPENAI_MODEL") == "gpt-test"


def test_config_value_strips_matching_quotes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    (tmp_path / ".receipts_ai.config").write_text(
        "\n".join(
            (
                'OLLAMA_URL="http://msi:11434"',
                "OPENAI_MODEL='gpt-test'",
            )
        ),
        encoding="utf-8",
    )

    assert config_value("OLLAMA_URL") == "http://msi:11434"
    assert config_value("OPENAI_MODEL") == "gpt-test"


def test_environment_value_takes_precedence_over_home_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    (tmp_path / ".receipts_ai.config").write_text(
        "OPENAI_API_KEY=config-key\n",
        encoding="utf-8",
    )

    assert config_value("OPENAI_API_KEY") == "env-key"


def test_first_config_value_checks_keys_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIMARY_KEY", raising=False)
    monkeypatch.delenv("ALTERNATE_KEY", raising=False)
    (tmp_path / ".receipts_ai.config").write_text(
        "ALTERNATE_KEY=alternate-value\n",
        encoding="utf-8",
    )

    assert first_config_value(("PRIMARY_KEY", "ALTERNATE_KEY")) == "alternate-value"


def test_config_value_rejects_malformed_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".receipts_ai.config").write_text("NOT_A_PAIR\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="KEY=VALUE"):
        config_value("OPENAI_API_KEY")
