from __future__ import annotations

from pathlib import Path

import pytest

from receipts_ai.config import CONFIG_FILE_ENV_VAR, config_value, first_config_value


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


def test_config_value_ignores_inline_comments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    (tmp_path / ".receipts_ai.config").write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=config-key # disabled locally sometimes",
                "OPENAI_MODEL = gpt-test    # temporary override",
            )
        ),
        encoding="utf-8",
    )

    assert config_value("OPENAI_API_KEY") == "config-key"
    assert config_value("OPENAI_MODEL") == "gpt-test"


def test_config_value_preserves_hash_inside_quotes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    (tmp_path / ".receipts_ai.config").write_text(
        'OPENAI_MODEL="gpt-test#preview" # keep quoted hash\n',
        encoding="utf-8",
    )

    assert config_value("OPENAI_MODEL") == "gpt-test#preview"


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


def test_config_file_env_var_overrides_home_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    home_path = tmp_path / "home"
    home_path.mkdir()
    config_path = tmp_path / "dev.receipts_ai.config"
    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv(CONFIG_FILE_ENV_VAR, str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (home_path / ".receipts_ai.config").write_text(
        "OPENAI_API_KEY=home-key\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "OPENAI_API_KEY=dev-key\n",
        encoding="utf-8",
    )

    assert config_value("OPENAI_API_KEY") == "dev-key"


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
