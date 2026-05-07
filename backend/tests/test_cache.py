from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import ModuleType

import pytest

from receipts_ai.cache import SqliteCallCache

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = BACKEND_ROOT / "devtools" / "invalidate_cache_namespace.py"


def load_invalidate_cache_namespace() -> ModuleType:
    spec = importlib.util.spec_from_file_location("invalidate_cache_namespace", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalidate_namespace_deletes_entries_from_only_that_namespace(tmp_path: Path):
    cache_path = tmp_path / "api-cache.sqlite"
    cache = SqliteCallCache(cache_path)
    cache.set("brave_search", {"query": "coffee"}, {"web": {"results": []}})
    cache.set("ollama", {"prompt": "choose"}, "Groceries")

    deleted_count = cache.invalidate_namespace("brave_search")

    assert deleted_count == 1
    assert cache_entries(cache_path, "brave_search") == []
    assert len(cache_entries(cache_path, "ollama")) == 1


def test_invalidate_cache_namespace_requires_existing_file(tmp_path: Path):
    cache_path = tmp_path / "api-cache.sqlite"
    module = load_invalidate_cache_namespace()

    with pytest.raises(FileNotFoundError, match="cache file does not exist"):
        module.invalidate_cache_namespace(cache_path, "brave_search")


def test_invalidate_cache_namespace_can_create_file_when_requested(tmp_path: Path):
    cache_path = tmp_path / "api-cache.sqlite"
    module = load_invalidate_cache_namespace()

    deleted_count = module.invalidate_cache_namespace(cache_path, "brave_search", allow_create=True)

    assert deleted_count == 0
    assert cache_entries(cache_path, "brave_search") == []


def test_sqlite_cache_migrates_existing_json_cache(tmp_path: Path):
    cache_path = tmp_path / "api-cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "brave_search": [
                    {
                        "created_at": "2026-05-07T12:00:00+00:00",
                        "request": {"query": "coffee"},
                        "response": {"web": {"results": []}},
                    }
                ],
                "ollama": [
                    {
                        "created_at": "2026-05-07T12:00:00+00:00",
                        "request": {"prompt": "choose"},
                        "response": "Groceries",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cache = SqliteCallCache(cache_path)

    assert cache.get("brave_search", {"query": "coffee"}) == {"web": {"results": []}}
    assert cache.get("ollama", {"prompt": "choose"}) == "Groceries"
    assert (tmp_path / "api-cache.json.bak").exists()


def cache_entries(cache_path: Path, namespace: str) -> list[dict[str, object]]:
    with sqlite3.connect(cache_path) as connection:
        rows = connection.execute(
            """
            SELECT request_json, response_json
            FROM cache_entries
            WHERE namespace = ?
            ORDER BY id
            """,
            (namespace,),
        ).fetchall()
    return [
        {
            "request": json.loads(request_json),
            "response": json.loads(response_json),
        }
        for request_json, response_json in rows
    ]
