from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from receipts_ai.cache import JsonCallCache

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
    cache_path = tmp_path / "api-cache.json"
    cache = JsonCallCache(cache_path)
    cache.set("brave_search", {"query": "coffee"}, {"web": {"results": []}})
    cache.set("ollama", {"prompt": "choose"}, "Groceries")

    deleted_count = cache.invalidate_namespace("brave_search")

    assert deleted_count == 1
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["brave_search"] == []
    assert len(payload["ollama"]) == 1


def test_invalidate_cache_namespace_requires_existing_file(tmp_path: Path):
    cache_path = tmp_path / "api-cache.json"
    module = load_invalidate_cache_namespace()

    with pytest.raises(FileNotFoundError, match="cache file does not exist"):
        module.invalidate_cache_namespace(cache_path, "brave_search")


def test_invalidate_cache_namespace_can_create_file_when_requested(tmp_path: Path):
    cache_path = tmp_path / "api-cache.json"
    module = load_invalidate_cache_namespace()

    deleted_count = module.invalidate_cache_namespace(cache_path, "brave_search", allow_create=True)

    assert deleted_count == 0
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["brave_search"] == []
