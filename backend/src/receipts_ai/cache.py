from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


class JsonCallCache:
    """Small JSON cache organized by service namespace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._payload = self._load()

    def get(self, namespace: str, request: Mapping[str, object]) -> Any | None:
        for entry in self._entries(namespace):
            if entry.get("request") == request:
                return entry.get("response")
        return None

    def set(self, namespace: str, request: Mapping[str, object], response: Any) -> None:
        entries = self._entries(namespace)
        stored_request = dict(request)
        replacement: dict[str, object] = {
            "created_at": datetime.now(UTC).isoformat(),
            "request": stored_request,
            "response": response,
        }
        for index, entry in enumerate(entries):
            if entry.get("request") == stored_request:
                entries[index] = replacement
                break
        else:
            entries.append(replacement)
        self._save()

    def invalidate_namespace(self, namespace: str) -> int:
        entries = self._entries(namespace)
        deleted_count = len(entries)
        entries.clear()
        self._save()
        return deleted_count

    def _load(self) -> dict[str, object]:
        if not self.path.exists():
            return {
                "version": 1,
                "azure_document_intelligence": [],
                "brave_search": [],
                "ollama": [],
            }

        with self.path.open("r", encoding="utf-8") as file:
            payload: object = json.load(file)
        if not isinstance(payload, dict):
            raise RuntimeError(f"cache file must contain a JSON object: {self.path}")

        cache = cast(dict[str, object], payload)
        cache.setdefault("version", 1)
        cache.setdefault("azure_document_intelligence", [])
        cache.setdefault("brave_search", [])
        cache.setdefault("ollama", [])
        return cache

    def _entries(self, namespace: str) -> list[dict[str, object]]:
        entries_object = self._payload.setdefault(namespace, [])
        if not isinstance(entries_object, list):
            raise RuntimeError(
                f"cache namespace {namespace!r} must contain a JSON array: {self.path}"
            )

        entries = cast(list[object], entries_object)
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"cache namespace {namespace!r} contains a non-object entry: {self.path}"
                )
        return cast(list[dict[str, object]], entries)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self._payload, file, indent=2, sort_keys=True)
            file.write("\n")
