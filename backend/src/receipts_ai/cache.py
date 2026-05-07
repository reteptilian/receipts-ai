from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


class SqliteCallCache:
    """Small SQLite cache organized by service namespace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._migrate_json_cache_if_needed()
        self._initialize()

    def get(self, namespace: str, request: Mapping[str, object]) -> Any | None:
        request_json = _canonical_json(dict(request))
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT response_json
                FROM cache_entries
                WHERE namespace = ? AND request_json = ?
                """,
                (namespace, request_json),
            ).fetchone()
        if row is None:
            return None
        return json.loads(cast(str, row["response_json"]))

    def set(self, namespace: str, request: Mapping[str, object], response: Any) -> None:
        created_at = datetime.now(UTC).isoformat()
        request_json = _canonical_json(dict(request))
        response_json = _canonical_json(response)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_entries (
                    namespace,
                    request_json,
                    response_json,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, request_json) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = excluded.created_at
                """,
                (namespace, request_json, response_json, created_at),
            )

    def invalidate_namespace(self, namespace: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM cache_entries WHERE namespace = ?",
                (namespace,),
            )
            return cursor.rowcount

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA user_version = 1")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(namespace, request_json)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS cache_entries_namespace_idx
                ON cache_entries(namespace)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate_json_cache_if_needed(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        if self._looks_like_sqlite_database():
            return

        try:
            with self.path.open("r", encoding="utf-8") as file:
                payload: object = json.load(file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cache file is not a SQLite database: {self.path}") from error

        if not isinstance(payload, dict):
            raise RuntimeError(f"JSON cache file must contain an object: {self.path}")

        backup_path = _backup_path(self.path)
        shutil.copy2(self.path, backup_path)
        self.path.unlink()
        self._initialize()
        self._import_json_payload(cast(dict[str, object], payload))

    def _looks_like_sqlite_database(self) -> bool:
        try:
            with self.path.open("rb") as file:
                return file.read(16) == b"SQLite format 3\x00"
        except OSError as error:
            raise RuntimeError(f"could not read cache file: {self.path}") from error

    def _import_json_payload(self, payload: dict[str, object]) -> None:
        for namespace, entries_object in payload.items():
            if namespace == "version":
                continue
            if not isinstance(entries_object, list):
                raise RuntimeError(
                    f"cache namespace {namespace!r} must contain a JSON array: {self.path}"
                )
            entries = cast(list[object], entries_object)
            for entry_object in entries:
                if not isinstance(entry_object, dict):
                    raise RuntimeError(
                        f"cache namespace {namespace!r} contains a non-object entry: {self.path}"
                    )
                entry = cast(dict[str, object], entry_object)
                request = entry.get("request")
                if not isinstance(request, dict):
                    raise RuntimeError(
                        f"cache namespace {namespace!r} contains an entry without "
                        f"a request object: {self.path}"
                    )
                response = entry.get("response")
                self.set(namespace, cast(dict[str, object], request), response)


JsonCallCache = SqliteCallCache


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _backup_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.bak")
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}.bak.{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1
