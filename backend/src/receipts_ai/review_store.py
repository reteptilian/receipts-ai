from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from receipts_ai.models.receipt_data_extraction import ReceiptDataExtraction
from receipts_ai.review_models import (
    ReceiptComparisonResult,
    ReceiptExtractionRecord,
    ReceiptReviewRecord,
    ReceiptReviewStatus,
    ReceiptSourceRecord,
)

DEFAULT_REVIEW_DB_PATH = Path(".receipts-review.sqlite")


class ReceiptReviewStore:
    def __init__(self, path: Path = DEFAULT_REVIEW_DB_PATH) -> None:
        self.path = path
        self._initialize()

    def upsert_source(self, *, sha256_hex: str, image_path: Path) -> ReceiptSourceRecord:
        now = _now()
        resolved_path = str(image_path.resolve())
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT created_at
                FROM receipt_sources
                WHERE sha256_hex = ?
                """,
                (sha256_hex,),
            ).fetchone()
            created_at = cast(str, existing["created_at"]) if existing is not None else now
            connection.execute(
                """
                INSERT INTO receipt_sources (
                    sha256_hex,
                    image_path,
                    file_url,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(sha256_hex) DO UPDATE SET
                    image_path = excluded.image_path,
                    file_url = excluded.file_url,
                    updated_at = excluded.updated_at
                """,
                (sha256_hex, resolved_path, image_path.resolve().as_uri(), created_at, now),
            )
        return ReceiptSourceRecord(
            sha256_hex=sha256_hex,
            image_path=resolved_path,
            file_url=image_path.resolve().as_uri(),
            created_at=_datetime_from_json(created_at),
            updated_at=_datetime_from_json(now),
        )

    def save_extraction(
        self,
        *,
        receipt_sha256_hex: str,
        pipeline: str,
        receipt_data: ReceiptDataExtraction,
        model: str | None = None,
        prompt: str | None = None,
        ocr_text: str | None = None,
        output_schema: dict[str, Any] | None = None,
        raw_response: str | None = None,
        created_at: datetime | None = None,
    ) -> ReceiptExtractionRecord:
        timestamp = _json_datetime(created_at or datetime.now(UTC))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO receipt_extractions (
                    receipt_sha256_hex,
                    pipeline,
                    model,
                    receipt_data_json,
                    prompt,
                    ocr_text,
                    output_schema_json,
                    raw_response,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_sha256_hex,
                    pipeline,
                    model,
                    _receipt_data_json(receipt_data),
                    prompt,
                    ocr_text,
                    _json_or_none(output_schema),
                    raw_response,
                    timestamp,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an extraction id")
            extraction_id = int(cursor.lastrowid)
        return ReceiptExtractionRecord(
            id=extraction_id,
            receipt_sha256_hex=receipt_sha256_hex,
            pipeline=pipeline,
            model=model,
            receipt_data=receipt_data,
            prompt=prompt,
            ocr_text=ocr_text,
            output_schema=output_schema,
            raw_response=raw_response,
            created_at=_datetime_from_json(timestamp),
        )

    def latest_extraction(
        self, receipt_sha256_hex: str, *, pipeline: str | None = None
    ) -> ReceiptExtractionRecord | None:
        sql = """
            SELECT *
            FROM receipt_extractions
            WHERE receipt_sha256_hex = ?
        """
        parameters: tuple[object, ...]
        if pipeline is None:
            sql += " ORDER BY id DESC LIMIT 1"
            parameters = (receipt_sha256_hex,)
        else:
            sql += " AND pipeline = ? ORDER BY id DESC LIMIT 1"
            parameters = (receipt_sha256_hex, pipeline)
        with self._connect() as connection:
            row = connection.execute(sql, parameters).fetchone()
        return _extraction_from_row(row) if row is not None else None

    def source(self, sha256_hex: str) -> ReceiptSourceRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM receipt_sources
                WHERE sha256_hex = ?
                """,
                (sha256_hex,),
            ).fetchone()
        return _source_from_row(row) if row is not None else None

    def sources(self) -> list[ReceiptSourceRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM receipt_sources
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [_source_from_row(row) for row in rows]

    def save_review(
        self,
        *,
        receipt_sha256_hex: str,
        corrected_receipt_data: ReceiptDataExtraction,
        status: ReceiptReviewStatus,
        source_extraction_id: int | None = None,
        notes: str | None = None,
        updated_at: datetime | None = None,
    ) -> ReceiptReviewRecord:
        timestamp = _json_datetime(updated_at or datetime.now(UTC))
        reviewed_at = timestamp if status == ReceiptReviewStatus.reviewed else None
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT created_at
                FROM receipt_reviews
                WHERE receipt_sha256_hex = ?
                """,
                (receipt_sha256_hex,),
            ).fetchone()
            created_at = cast(str, existing["created_at"]) if existing is not None else timestamp
            connection.execute(
                """
                INSERT INTO receipt_reviews (
                    receipt_sha256_hex,
                    status,
                    corrected_receipt_data_json,
                    source_extraction_id,
                    notes,
                    created_at,
                    updated_at,
                    reviewed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(receipt_sha256_hex) DO UPDATE SET
                    status = excluded.status,
                    corrected_receipt_data_json = excluded.corrected_receipt_data_json,
                    source_extraction_id = excluded.source_extraction_id,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at,
                    reviewed_at = excluded.reviewed_at
                """,
                (
                    receipt_sha256_hex,
                    status.value,
                    _receipt_data_json(corrected_receipt_data),
                    source_extraction_id,
                    notes,
                    created_at,
                    timestamp,
                    reviewed_at,
                ),
            )
        return ReceiptReviewRecord(
            receipt_sha256_hex=receipt_sha256_hex,
            status=status,
            corrected_receipt_data=corrected_receipt_data,
            source_extraction_id=source_extraction_id,
            notes=notes,
            created_at=_datetime_from_json(created_at),
            updated_at=_datetime_from_json(timestamp),
            reviewed_at=_datetime_from_json_or_none(reviewed_at),
        )

    def review(self, receipt_sha256_hex: str) -> ReceiptReviewRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM receipt_reviews
                WHERE receipt_sha256_hex = ?
                """,
                (receipt_sha256_hex,),
            ).fetchone()
        return _review_from_row(row) if row is not None else None

    def reviewed_receipt_data(self, receipt_sha256_hex: str) -> ReceiptDataExtraction | None:
        review = self.review(receipt_sha256_hex)
        if review is None or review.status != ReceiptReviewStatus.reviewed:
            return None
        return review.corrected_receipt_data

    def reviews(self) -> list[ReceiptReviewRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM receipt_reviews
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [_review_from_row(row) for row in rows]

    def save_comparison(self, result: ReceiptComparisonResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO receipt_comparisons (
                    receipt_sha256_hex,
                    candidate_extraction_id,
                    candidate_pipeline,
                    result_json,
                    score,
                    mismatch_count,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.receipt_sha256_hex,
                    result.candidate_extraction_id,
                    result.candidate_pipeline,
                    result.model_dump_json(by_alias=True),
                    result.score,
                    result.mismatch_count,
                    result.created_at.isoformat(),
                ),
            )

    def latest_comparisons(self, receipt_sha256_hex: str) -> list[ReceiptComparisonResult]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT result_json
                FROM receipt_comparisons
                WHERE receipt_sha256_hex = ?
                ORDER BY id DESC
                """,
                (receipt_sha256_hex,),
            ).fetchall()
        return [
            ReceiptComparisonResult.model_validate_json(cast(str, row["result_json"]))
            for row in rows
        ]

    def reviewed_training_examples(
        self,
    ) -> Iterable[tuple[ReceiptSourceRecord, ReceiptReviewRecord]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    sources.sha256_hex AS source_sha256_hex,
                    sources.image_path,
                    sources.file_url,
                    sources.created_at AS source_created_at,
                    sources.updated_at AS source_updated_at,
                    reviews.*
                FROM receipt_reviews reviews
                JOIN receipt_sources sources
                    ON sources.sha256_hex = reviews.receipt_sha256_hex
                WHERE reviews.status = ?
                ORDER BY reviews.updated_at ASC
                """,
                (ReceiptReviewStatus.reviewed.value,),
            ).fetchall()
        for row in rows:
            yield (
                ReceiptSourceRecord(
                    sha256_hex=cast(str, row["source_sha256_hex"]),
                    image_path=cast(str, row["image_path"]),
                    file_url=cast(str | None, row["file_url"]),
                    created_at=_datetime_from_json(cast(str, row["source_created_at"])),
                    updated_at=_datetime_from_json(cast(str, row["source_updated_at"])),
                ),
                _review_from_row(row),
            )

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA user_version = 1")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS receipt_sources (
                    sha256_hex TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    file_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS receipt_extractions (
                    id INTEGER PRIMARY KEY,
                    receipt_sha256_hex TEXT NOT NULL,
                    pipeline TEXT NOT NULL,
                    model TEXT,
                    receipt_data_json TEXT NOT NULL,
                    prompt TEXT,
                    ocr_text TEXT,
                    output_schema_json TEXT,
                    raw_response TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(receipt_sha256_hex) REFERENCES receipt_sources(sha256_hex)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS receipt_extractions_receipt_pipeline_idx
                ON receipt_extractions(receipt_sha256_hex, pipeline, id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS receipt_reviews (
                    receipt_sha256_hex TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    corrected_receipt_data_json TEXT NOT NULL,
                    source_extraction_id INTEGER,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    FOREIGN KEY(receipt_sha256_hex) REFERENCES receipt_sources(sha256_hex),
                    FOREIGN KEY(source_extraction_id) REFERENCES receipt_extractions(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS receipt_comparisons (
                    id INTEGER PRIMARY KEY,
                    receipt_sha256_hex TEXT NOT NULL,
                    candidate_extraction_id INTEGER,
                    candidate_pipeline TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    score REAL NOT NULL,
                    mismatch_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(receipt_sha256_hex) REFERENCES receipt_sources(sha256_hex),
                    FOREIGN KEY(candidate_extraction_id) REFERENCES receipt_extractions(id)
                )
                """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def _source_from_row(row: sqlite3.Row) -> ReceiptSourceRecord:
    return ReceiptSourceRecord(
        sha256_hex=cast(str, row["sha256_hex"]),
        image_path=cast(str, row["image_path"]),
        file_url=cast(str | None, row["file_url"]),
        created_at=_datetime_from_json(cast(str, row["created_at"])),
        updated_at=_datetime_from_json(cast(str, row["updated_at"])),
    )


def _extraction_from_row(row: sqlite3.Row) -> ReceiptExtractionRecord:
    return ReceiptExtractionRecord(
        id=cast(int, row["id"]),
        receipt_sha256_hex=cast(str, row["receipt_sha256_hex"]),
        pipeline=cast(str, row["pipeline"]),
        model=cast(str | None, row["model"]),
        receipt_data=ReceiptDataExtraction.model_validate_json(cast(str, row["receipt_data_json"])),
        prompt=cast(str | None, row["prompt"]),
        ocr_text=cast(str | None, row["ocr_text"]),
        output_schema=_json_object_or_none(cast(str | None, row["output_schema_json"])),
        raw_response=cast(str | None, row["raw_response"]),
        created_at=_datetime_from_json(cast(str, row["created_at"])),
    )


def _review_from_row(row: sqlite3.Row) -> ReceiptReviewRecord:
    return ReceiptReviewRecord(
        receipt_sha256_hex=cast(str, row["receipt_sha256_hex"]),
        status=ReceiptReviewStatus(cast(str, row["status"])),
        corrected_receipt_data=ReceiptDataExtraction.model_validate_json(
            cast(str, row["corrected_receipt_data_json"])
        ),
        source_extraction_id=cast(int | None, row["source_extraction_id"]),
        notes=cast(str | None, row["notes"]),
        created_at=_datetime_from_json(cast(str, row["created_at"])),
        updated_at=_datetime_from_json(cast(str, row["updated_at"])),
        reviewed_at=_datetime_from_json_or_none(cast(str | None, row["reviewed_at"])),
    )


def _receipt_data_json(receipt_data: ReceiptDataExtraction) -> str:
    return receipt_data.model_dump_json(by_alias=True, exclude_none=True)


def _json_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_object_or_none(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("stored JSON value is not an object")
    return cast(dict[str, Any], payload)


def _now() -> str:
    return _json_datetime(datetime.now(UTC))


def _json_datetime(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _datetime_from_json(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _datetime_from_json_or_none(value: str | None) -> datetime | None:
    if value is None:
        return None
    return _datetime_from_json(value)
