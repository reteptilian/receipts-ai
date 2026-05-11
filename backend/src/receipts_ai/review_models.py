from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from receipts_ai.models.receipt_data_extraction import ReceiptDataExtraction


class ReceiptReviewStatus(StrEnum):
    draft = "draft"
    reviewed = "reviewed"


class ReceiptSourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sha256_hex: Annotated[
        str,
        Field(alias="sha256Hex", pattern="^[a-f0-9]{64}$"),
    ]
    image_path: Annotated[str, Field(alias="imagePath", min_length=1)]
    file_url: Annotated[str | None, Field(alias="fileUrl", min_length=1)] = None
    created_at: Annotated[AwareDatetime, Field(alias="createdAt")]
    updated_at: Annotated[AwareDatetime, Field(alias="updatedAt")]


class ReceiptExtractionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: Annotated[int | None, Field(ge=1)] = None
    receipt_sha256_hex: Annotated[
        str,
        Field(alias="receiptSha256Hex", pattern="^[a-f0-9]{64}$"),
    ]
    pipeline: Annotated[str, Field(min_length=1)]
    model: Annotated[str | None, Field(min_length=1)] = None
    receipt_data: Annotated[ReceiptDataExtraction, Field(alias="receiptData")]
    prompt: Annotated[str | None, Field(min_length=1)] = None
    ocr_text: Annotated[str | None, Field(alias="ocrText", min_length=1)] = None
    output_schema: Annotated[dict[str, Any] | None, Field(alias="outputSchema")] = None
    raw_response: Annotated[str | None, Field(alias="rawResponse", min_length=1)] = None
    created_at: Annotated[AwareDatetime, Field(alias="createdAt")]


class ReceiptReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    receipt_sha256_hex: Annotated[
        str,
        Field(alias="receiptSha256Hex", pattern="^[a-f0-9]{64}$"),
    ]
    status: ReceiptReviewStatus
    corrected_receipt_data: Annotated[
        ReceiptDataExtraction,
        Field(alias="correctedReceiptData"),
    ]
    source_extraction_id: Annotated[int | None, Field(alias="sourceExtractionId", ge=1)] = None
    notes: str | None = None
    created_at: Annotated[AwareDatetime, Field(alias="createdAt")]
    updated_at: Annotated[AwareDatetime, Field(alias="updatedAt")]
    reviewed_at: Annotated[datetime | None, Field(alias="reviewedAt")] = None


class ReceiptComparisonField(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    path: Annotated[str, Field(min_length=1)]
    expected: str | None = None
    actual: str | None = None
    similarity: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    matches: bool


class ReceiptComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    receipt_sha256_hex: Annotated[
        str,
        Field(alias="receiptSha256Hex", pattern="^[a-f0-9]{64}$"),
    ]
    candidate_extraction_id: Annotated[int | None, Field(alias="candidateExtractionId", ge=1)] = (
        None
    )
    candidate_pipeline: Annotated[str, Field(alias="candidatePipeline", min_length=1)]
    field_count: Annotated[int, Field(alias="fieldCount", ge=0)]
    mismatch_count: Annotated[int, Field(alias="mismatchCount", ge=0)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    fields: list[ReceiptComparisonField]
    created_at: Annotated[AwareDatetime, Field(alias="createdAt")]
