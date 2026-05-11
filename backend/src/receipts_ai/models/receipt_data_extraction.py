from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from receipts_ai.models.transaction import LineType


class ReceiptExtractionPipeline(StrEnum):
    azure = "azure"
    visionkit_ollama = "visionkit_ollama"


class ReceiptDataExtractionMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    pipeline: Annotated[
        ReceiptExtractionPipeline,
        Field(description="Pipeline that produced this normalized receipt extraction."),
    ]
    model: Annotated[
        str | None,
        Field(
            description="Extractor model or model combination, such as prebuilt-receipt or a local Ollama model.",
            min_length=1,
        ),
    ] = None
    confidence: Annotated[
        float | None,
        Field(
            description="Overall extraction confidence reported by the pipeline.", ge=0.0, le=1.0
        ),
    ] = None
    raw_text: Annotated[
        str | None,
        Field(alias="rawText", description="Full OCR/extractor text content.", min_length=1),
    ] = None


class ExtractedReceiptItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    description: Annotated[
        str,
        Field(description="Receipt line item name or description.", min_length=1),
    ]
    quantity: Annotated[float | None, Field(gt=0.0)] = None
    unit_price: Annotated[
        str | None,
        Field(
            alias="unitPrice",
            description="Price for one unit of this item, as shown on the receipt.",
        ),
    ] = None
    amount: Annotated[
        str,
        Field(description="Total amount for this receipt line, as shown on the receipt."),
    ]
    discount_amount: Annotated[
        str | None,
        Field(alias="discountAmount", description="Discount amount applied to this item."),
    ] = None
    discount_description: Annotated[
        str | None,
        Field(
            alias="discountDescription",
            description="Receipt text that describes the item discount.",
        ),
    ] = None
    line_type: Annotated[
        LineType,
        Field(alias="lineType", description="Receipt line type."),
    ] = LineType.item
    confidence: Annotated[
        float | None,
        Field(description="Line-level extraction confidence.", ge=0.0, le=1.0),
    ] = None


class ReceiptDataExtraction(BaseModel):
    """
    Pipeline-neutral receipt extraction output before conversion into the app Transaction model.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    merchant_name: Annotated[
        str,
        Field(
            alias="merchantName",
            description="Merchant name extracted from the receipt.",
            min_length=1,
        ),
    ]
    transaction_date: Annotated[
        date,
        Field(alias="transactionDate", description="Receipt transaction date."),
    ]
    currency: Annotated[
        str,
        Field(
            description="ISO 4217 currency code inferred from receipt amounts.",
            pattern="^[A-Z]{3}$",
        ),
    ] = "USD"
    receipt_number: Annotated[
        str | None,
        Field(
            alias="receiptNumber",
            description="Receipt or transaction number shown by the merchant.",
            min_length=1,
        ),
    ] = None
    subtotal: Annotated[
        str | None,
        Field(description="Receipt subtotal before tax, tip, and other total-level adjustments."),
    ] = None
    total_tax: Annotated[
        str | None,
        Field(alias="totalTax", description="Total tax amount extracted from the receipt."),
    ] = None
    total: Annotated[
        str,
        Field(description="Receipt total as shown by the merchant or source document."),
    ]
    items: Annotated[
        list[ExtractedReceiptItem],
        Field(description="Extracted receipt lines.", min_length=1),
    ]
    extraction: ReceiptDataExtractionMetadata
