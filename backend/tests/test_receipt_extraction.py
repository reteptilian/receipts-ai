from __future__ import annotations

import json
from pathlib import Path

import pytest

from receipts_ai.receipt_extraction import (
    receipt_from_document_intelligence_result,
    transaction_from_document_intelligence_result,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "azure"


def test_extracts_receipt_items_from_azure_receipt_result():
    payload = json.loads((FIXTURE_DIR / "fred_meyer_receipt.json").read_text())

    transaction = transaction_from_document_intelligence_result(payload)
    receipt = transaction.receipt

    assert transaction.source == "receipt"
    assert transaction.payee == "FredMeyer"
    assert transaction.transaction_date.isoformat() == "2025-10-14"
    assert transaction.amount == "-17.46"
    assert transaction.currency == "USD"
    assert receipt is not None

    assert receipt.total == "17.46"
    assert receipt.subtotal is None
    assert receipt.extraction is not None
    assert receipt.extraction.model == "prebuilt-receipt"
    assert receipt.extraction.confidence == 0.971
    assert receipt.extraction.raw_text is not None
    assert receipt.extraction.raw_text.startswith("FredMeyer")
    assert [(item.description, item.amount, item.confidence) for item in receipt.items] == [
        ("NBSC SALTINE", "4.49", 0.979),
        ("FDFL BREAD", "6.99", 0.979),
        ("PARSLEY", "1.29", 0.982),
        ("SILK CASHEW", "4.69", 0.979),
    ]
    assert [item.raw_description for item in receipt.items] == [
        "NBSC SALTINE",
        "FDFL BREAD",
        "PARSLEY",
        "SILK CASHEW",
    ]


def test_extracts_optional_item_quantity_and_unit_price():
    payload = {
        "content": "Tea 2 3.50 7.00",
        "modelId": "prebuilt-receipt",
        "documents": [
            {
                "confidence": 0.9,
                "fields": {
                    "MerchantName": {"valueString": "Tea Shop"},
                    "TransactionDate": {"valueDate": "2026-04-27"},
                    "Total": {
                        "type": "currency",
                        "valueCurrency": {"amount": 7.0, "currencyCode": "USD"},
                    },
                    "Items": {
                        "type": "array",
                        "valueArray": [
                            {
                                "confidence": 0.8,
                                "type": "object",
                                "valueObject": {
                                    "Description": {"valueString": "Tea"},
                                    "Quantity": {"valueNumber": 2},
                                    "Price": {
                                        "type": "currency",
                                        "valueCurrency": {
                                            "amount": 3.5,
                                            "currencyCode": "USD",
                                        },
                                    },
                                    "TotalPrice": {
                                        "type": "currency",
                                        "valueCurrency": {
                                            "amount": 7.0,
                                            "currencyCode": "USD",
                                        },
                                    },
                                },
                            }
                        ],
                    },
                },
            }
        ],
    }

    receipt = receipt_from_document_intelligence_result(payload)

    assert receipt.total == "7.0"
    assert receipt.items[0].description == "Tea"
    assert receipt.items[0].raw_description == "Tea"
    assert receipt.items[0].quantity == 2.0
    assert receipt.items[0].unit_price == "3.5"
    assert receipt.items[0].amount == "7.0"


def test_receipt_extractor_preserves_legacy_receipt_return_type():
    payload = json.loads((FIXTURE_DIR / "fred_meyer_receipt.json").read_text())

    receipt = receipt_from_document_intelligence_result(payload)

    assert receipt.total == "17.46"
    assert receipt.items[0].description == "NBSC SALTINE"


def test_rejects_receipt_result_without_items():
    with pytest.raises(ValueError, match="line items"):
        receipt_from_document_intelligence_result(
            {
                "documents": [
                    {
                        "fields": {
                            "MerchantName": {"valueString": "Empty Store"},
                            "TransactionDate": {"valueDate": "2026-04-27"},
                            "Total": {
                                "type": "currency",
                                "valueCurrency": {"amount": 0, "currencyCode": "USD"},
                            },
                            "Items": {
                                "type": "array",
                                "valueArray": [],
                            },
                        }
                    }
                ]
            }
        )
