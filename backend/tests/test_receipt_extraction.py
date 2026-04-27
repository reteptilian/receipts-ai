from __future__ import annotations

import json
from pathlib import Path

import pytest

from receipts_ai.receipt_extraction import receipt_from_document_intelligence_result

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "azure"


def test_extracts_receipt_items_from_azure_receipt_result():
    payload = json.loads((FIXTURE_DIR / "fred_meyer_receipt.json").read_text())

    receipt = receipt_from_document_intelligence_result(payload)

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


def test_rejects_receipt_result_without_items():
    with pytest.raises(ValueError, match="line items"):
        receipt_from_document_intelligence_result(
            {
                "documents": [
                    {
                        "fields": {
                            "Items": {
                                "type": "array",
                                "valueArray": [],
                            }
                        }
                    }
                ]
            }
        )
