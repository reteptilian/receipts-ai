from __future__ import annotations

import base64
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from receipts_ai.models.transaction import Source
from receipts_ai.openai_receipt_extraction import (
    DEFAULT_OPENAI_MODEL,
    OPENAI_API_KEY_ENV_VAR,
    ResponsesAPIReceiptClient,
    openai_receipt_request_payload,
    transaction_from_openai_receipt,
)


def test_openai_receipt_payload_sends_images_as_data_urls(tmp_path: Path):
    receipt_path = tmp_path / "receipt.png"
    receipt_bytes = b"image bytes"
    receipt_path.write_bytes(receipt_bytes)

    payload = openai_receipt_request_payload(receipt_path, model=DEFAULT_OPENAI_MODEL)

    user_content = payload["input"][1]["content"]
    image_part = user_content[0]
    assert image_part == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{base64.b64encode(receipt_bytes).decode('ascii')}",
        "detail": "high",
    }
    assert payload["text"] == {"format": {"type": "json_object"}}
    assert "Transaction" in payload["input"][0]["content"][0]["text"]


def test_openai_receipt_payload_sends_pdfs_as_input_files(tmp_path: Path):
    receipt_path = tmp_path / "receipt.pdf"
    receipt_bytes = b"%PDF"
    receipt_path.write_bytes(receipt_bytes)

    payload = openai_receipt_request_payload(receipt_path, model=DEFAULT_OPENAI_MODEL)

    file_part = payload["input"][1]["content"][0]
    assert file_part == {
        "type": "input_file",
        "filename": "receipt.pdf",
        "file_data": f"data:application/pdf;base64,{base64.b64encode(receipt_bytes).decode('ascii')}",
    }


def test_openai_receipt_prompt_explains_costco_instant_savings(tmp_path: Path):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image bytes")

    payload = openai_receipt_request_payload(receipt_path, model=DEFAULT_OPENAI_MODEL)

    system_prompt = payload["input"][0]["content"][0]["text"]
    assert "instant savings" in system_prompt
    assert "discountAmount" in system_prompt
    assert "0000376418 /1721554 3.50-" in system_prompt
    assert "netAmount" in system_prompt


def test_transaction_from_openai_receipt_validates_model_response(tmp_path: Path):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image bytes")

    class FakeClient:
        def extract_transaction(self, receipt_path: Path, *, model: str) -> dict[str, Any]:
            assert receipt_path.name == "receipt.png"
            assert model == "gpt-test"
            return {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": """
                                {
                                  "id": "receipt_1",
                                  "source": "receipt",
                                  "transactionDate": "2026-04-27",
                                  "payee": "Coffee Shop",
                                  "amount": "-7.00",
                                  "currency": "USD",
                                  "receipt": {
                                    "total": "7.00",
                                    "items": [
                                      {"description": "Coffee", "amount": "7.00", "netAmount": "7.00"}
                                    ]
                                  }
                                }
                                """,
                            }
                        ]
                    }
                ]
            }

    transaction = transaction_from_openai_receipt(
        receipt_path, model="gpt-test", client=FakeClient()
    )

    assert transaction.id == "receipt_1"
    assert transaction.source == Source.receipt
    assert transaction.transaction_date == date(2026, 4, 27)
    assert transaction.receipt is not None
    assert transaction.receipt.source_document_id == str(receipt_path)
    assert transaction.receipt.extraction is not None
    assert transaction.receipt.extraction.model == "gpt-test"


def test_transaction_from_openai_receipt_normalizes_null_strings(tmp_path: Path):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image bytes")

    class FakeClient:
        def extract_transaction(self, receipt_path: Path, *, model: str) -> dict[str, Any]:
            assert receipt_path.name == "receipt.png"
            assert model == "gpt-test"
            return {
                "output_text": """
                {
                  "id": "receipt_1",
                  "source": "receipt",
                  "transactionDate": "2026-04-27",
                  "payee": "Costco",
                  "amount": "-8.99",
                  "currency": "USD",
                  "receipt": {
                    "total": "8.99",
                    "items": [
                      {
                        "description": "JP Rainbow",
                        "rawDescription": "1721554 JP RAINBOW",
                        "amount": "12.49",
                        "netAmount": "12.49",
                        "discountAmount": "null"
                      }
                    ]
                  }
                }
                """
            }

    transaction = transaction_from_openai_receipt(
        receipt_path, model="gpt-test", client=FakeClient()
    )

    assert transaction.receipt is not None
    assert transaction.receipt.items[0].discount_amount is None


def test_transaction_from_openai_receipt_logs_invalid_payload(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
):
    receipt_path = tmp_path / "receipt.png"
    receipt_path.write_bytes(b"image bytes")

    class FakeClient:
        def extract_transaction(self, receipt_path: Path, *, model: str) -> dict[str, Any]:
            assert receipt_path.name == "receipt.png"
            assert model == "gpt-test"
            return {
                "output_text": """
                {
                  "id": "receipt_1",
                  "source": "receipt",
                  "transactionDate": "2026-04-27",
                  "payee": "Costco",
                  "amount": "-8.99",
                  "currency": "US dollars",
                  "receipt": {
                    "total": "8.99",
                    "items": [{"description": "JP Rainbow", "amount": "12.49"}]
                  }
                }
                """
            }

    with caplog.at_level(logging.ERROR, logger="receipts_ai.openai_receipt_extraction"):
        with pytest.raises(ValueError):
            transaction_from_openai_receipt(receipt_path, model="gpt-test", client=FakeClient())

    assert "OpenAI transaction payload failed validation" in caplog.text
    assert '"currency": "US dollars"' in caplog.text
    assert '"payee": "Costco"' in caplog.text


def test_responses_client_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(OPENAI_API_KEY_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=OPENAI_API_KEY_ENV_VAR):
        ResponsesAPIReceiptClient()
