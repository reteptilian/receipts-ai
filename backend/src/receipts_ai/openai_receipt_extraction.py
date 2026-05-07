from __future__ import annotations

import base64
import json
import logging
import mimetypes
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import ValidationError

from receipts_ai.cache import SqliteCallCache
from receipts_ai.config import config_value
from receipts_ai.models.transaction import ExtractionMetadata, Source, Transaction

OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_MODEL_ENV_VAR = "OPENAI_MODEL"
DEFAULT_OPENAI_MODEL = "gpt-4o"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_CACHE_NAMESPACE = "openai_receipt_extraction"
LOGGER = logging.getLogger(__name__)
NULL_STRING_FIELDS = {
    "accountId",
    "braveSearchResult",
    "categoryId",
    "createdAt",
    "description",
    "discountAmount",
    "discountDescription",
    "externalId",
    "extractedAt",
    "extraction",
    "id",
    "kind",
    "lineType",
    "netAmount",
    "notes",
    "postedDate",
    "rawDescription",
    "rawText",
    "receipt",
    "receiptNumber",
    "sourceDocumentId",
    "subtotal",
    "taxonomy1",
    "taxonomy2",
    "taxonomy3",
    "taxonomy4",
    "taxonomy5",
    "taxonomy6",
    "taxonomy7",
    "taxonomy8",
    "taxonomy9",
    "unitPrice",
    "updatedAt",
}


class OpenAIReceiptClient(Protocol):
    def extract_transaction(self, receipt_path: Path, *, model: str) -> dict[str, Any]: ...


class ResponsesAPIReceiptClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        responses_url: str = OPENAI_RESPONSES_URL,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else _openai_api_key()
        self.responses_url = responses_url
        self.timeout_seconds = timeout_seconds

    def extract_transaction(self, receipt_path: Path, *, model: str) -> dict[str, Any]:
        request_payload = openai_receipt_request_payload(receipt_path, model=model)
        body = json.dumps(request_payload).encode("utf-8")
        LOGGER.info(
            "Calling OpenAI receipt extraction model=%s file=%s size_bytes=%d",
            model,
            receipt_path,
            receipt_path.stat().st_size,
        )
        request = urllib.request.Request(
            self.responses_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI receipt extraction failed: {error.code} {error_body}"
            ) from error

        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI receipt extraction response was not a JSON object")
        response_payload = cast(dict[str, Any], payload)
        LOGGER.info(
            "OpenAI receipt extraction completed response_id=%s status=%s",
            response_payload.get("id"),
            response_payload.get("status"),
        )
        LOGGER.debug("OpenAI receipt extraction response keys: %s", sorted(response_payload))
        return response_payload


class CachedOpenAIReceiptClient:
    def __init__(
        self,
        *,
        cache: SqliteCallCache,
        client_factory: type[ResponsesAPIReceiptClient] | None = None,
    ) -> None:
        self.cache = cache
        self.client_factory = (
            client_factory if client_factory is not None else ResponsesAPIReceiptClient
        )

    def extract_transaction(self, receipt_path: Path, *, model: str) -> dict[str, Any]:
        request = openai_receipt_cache_request(receipt_path, model=model)
        cached = self.cache.get(OPENAI_CACHE_NAMESPACE, request)
        if cached is not None:
            LOGGER.info(
                "Using cached OpenAI receipt extraction response model=%s file=%s",
                model,
                receipt_path,
            )
            if not isinstance(cached, dict):
                raise RuntimeError(
                    "cached OpenAI receipt extraction response was not a JSON object"
                )
            return cast(dict[str, Any], cached)

        LOGGER.info(
            "OpenAI receipt extraction cache miss model=%s file=%s",
            model,
            receipt_path,
        )
        response = self.client_factory().extract_transaction(receipt_path, model=model)
        self.cache.set(OPENAI_CACHE_NAMESPACE, request, response)
        return response


def transaction_from_openai_receipt(
    receipt_path: Path,
    *,
    model: str | None = None,
    client: OpenAIReceiptClient | None = None,
    cache: SqliteCallCache | None = None,
) -> Transaction:
    selected_model = (
        model
        if model is not None
        else config_value(OPENAI_MODEL_ENV_VAR, DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL
    )
    selected_client: OpenAIReceiptClient
    if client is not None:
        selected_client = client
    elif cache is not None:
        selected_client = CachedOpenAIReceiptClient(cache=cache)
    else:
        selected_client = ResponsesAPIReceiptClient()

    LOGGER.info(
        "Starting OpenAI receipt transaction extraction model=%s file=%s",
        selected_model,
        receipt_path,
    )
    response = selected_client.extract_transaction(receipt_path, model=selected_model)
    output_text = _response_output_text(response)
    transaction_payload = _transaction_payload_from_output_text(output_text)
    transaction_payload = _normalize_openai_transaction_payload(transaction_payload)
    LOGGER.debug("Validating OpenAI transaction payload keys: %s", sorted(transaction_payload))
    try:
        transaction = Transaction.model_validate(transaction_payload)
    except ValidationError:
        LOGGER.exception(
            "OpenAI transaction payload failed validation. output_text=%s payload=%s",
            output_text,
            json.dumps(transaction_payload, indent=2, sort_keys=True, default=str),
        )
        raise
    if transaction.source != Source.receipt:
        LOGGER.info("Normalizing OpenAI transaction source from %s to receipt", transaction.source)
        transaction.source = Source.receipt
    if transaction.receipt is not None:
        transaction.receipt.source_document_id = transaction.receipt.source_document_id or str(
            receipt_path
        )
        if transaction.receipt.extraction is None:
            transaction.receipt.extraction = ExtractionMetadata(model=selected_model)
        elif transaction.receipt.extraction.model is None:
            transaction.receipt.extraction.model = selected_model
        LOGGER.info(
            "OpenAI receipt transaction validated id=%s payee=%s date=%s total=%s item_count=%d",
            transaction.id,
            transaction.payee,
            transaction.transaction_date,
            transaction.receipt.total,
            len(transaction.receipt.items),
        )
    else:
        LOGGER.info(
            "OpenAI transaction validated without receipt id=%s payee=%s date=%s",
            transaction.id,
            transaction.payee,
            transaction.transaction_date,
        )
    return transaction


def openai_receipt_request_payload(receipt_path: Path, *, model: str) -> dict[str, Any]:
    schema = Transaction.model_json_schema(by_alias=True)
    LOGGER.debug("Building OpenAI receipt request model=%s file=%s", model, receipt_path)
    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": _system_prompt(schema),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    _receipt_input_part(receipt_path),
                    {
                        "type": "input_text",
                        "text": (
                            "Extract this receipt into one Transaction JSON object. "
                            "Do not include Markdown, comments, or surrounding text."
                        ),
                    },
                ],
            },
        ],
        "text": {"format": {"type": "json_object"}},
    }


def openai_receipt_cache_request(receipt_path: Path, *, model: str) -> dict[str, object]:
    return {
        "model": model,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _file_sha256(receipt_path),
        "schema": "Transaction",
    }


def _system_prompt(schema: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "Role: You are a specialized financial data extraction agent. "
            "Task: Analyze the attached receipt image and convert it into a structured JSON format. "
            "Step 1: Visual Extraction – Extract all line items, quantities, and prices exactly as they appear. "
            "Step 2: Semantic Decoding – For each item, look at the cryptic abbreviation and use your internal"
            'knowledge to determine the full, human-readable product name (e.g., "Hlf Gal 2% Mlk" -> "Half Gallon 2% Milk").'
            "Step 3: Return exactly one JSON object that validates against the Transaction schema below. ",
            "Use the schema field aliases, such as transactionDate, categoryId, and rawDescription. ",
            "Set source to receipt. Set kind to expense unless the receipt is clearly income, transfer, or adjustment. ",
            "Use ISO dates. Use ISO 4217 currency codes, defaulting to USD when the receipt does not state one. ",
            "Encode all money amounts as decimal strings. The transaction amount is from the account perspective, so purchases are negative and receipt item amounts are normally positive. ",
            "Always set each receipt item's netAmount. When no item-level discount applies, netAmount must equal amount. ",
            "Include sales tax as its own item. ",
            "Do not include items for summary receipts totals such as receipt total or total savings or total discounts. "
            "Costco and similar receipts may show instant savings or coupon discounts as a separate raw line immediately after the item they apply to. These lines often look like a long numeric code, a slash code, and a trailing negative amount, for example '0000376418 /1721554 3.50-'. When an instant-savings line immediately follows an item, collapse it into the previous receipt item instead of returning it as a separate item: keep the previous item amount as the pre-discount amount, set discountAmount to the signed negative discount, set discountDescription to the raw discount line text, and set netAmount to item amount plus discountAmount. If multiple savings lines apply to the same item, combine them into one signed discountAmount and join their raw texts in discountDescription. ",
            "For example, if the receipt shows '1721554 JP RAINBOW 12.49' followed by '0000376418 /1721554 3.50-', return one item with rawDescription '1721554 JP RAINBOW', amount '12.49', discountAmount '-3.50', discountDescription '0000376418 /1721554 3.50-', and netAmount '8.99'.",
            "Use null rather than guessing when optional values are not visible. Do not invent loyalty IDs, card numbers, or addresses. ",
            "Schema JSON:",
            json.dumps(schema, separators=(",", ":")),
        )
    )


def _receipt_input_part(receipt_path: Path) -> dict[str, object]:
    mime_type = mimetypes.guess_type(receipt_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(receipt_path.read_bytes()).decode("ascii")
    if mime_type.startswith("image/"):
        return {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{encoded}",
            "detail": "high",
        }
    return {
        "type": "input_file",
        "filename": receipt_path.name,
        "file_data": f"data:{mime_type};base64,{encoded}",
    }


def _transaction_payload_from_output_text(output_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"OpenAI receipt extraction did not return valid JSON: {output_text}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError("OpenAI receipt extraction JSON was not an object")
    return cast(dict[str, Any], payload)


def _normalize_openai_transaction_payload(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        fields = cast(dict[str, Any], value)
        return {
            field_key: _normalize_openai_transaction_payload(field_value, key=field_key)
            for field_key, field_value in fields.items()
        }
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [_normalize_openai_transaction_payload(item, key=key) for item in items]
    if isinstance(value, str) and value.strip().lower() in {"null", "none", "n/a"}:
        if key in NULL_STRING_FIELDS:
            LOGGER.debug("Normalizing string %r to JSON null for field %s", value, key)
            return None
    return value


def _response_output_text(response: Mapping[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for output_item in _list_value(response.get("output")):
        for content_item in _list_value(_dict_value(output_item).get("content")):
            content = _dict_value(content_item)
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif content.get("type") == "refusal":
                refusal = content.get("refusal")
                if isinstance(refusal, str):
                    raise RuntimeError(f"OpenAI refused receipt extraction: {refusal}")

    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError("OpenAI receipt extraction response did not contain output text")
    return text


def _openai_api_key() -> str:
    api_key = config_value(OPENAI_API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(f"Set {OPENAI_API_KEY_ENV_VAR} to call OpenAI receipt extraction.")
    return api_key


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return cast(list[Any], value)
    return []
