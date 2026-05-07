from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol, cast

from receipts_ai.cache import SqliteCallCache
from receipts_ai.config import first_config_value

DEFAULT_RECEIPT_MODEL_ID = "prebuilt-receipt"
ENDPOINT_ENV_VARS = (
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    "DOCUMENTINTELLIGENCE_ENDPOINT",
)
KEY_ENV_VARS = (
    "AZURE_DOCUMENT_INTELLIGENCE_KEY",
    "DOCUMENTINTELLIGENCE_API_KEY",
)

logger = logging.getLogger(__name__)


class AnalyzeDocumentClient(Protocol):
    def begin_analyze_document(self, model_id: str, body: object) -> Any: ...


def analyze_receipt_file(
    path: str | Path,
    *,
    client: AnalyzeDocumentClient | None = None,
    cache: SqliteCallCache | None = None,
) -> Any:
    return analyze_receipt_bytes(Path(path).read_bytes(), client=client, cache=cache)


def analyze_receipt_bytes(
    document: bytes,
    *,
    client: AnalyzeDocumentClient | None = None,
    cache: SqliteCallCache | None = None,
) -> Any:
    if not document:
        raise ValueError("document must not be empty")

    cache_request = {
        "document_sha256": hashlib.sha256(document).hexdigest(),
        "model_id": DEFAULT_RECEIPT_MODEL_ID,
    }
    if cache is not None:
        cached_response = cache.get("azure_document_intelligence", cache_request)
        if cached_response is not None:
            logger.info(
                "Using cached Azure Document Intelligence response: document_sha256=%s",
                cache_request["document_sha256"],
            )
            return cached_response

    active_client = client if client is not None else create_document_intelligence_client()
    request = _new_analyze_document_request(document)
    poller = active_client.begin_analyze_document(DEFAULT_RECEIPT_MODEL_ID, request)
    result = poller.result()
    if cache is not None:
        cache.set("azure_document_intelligence", cache_request, to_jsonable(result))
        logger.info(
            "Cached Azure Document Intelligence response: document_sha256=%s",
            cache_request["document_sha256"],
        )
    return result


def create_document_intelligence_client() -> AnalyzeDocumentClient:
    endpoint = _document_intelligence_endpoint()
    key = _document_intelligence_key()

    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError as exc:
        raise RuntimeError(
            "Azure Document Intelligence dependencies are not installed. "
            "Run `uv sync` and try again."
        ) from exc

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )
    return cast(AnalyzeDocumentClient, cast(object, client))


def _document_intelligence_endpoint() -> str:
    endpoint = first_config_value(ENDPOINT_ENV_VARS)
    if endpoint:
        return endpoint

    env_var_list = ", ".join(ENDPOINT_ENV_VARS)
    raise RuntimeError(f"Set one of these environment variables: {env_var_list}")


def _document_intelligence_key() -> str:
    key = first_config_value(KEY_ENV_VARS)
    if key:
        return key

    env_var_list = ", ".join(KEY_ENV_VARS)
    raise RuntimeError(f"Set one of these environment variables: {env_var_list}")


def _new_analyze_document_request(document: bytes) -> object:
    try:
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    except ImportError as exc:
        raise RuntimeError(
            "Azure Document Intelligence dependencies are not installed. "
            "Run `uv sync` and try again."
        ) from exc

    return AnalyzeDocumentRequest(bytes_source=document)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        dict_value = cast(dict[Any, Any], value)
        return {key: to_jsonable(child) for key, child in dict_value.items()}
    if isinstance(value, list | tuple):
        sequence_value = cast(list[Any] | tuple[Any, ...], value)
        return [to_jsonable(child) for child in sequence_value]
    return value


def pretty_print_analysis(value: Any) -> None:
    print(json.dumps(to_jsonable(value), indent=2, sort_keys=True, default=str))
