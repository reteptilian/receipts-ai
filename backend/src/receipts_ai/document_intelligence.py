from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, cast

DEFAULT_RECEIPT_MODEL_ID = "prebuilt-receipt"
ENDPOINT_ENV_VARS = (
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    "DOCUMENTINTELLIGENCE_ENDPOINT",
)
KEY_ENV_VARS = (
    "AZURE_DOCUMENT_INTELLIGENCE_KEY",
    "DOCUMENTINTELLIGENCE_API_KEY",
)


class AnalyzeDocumentClient(Protocol):
    def begin_analyze_document(self, model_id: str, body: object) -> Any: ...


def analyze_receipt_file(path: str | Path, *, client: AnalyzeDocumentClient | None = None) -> Any:
    return analyze_receipt_bytes(Path(path).read_bytes(), client=client)


def analyze_receipt_bytes(document: bytes, *, client: AnalyzeDocumentClient | None = None) -> Any:
    if not document:
        raise ValueError("document must not be empty")

    active_client = client if client is not None else create_document_intelligence_client()
    request = _new_analyze_document_request(document)
    poller = active_client.begin_analyze_document(DEFAULT_RECEIPT_MODEL_ID, request)
    return poller.result()


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
    for env_var in ENDPOINT_ENV_VARS:
        endpoint = os.getenv(env_var)
        if endpoint:
            return endpoint

    env_var_list = ", ".join(ENDPOINT_ENV_VARS)
    raise RuntimeError(f"Set one of these environment variables: {env_var_list}")


def _document_intelligence_key() -> str:
    for env_var in KEY_ENV_VARS:
        key = os.getenv(env_var)
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
