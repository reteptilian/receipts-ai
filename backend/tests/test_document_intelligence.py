from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from receipts_ai import document_intelligence
from receipts_ai.cache import SqliteCallCache
from receipts_ai.document_intelligence import (
    DEFAULT_RECEIPT_MODEL_ID,
    analyze_receipt_bytes,
    to_jsonable,
)


def _request_factory(document: bytes) -> object:
    return SimpleNamespace(bytes_source=document)


class FakePoller:
    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def begin_analyze_document(self, model_id: str, body: object) -> FakePoller:
        self.calls.append((model_id, body))
        return FakePoller({"status": "ok"})


class FakeAzureKeyCredential:
    def __init__(self, key: str) -> None:
        self.key = key


class FakeDocumentIntelligenceClient:
    created_with: dict[str, object] | None = None

    def __init__(self, *, endpoint: str, credential: object) -> None:
        type(self).created_with = {"endpoint": endpoint, "credential": credential}


def test_analyze_receipt_bytes_uses_prebuilt_receipt_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        document_intelligence,
        "_new_analyze_document_request",
        _request_factory,
    )
    client = FakeClient()

    result = analyze_receipt_bytes(b"receipt bytes", client=client)

    assert result == {"status": "ok"}
    assert client.calls == [
        (
            DEFAULT_RECEIPT_MODEL_ID,
            SimpleNamespace(bytes_source=b"receipt bytes"),
        )
    ]


def test_analyze_receipt_bytes_rejects_empty_document():
    with pytest.raises(ValueError, match="document must not be empty"):
        analyze_receipt_bytes(b"", client=FakeClient())


def test_analyze_receipt_bytes_reuses_sqlite_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        document_intelligence,
        "_new_analyze_document_request",
        _request_factory,
    )
    cache_path = tmp_path / "api-cache.sqlite"
    first_client = FakeClient()
    first_result = analyze_receipt_bytes(
        b"receipt bytes",
        client=first_client,
        cache=SqliteCallCache(cache_path),
    )

    second_client = FakeClient()
    second_result = analyze_receipt_bytes(
        b"receipt bytes",
        client=second_client,
        cache=SqliteCallCache(cache_path),
    )

    assert first_result == second_result
    assert len(first_client.calls) == 1
    assert second_client.calls == []
    assert (
        SqliteCallCache(cache_path).get(
            "azure_document_intelligence",
            {
                "document_sha256": hashlib.sha256(b"receipt bytes").hexdigest(),
                "model_id": DEFAULT_RECEIPT_MODEL_ID,
            },
        )
        == first_result
    )


def test_create_document_intelligence_client_uses_key_credential(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_documentintelligence_module = SimpleNamespace(
        DocumentIntelligenceClient=FakeDocumentIntelligenceClient
    )
    fake_credentials_module = SimpleNamespace(AzureKeyCredential=FakeAzureKeyCredential)
    monkeypatch.setitem(
        sys.modules,
        "azure.ai.documentintelligence",
        fake_documentintelligence_module,
    )
    monkeypatch.setitem(sys.modules, "azure.core.credentials", fake_credentials_module)
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://example.test/")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "test-key")

    document_intelligence.create_document_intelligence_client()

    created_with = FakeDocumentIntelligenceClient.created_with
    assert created_with is not None
    assert created_with["endpoint"] == "https://example.test/"
    credential = created_with["credential"]
    assert isinstance(credential, FakeAzureKeyCredential)
    assert credential.key == "test-key"


def test_create_document_intelligence_client_reads_alternate_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_documentintelligence_module = SimpleNamespace(
        DocumentIntelligenceClient=FakeDocumentIntelligenceClient
    )
    fake_credentials_module = SimpleNamespace(AzureKeyCredential=FakeAzureKeyCredential)
    monkeypatch.setitem(
        sys.modules,
        "azure.ai.documentintelligence",
        fake_documentintelligence_module,
    )
    monkeypatch.setitem(sys.modules, "azure.core.credentials", fake_credentials_module)
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://example.test/")
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)
    monkeypatch.setenv("DOCUMENTINTELLIGENCE_API_KEY", "test-key")

    document_intelligence.create_document_intelligence_client()

    created_with = FakeDocumentIntelligenceClient.created_with
    assert created_with is not None
    credential = created_with["credential"]
    assert isinstance(credential, FakeAzureKeyCredential)
    assert credential.key == "test-key"


def test_create_document_intelligence_client_requires_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://example.test/")
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)
    monkeypatch.delenv("DOCUMENTINTELLIGENCE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_DOCUMENT_INTELLIGENCE_KEY"):
        document_intelligence.create_document_intelligence_client()


def test_to_jsonable_prefers_as_dict():
    class AzureLikeResult:
        def as_dict(self) -> dict[str, object]:
            return {"documents": [{"fields": {"MerchantName": {"content": "Cafe"}}}]}

    assert to_jsonable(AzureLikeResult()) == {
        "documents": [{"fields": {"MerchantName": {"content": "Cafe"}}}]
    }
