from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date
from email.message import Message
from io import BytesIO

import pytest

from receipts_ai.categorization import (
    DEFAULT_OLLAMA_URL,
    UrlLibOllamaClient,
    categorize_receipt_items,
    create_ollama_category_client,
    load_budget_categories,
)
from receipts_ai.models.transaction import Receipt, ReceiptItem, Source, Transaction


class FakeCategoryClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_load_budget_categories_exposes_nested_categories():
    categories = load_budget_categories()

    assert "Food & Dining" in categories
    food_categories = categories["Food & Dining"]
    assert isinstance(food_categories, dict)
    assert "Groceries" in food_categories


def test_categorize_receipt_items_uses_search_results_and_leaf_category_only():
    item = ReceiptItem(
        description="Confusing receipt text",
        raw_description="RAW CONFUSING CODE",
        amount="4.49",
        brave_search_result=json.dumps(
            [
                {
                    "title": "Premium Saltines",
                    "description": "Crisp crackers sold in grocery stores.",
                }
            ]
        ),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    client = FakeCategoryClient(["Food & Dining", "Groceries"])

    result = categorize_receipt_items(transaction, client=client)

    assert result is transaction
    assert item.category_id == "Groceries"
    assert len(client.prompts) == 2
    assert "Food & Dining" in client.prompts[0]
    assert "Groceries" not in client.prompts[0]
    assert "Groceries" in client.prompts[1]
    assert "Food & Dining" not in client.prompts[1].split("Options:", maxsplit=1)[1]
    assert "Premium Saltines" in client.prompts[0]
    assert "Crisp crackers sold in grocery stores." in client.prompts[1]
    assert "RAW CONFUSING CODE" not in "\n".join(client.prompts)
    assert "Confusing receipt text" not in "\n".join(client.prompts)


def test_categorize_receipt_items_rejects_non_leaf_category_response():
    item = ReceiptItem(
        description="Coffee",
        amount="7.00",
        brave_search_result=json.dumps([{"title": "Coffee drink", "description": "Cafe item"}]),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    client = FakeCategoryClient(["Food & Dining", "Food & Dining"])

    with pytest.raises(RuntimeError, match="invalid category"):
        categorize_receipt_items(transaction, client=client)


def test_create_ollama_category_client_uses_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA_URL", "http://example.test:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")

    client = create_ollama_category_client()

    assert isinstance(client, UrlLibOllamaClient)
    assert client.url == "http://example.test:11434"
    assert client.generate_url == "http://example.test:11434/api/generate"
    assert client.model == "llama3.2"


def test_create_ollama_category_client_defaults_url_and_requires_model(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL_NAME", raising=False)

    with pytest.raises(RuntimeError, match="OLLAMA_MODEL"):
        create_ollama_category_client()

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    client = create_ollama_category_client()
    assert isinstance(client, UrlLibOllamaClient)
    assert client.url == DEFAULT_OLLAMA_URL
    assert client.generate_url == f"{DEFAULT_OLLAMA_URL}/api/generate"


def test_url_lib_ollama_client_posts_generate_request(monkeypatch: pytest.MonkeyPatch):
    requests: list[tuple[urllib.request.Request, float]] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse({"response": "Groceries\n"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(
        url="http://example.test:11434/", model="llama3.2", timeout_seconds=3.0
    )

    result = client.complete("Choose one")

    assert result == "Groceries\n"
    request, timeout = requests[0]
    assert timeout == 3.0
    assert request.full_url == "http://example.test:11434/api/generate"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    request_data = request.data
    assert isinstance(request_data, bytes)
    assert json.loads(request_data) == {
        "model": "llama3.2",
        "prompt": "Choose one",
        "stream": False,
        "options": {"temperature": 0},
    }


def test_url_lib_ollama_client_accepts_generate_endpoint_url(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == 30.0
        requests.append(request)
        return FakeResponse({"response": "Groceries"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434/api/generate", model="llama3.2")

    client.complete("Choose one")

    assert client.generate_url == "http://example.test:11434/api/generate"
    assert requests[0].full_url == "http://example.test:11434/api/generate"


def test_url_lib_ollama_client_logs_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert request.full_url == "http://example.test:11434/api/generate"
        assert timeout == 30.0
        return FakeResponse({"response": "Groceries"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    with caplog.at_level("INFO", logger="receipts_ai.categorization"):
        client.complete("Choose one")

    assert (
        "Sending Ollama generate request: url=http://example.test:11434/api/generate model=llama3.2"
    ) in caplog.text
    assert (
        "Received Ollama generate response: url=http://example.test:11434/api/generate "
        "model=llama3.2"
    ) in caplog.text


def test_url_lib_ollama_client_error_includes_endpoint(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == 30.0
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            Message(),
            BytesIO(b"404 page not found"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    with pytest.raises(
        RuntimeError,
        match="Ollama request to http://example.test:11434/api/generate failed with HTTP 404",
    ):
        client.complete("Choose one")
