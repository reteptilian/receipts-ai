from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from email.message import Message
from io import BytesIO

import pytest

from receipts_ai.brave_search import (
    DEFAULT_BRAVE_SEARCH_ENDPOINT,
    UrlLibBraveSearchClient,
    create_brave_search_client,
    enrich_receipt_items_with_brave_search,
)
from receipts_ai.models.transaction import Receipt, ReceiptItem


class FakeBraveClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> dict[str, object]:
        self.queries.append(query)
        return {"query": {"original": query}, "web": {"results": [{"title": "Saltines"}]}}


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_create_brave_search_client_uses_env_key_and_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    monkeypatch.delenv("BRAVE_SEARCH_ENDPOINT", raising=False)

    client = create_brave_search_client()

    assert isinstance(client, UrlLibBraveSearchClient)
    assert client.endpoint == DEFAULT_BRAVE_SEARCH_ENDPOINT
    assert client.key == "test-key"


def test_create_brave_search_client_uses_env_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    monkeypatch.setenv("BRAVE_SEARCH_ENDPOINT", "https://example.test/search")

    client = create_brave_search_client()

    assert isinstance(client, UrlLibBraveSearchClient)
    assert client.endpoint == "https://example.test/search"


def test_create_brave_search_client_reads_alternate_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "alternate-key")

    client = create_brave_search_client()

    assert isinstance(client, UrlLibBraveSearchClient)
    assert client.key == "alternate-key"


def test_create_brave_search_client_requires_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="BRAVE_SEARCH_API_KEY"):
        create_brave_search_client()


def test_request_delay_seconds_can_come_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BRAVE_SEARCH_REQUEST_DELAY_SECONDS", "1.25")
    receipt = Receipt(
        items=[
            ReceiptItem(description="Coffee", raw_description="COF", amount="7.00"),
            ReceiptItem(description="Bagel", raw_description="BGL", amount="3.00"),
        ]
    )
    sleeps: list[float] = []

    enrich_receipt_items_with_brave_search(
        receipt,
        client=FakeBraveClient(),
        sleep=sleeps.append,
    )

    assert sleeps == [1.25]


def test_url_lib_client_sends_query_and_subscription_token(monkeypatch: pytest.MonkeyPatch):
    requests: list[tuple[urllib.request.Request, float]] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse({"web": {"results": [{"title": "Nabisco Premium Saltines"}]}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibBraveSearchClient(
        endpoint="https://example.test/search?country=us", key="test-key", timeout_seconds=3.0
    )

    result = client.search("NBSC SALTINE")

    assert result == {"web": {"results": [{"title": "Nabisco Premium Saltines"}]}}
    request, timeout = requests[0]
    assert timeout == 3.0
    assert request.full_url == "https://example.test/search?country=us&q=NBSC+SALTINE"
    assert request.get_header("X-subscription-token") == "test-key"
    assert request.get_header("Accept") == "application/json"


def test_url_lib_client_gives_rate_limit_hint(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == 10.0
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            Message(),
            BytesIO(b'{"code":"RATE_LIMITED"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibBraveSearchClient(endpoint="https://example.test/search", key="test-key")

    with pytest.raises(RuntimeError, match="--brave-search-delay-seconds"):
        client.search("NBSC SALTINE")


def test_enrich_receipt_items_with_brave_search_uses_raw_description_first():
    receipt = Receipt(
        items=[
            ReceiptItem(
                description="Saltine Crackers", raw_description="NBSC SALTINE", amount="4.49"
            ),
            ReceiptItem(description="Parsley", amount="1.29"),
        ]
    )
    client = FakeBraveClient()

    result = enrich_receipt_items_with_brave_search(receipt, client=client)

    assert result is receipt
    assert client.queries == ["NBSC SALTINE", "Parsley"]
    assert receipt.items[0].brave_search_result is not None
    assert json.loads(receipt.items[0].brave_search_result)["query"]["original"] == "NBSC SALTINE"


def test_enrich_receipt_items_with_brave_search_throttles_between_requests(
    caplog: pytest.LogCaptureFixture,
):
    receipt = Receipt(
        items=[
            ReceiptItem(description="Coffee", raw_description="COF", amount="7.00"),
            ReceiptItem(description="Bagel", raw_description="BGL", amount="3.00"),
            ReceiptItem(description="Tea", raw_description="TEA", amount="4.00"),
        ]
    )
    sleeps: list[float] = []

    with caplog.at_level(logging.INFO, logger="receipts_ai.brave_search"):
        enrich_receipt_items_with_brave_search(
            receipt,
            client=FakeBraveClient(),
            request_delay_seconds=1.1,
            sleep=sleeps.append,
        )

    assert sleeps == [1.1, 1.1]
    assert "Enriching receipt item 1 with Brave Search: COF" in caplog.text
    assert "Sleeping 1.10 seconds before the next Brave Search query" in caplog.text
