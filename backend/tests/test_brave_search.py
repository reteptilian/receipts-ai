from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import date
from email.message import Message
from io import BytesIO
from pathlib import Path

import pytest

from receipts_ai.brave_search import (
    DEFAULT_BRAVE_SEARCH_ENDPOINT,
    CachedBraveSearchClient,
    UrlLibBraveSearchClient,
    create_brave_search_client,
    enrich_receipt_items_with_brave_search,
    enrich_transactions_with_brave_search,
)
from receipts_ai.cache import JsonCallCache
from receipts_ai.models.transaction import Receipt, Source, Transaction
from receipts_ai.models.transaction import ReceiptItem as GeneratedReceiptItem


def ReceiptItem(**kwargs):  # noqa: N802
    if "amount" in kwargs and "net_amount" not in kwargs:
        kwargs["net_amount"] = kwargs["amount"]
    return GeneratedReceiptItem(**kwargs)


class FakeBraveClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> dict[str, object]:
        self.queries.append(query)
        return {
            "query": {"original": query},
            "web": {
                "results": [
                    {
                        "description": "Crisp crackers with sea salt.",
                        "thumbnail": {"src": "https://example.test/image.jpg"},
                        "title": "Premium Saltines",
                        "url": "https://example.test/saltines",
                    }
                ]
            },
        }


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


def test_create_brave_search_client_reads_home_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_ENDPOINT", raising=False)
    (tmp_path / ".receipts_ai.config").write_text(
        "\n".join(
            (
                "BRAVE_SEARCH_API_KEY=config-key",
                "BRAVE_SEARCH_ENDPOINT=https://example.test/config-search",
            )
        ),
        encoding="utf-8",
    )

    client = create_brave_search_client()

    assert isinstance(client, UrlLibBraveSearchClient)
    assert client.key == "config-key"
    assert client.endpoint == "https://example.test/config-search"


def test_create_brave_search_client_requires_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="BRAVE_SEARCH_API_KEY"):
        create_brave_search_client()


def test_request_delay_seconds_can_come_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BRAVE_SEARCH_REQUEST_DELAY_SECONDS", "1.25")
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-10.00",
        currency="USD",
        receipt=Receipt(
            items=[
                ReceiptItem(description="Coffee", raw_description="COF", amount="7.00"),
                ReceiptItem(description="Bagel", raw_description="BGL", amount="3.00"),
            ]
        ),
    )
    sleeps: list[float] = []

    enrich_receipt_items_with_brave_search(
        transaction,
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


def test_cached_brave_search_client_reuses_json_file(tmp_path: Path):
    cache_path = tmp_path / "api-cache.json"
    first_client = FakeBraveClient()
    cached_client = CachedBraveSearchClient(client=first_client, cache=JsonCallCache(cache_path))

    first_result = cached_client.search("FredMeyer NBSC SALTINE $2.25")

    second_client = FakeBraveClient()
    second_cached_client = CachedBraveSearchClient(
        client=second_client, cache=JsonCallCache(cache_path)
    )
    second_result = second_cached_client.search("FredMeyer NBSC SALTINE $2.25")

    assert first_result == second_result
    assert first_client.queries == ["FredMeyer NBSC SALTINE $2.25"]
    assert second_client.queries == []
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["ollama"] == []
    assert payload["brave_search"][0]["request"] == {"query": "FredMeyer NBSC SALTINE $2.25"}


def test_cached_brave_search_client_does_not_create_client_on_cache_hit(tmp_path: Path):
    cache_path = tmp_path / "api-cache.json"
    cache = JsonCallCache(cache_path)
    cache.set("brave_search", {"query": "Coffee Shop COF $7.00"}, {"cached": True})
    factory_calls = 0

    def client_factory() -> FakeBraveClient:
        nonlocal factory_calls
        factory_calls += 1
        return FakeBraveClient()

    client = CachedBraveSearchClient(cache=JsonCallCache(cache_path), client_factory=client_factory)

    result = client.search("Coffee Shop COF $7.00")

    assert result == {"cached": True}
    assert factory_calls == 0


def test_enrich_receipt_items_with_brave_search_uses_raw_description_first():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-5.78",
        currency="USD",
        receipt=Receipt(
            items=[
                ReceiptItem(
                    description="Saltine Crackers",
                    raw_description="NBSC SALTINE",
                    unit_price="2.25",
                    amount="4.49",
                ),
                ReceiptItem(description="Parsley", amount="1.29"),
            ]
        ),
    )
    client = FakeBraveClient()

    result = enrich_receipt_items_with_brave_search(transaction, client=client)

    assert result is transaction
    assert client.queries == ["FredMeyer NBSC SALTINE $2.25", "FredMeyer Parsley $1.29"]
    assert transaction.receipt is not None
    assert transaction.receipt.items[0].brave_search_result is not None
    assert json.loads(transaction.receipt.items[0].brave_search_result) == [
        {
            "description": "Crisp crackers with sea salt.",
            "title": "Premium Saltines",
        }
    ]


def test_enrich_receipt_items_with_brave_search_stores_only_titles_and_descriptions():
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=Receipt(
            items=[
                ReceiptItem(
                    description="Saltine Crackers",
                    raw_description="NBSC SALTINE",
                    amount="4.49",
                ),
            ]
        ),
    )

    enrich_receipt_items_with_brave_search(transaction, client=FakeBraveClient())

    assert transaction.receipt is not None
    assert transaction.receipt.items[0].brave_search_result is not None
    assert json.loads(transaction.receipt.items[0].brave_search_result) == [
        {
            "description": "Crisp crackers with sea salt.",
            "title": "Premium Saltines",
        }
    ]


def test_enrich_receipt_items_with_brave_search_throttles_between_requests(
    caplog: pytest.LogCaptureFixture,
):
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-14.00",
        currency="USD",
        receipt=Receipt(
            items=[
                ReceiptItem(description="Coffee", raw_description="COF", amount="7.00"),
                ReceiptItem(description="Bagel", raw_description="BGL", amount="3.00"),
                ReceiptItem(description="Tea", raw_description="TEA", amount="4.00"),
            ]
        ),
    )
    sleeps: list[float] = []

    with caplog.at_level(logging.INFO, logger="receipts_ai.brave_search"):
        enrich_receipt_items_with_brave_search(
            transaction,
            client=FakeBraveClient(),
            request_delay_seconds=1.1,
            sleep=sleeps.append,
        )

    assert sleeps == [1.1, 1.1]
    assert "Enriching receipt item 1 with Brave Search: Coffee Shop COF $7.00" in caplog.text
    assert "Sleeping 1.10 seconds before the next Brave Search query" in caplog.text


def test_enrich_receipt_items_with_brave_search_skips_delay_for_cached_queries(
    tmp_path: Path,
):
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-14.00",
        currency="USD",
        receipt=Receipt(
            items=[
                ReceiptItem(description="Coffee", raw_description="COF", amount="7.00"),
                ReceiptItem(description="Bagel", raw_description="BGL", amount="3.00"),
                ReceiptItem(description="Tea", raw_description="TEA", amount="4.00"),
            ]
        ),
    )
    cache = JsonCallCache(tmp_path / "api-cache.json")
    cache.set(
        "brave_search",
        {"query": "Coffee Shop BGL $3.00"},
        {
            "web": {
                "results": [
                    {
                        "description": "Cached bagel result.",
                        "title": "Bagel",
                    }
                ]
            }
        },
    )
    brave_client = FakeBraveClient()
    wrapped_client = CachedBraveSearchClient(client=brave_client, cache=cache)
    sleeps: list[float] = []

    enrich_receipt_items_with_brave_search(
        transaction,
        client=wrapped_client,
        request_delay_seconds=1.1,
        sleep=sleeps.append,
    )

    assert sleeps == [1.1]
    assert brave_client.queries == ["Coffee Shop COF $7.00", "Coffee Shop TEA $4.00"]


def test_enrich_transactions_with_brave_search_uses_description_without_amount():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="COSTCO WHSE",
        description="POS PURCHASE COSTCO WHSE #123",
        amount="-42.19",
        currency="USD",
    )
    client = FakeBraveClient()

    result = enrich_transactions_with_brave_search([transaction], client=client)

    assert result == [transaction]
    assert client.queries == ["COSTCO WHSE POS PURCHASE COSTCO WHSE #123"]
    assert transaction.brave_search_result is not None
    assert json.loads(transaction.brave_search_result) == [
        {
            "description": "Crisp crackers with sea salt.",
            "title": "Premium Saltines",
        }
    ]
