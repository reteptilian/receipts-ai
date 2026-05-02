from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Protocol, cast

from receipts_ai.cache import JsonCallCache
from receipts_ai.models.transaction import ReceiptItem, Transaction

DEFAULT_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
ENDPOINT_ENV_VARS = ("BRAVE_SEARCH_ENDPOINT",)
KEY_ENV_VARS = ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY")
REQUEST_DELAY_SECONDS_ENV_VARS = ("BRAVE_SEARCH_REQUEST_DELAY_SECONDS",)

logger = logging.getLogger(__name__)


class BraveSearchClient(Protocol):
    def search(self, query: str) -> Any: ...


class UrlLibBraveSearchClient:
    def __init__(self, *, endpoint: str, key: str, timeout_seconds: float = 10.0) -> None:
        self.endpoint = endpoint
        self.key = key
        self.timeout_seconds = timeout_seconds

    def search(self, query: str) -> Any:
        if not query:
            raise ValueError("query must not be empty")

        url = _url_with_query(self.endpoint, {"q": query})
        logger.info("Sending Brave Search query: %s", query)
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.key,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            excerpt = body[:500]
            if exc.code == 429:
                raise RuntimeError(
                    "Brave Search request failed with HTTP 429 rate limit. "
                    "Try a larger --brave-search-delay-seconds value, such as 1.1. "
                    f"Response: {excerpt}"
                ) from exc
            raise RuntimeError(
                f"Brave Search request failed with HTTP {exc.code}: {excerpt}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Brave Search request failed: {exc.reason}") from exc

        result = json.loads(payload)
        logger.info("Received Brave Search response for query: %s", query)
        return result


class CachedBraveSearchClient:
    def __init__(
        self,
        *,
        cache: JsonCallCache,
        client: BraveSearchClient | None = None,
        client_factory: Callable[[], BraveSearchClient] | None = None,
    ) -> None:
        if client is None and client_factory is None:
            raise ValueError("client or client_factory is required")
        self.cache = cache
        self._client = client
        self._client_factory = client_factory

    def search(self, query: str) -> Any:
        request = {"query": query}
        cached_response = self.cache.get("brave_search", request)
        if cached_response is not None:
            logger.info("Using cached Brave Search response for query: %s", query)
            return cached_response

        result = self._active_client().search(query)
        self.cache.set("brave_search", request, result)
        logger.info("Cached Brave Search response for query: %s", query)
        return result

    def is_cached(self, query: str) -> bool:
        return self.cache.get("brave_search", {"query": query}) is not None

    def _active_client(self) -> BraveSearchClient:
        if self._client is None:
            if self._client_factory is None:
                raise RuntimeError("client_factory is not configured")
            self._client = self._client_factory()
        return self._client


def create_brave_search_client() -> BraveSearchClient:
    return UrlLibBraveSearchClient(endpoint=_brave_search_endpoint(), key=_brave_search_key())


def enrich_receipt_items_with_brave_search(
    transaction: Transaction,
    *,
    client: BraveSearchClient | None = None,
    request_delay_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Transaction:
    if transaction.receipt is None:
        raise ValueError("transaction does not contain a receipt")

    active_client = client if client is not None else create_brave_search_client()
    active_request_delay_seconds = (
        _brave_search_request_delay_seconds()
        if request_delay_seconds is None
        else request_delay_seconds
    )
    if active_request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must not be negative")

    sent_live_request = False
    for index, item in enumerate(transaction.receipt.items):
        query = _receipt_item_query(transaction.payee or "", item)
        if not query:
            continue
        query_is_cached = (
            active_client.is_cached(query)
            if isinstance(active_client, CachedBraveSearchClient)
            else False
        )
        if sent_live_request and not query_is_cached and active_request_delay_seconds > 0:
            logger.info(
                "Sleeping %.2f seconds before the next Brave Search query",
                active_request_delay_seconds,
            )
            sleep(active_request_delay_seconds)
        logger.info("Enriching receipt item %s with Brave Search: %s", index + 1, query)
        result = active_client.search(query)
        if not query_is_cached:
            sent_live_request = True
        item.brave_search_result = json.dumps(_search_result_summaries(result), sort_keys=True)
        logger.info("Stored Brave Search response on receipt item %s", index + 1)
    return transaction


def enrich_transactions_with_brave_search(
    transactions: list[Transaction] | tuple[Transaction, ...],
    *,
    client: BraveSearchClient | None = None,
    request_delay_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Transaction] | tuple[Transaction, ...]:
    active_client = client if client is not None else create_brave_search_client()
    active_request_delay_seconds = (
        _brave_search_request_delay_seconds()
        if request_delay_seconds is None
        else request_delay_seconds
    )
    if active_request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must not be negative")

    sent_live_request = False
    for index, transaction in enumerate(transactions):
        query = _transaction_query(transaction)
        if not query:
            continue
        query_is_cached = (
            active_client.is_cached(query)
            if isinstance(active_client, CachedBraveSearchClient)
            else False
        )
        if sent_live_request and not query_is_cached and active_request_delay_seconds > 0:
            logger.info(
                "Sleeping %.2f seconds before the next Brave Search query",
                active_request_delay_seconds,
            )
            sleep(active_request_delay_seconds)
        logger.info("Enriching transaction %s with Brave Search: %s", index + 1, query)
        result = active_client.search(query)
        if not query_is_cached:
            sent_live_request = True
        transaction.brave_search_result = json.dumps(_search_result_summaries(result), sort_keys=True)
        logger.info("Stored Brave Search response on transaction %s", index + 1)
    return transactions


def _search_result_summaries(result: Any) -> list[dict[str, str]]:
    if not isinstance(result, dict):
        return []

    result_object = cast(dict[str, object], result)
    web = result_object.get("web")
    if not isinstance(web, dict):
        return []

    web_object = cast(dict[str, object], web)
    results = web_object.get("results")
    if not isinstance(results, list):
        return []

    summaries: list[dict[str, str]] = []
    for search_result_object in cast(list[object], results):
        if not isinstance(search_result_object, dict):
            continue
        search_result = cast(dict[str, object], search_result_object)

        title = search_result.get("title")
        description = search_result.get("description")
        summaries.append(
            {
                "description": description if isinstance(description, str) else "",
                "title": title if isinstance(title, str) else "",
            }
        )

    return summaries


def _receipt_item_query(payee: str, item: ReceiptItem) -> str:
    description = item.raw_description or item.description
    price = item.unit_price if item.unit_price is not None else item.amount
    return " ".join((payee.strip(), description.strip(), _format_item_amount(price)))


def _transaction_query(transaction: Transaction) -> str:
    parts = [transaction.payee]
    if transaction.description and transaction.description != transaction.payee:
        parts.append(transaction.description)
    return " ".join(part.strip() for part in parts if part and part.strip())


def _format_item_amount(amount: str) -> str:
    decimal_amount = Decimal(amount)
    absolute_amount = abs(decimal_amount)
    exponent = decimal_amount.as_tuple().exponent
    decimal_places = max(2, -exponent if isinstance(exponent, int) else 2)
    prefix = "-$" if decimal_amount < 0 else "$"
    return f"{prefix}{absolute_amount:.{decimal_places}f}"


def _brave_search_endpoint() -> str:
    for env_var in ENDPOINT_ENV_VARS:
        endpoint = os.getenv(env_var)
        if endpoint:
            return endpoint

    return DEFAULT_BRAVE_SEARCH_ENDPOINT


def _brave_search_key() -> str:
    for env_var in KEY_ENV_VARS:
        key = os.getenv(env_var)
        if key:
            return key

    env_var_list = ", ".join(KEY_ENV_VARS)
    raise RuntimeError(f"Set one of these environment variables: {env_var_list}")


def _brave_search_request_delay_seconds() -> float:
    for env_var in REQUEST_DELAY_SECONDS_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            try:
                delay = float(value)
            except ValueError as exc:
                raise RuntimeError(f"{env_var} must be a number of seconds") from exc
            if delay < 0:
                raise RuntimeError(f"{env_var} must not be negative")
            return delay

    return 0.0


def _url_with_query(endpoint: str, params: dict[str, str]) -> str:
    separator = "&" if urllib.parse.urlparse(endpoint).query else "?"
    return f"{endpoint}{separator}{urllib.parse.urlencode(params)}"
