from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from receipts_ai.models.transaction import Receipt

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


def create_brave_search_client() -> BraveSearchClient:
    return UrlLibBraveSearchClient(endpoint=_brave_search_endpoint(), key=_brave_search_key())


def enrich_receipt_items_with_brave_search(
    receipt: Receipt,
    *,
    client: BraveSearchClient | None = None,
    request_delay_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Receipt:
    active_client = client if client is not None else create_brave_search_client()
    active_request_delay_seconds = (
        _brave_search_request_delay_seconds()
        if request_delay_seconds is None
        else request_delay_seconds
    )
    if active_request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must not be negative")

    for index, item in enumerate(receipt.items):
        query = item.raw_description or item.description
        if not query:
            continue
        if index > 0 and active_request_delay_seconds > 0:
            logger.info(
                "Sleeping %.2f seconds before the next Brave Search query",
                active_request_delay_seconds,
            )
            sleep(active_request_delay_seconds)
        logger.info("Enriching receipt item %s with Brave Search: %s", index + 1, query)
        result = active_client.search(query)
        item.brave_search_result = json.dumps(result, sort_keys=True)
        logger.info("Stored Brave Search response on receipt item %s", index + 1)
    return receipt


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
