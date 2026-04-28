from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from importlib import resources
from typing import Protocol, cast

from receipts_ai.models.transaction import ReceiptItem, Transaction

DEFAULT_OLLAMA_URL = "http://localhost:11434"
OLLAMA_URL_ENV_VARS = ("OLLAMA_URL",)
OLLAMA_MODEL_ENV_VARS = ("OLLAMA_MODEL", "OLLAMA_MODEL_NAME")

logger = logging.getLogger(__name__)


class CategoryModelClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class UrlLibOllamaClient:
    def __init__(self, *, url: str, model: str, timeout_seconds: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.generate_url = (
            self.url if self.url.endswith("/api/generate") else f"{self.url}/api/generate"
        )
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty")

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.generate_url,
            data=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )

        logger.info(
            "Sending Ollama generate request: url=%s model=%s prompt_chars=%s",
            self.generate_url,
            self.model,
            len(prompt),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request to {self.generate_url} failed with HTTP {exc.code}: {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request to {self.generate_url} failed: {exc.reason}"
            ) from exc

        result: object = json.loads(response_payload)
        if not isinstance(result, dict):
            raise RuntimeError("Ollama response was not a JSON object")
        result_object = cast(dict[str, object], result)
        response_text = result_object.get("response")
        if not isinstance(response_text, str):
            raise RuntimeError("Ollama response did not include a response string")
        logger.info(
            "Received Ollama generate response: url=%s model=%s response_chars=%s",
            self.generate_url,
            self.model,
            len(response_text),
        )
        return response_text


def create_ollama_category_client() -> CategoryModelClient:
    return UrlLibOllamaClient(url=_ollama_url(), model=_ollama_model())


def categorize_receipt_items(
    transaction: Transaction,
    *,
    client: CategoryModelClient | None = None,
    categories: dict[str, object] | None = None,
) -> Transaction:
    if transaction.receipt is None:
        raise ValueError("transaction does not contain a receipt")

    active_client = client if client is not None else create_ollama_category_client()
    active_categories = categories if categories is not None else load_budget_categories()
    top_level_categories = tuple(active_categories.keys())

    for index, item in enumerate(transaction.receipt.items, start=1):
        logger.info("Categorizing receipt item %s", index)
        top_level_category = _choose_category(
            client=active_client,
            prompt=_top_level_prompt(item, top_level_categories),
            choices=top_level_categories,
        )
        leaf_categories = tuple(_leaf_categories(active_categories[top_level_category]))
        if not leaf_categories:
            raise RuntimeError(f"budget category has no leaf categories: {top_level_category}")

        item.category_id = _choose_category(
            client=active_client,
            prompt=_leaf_category_prompt(item, top_level_category, leaf_categories),
            choices=leaf_categories,
        )
        logger.info("Categorized receipt item %s as %s", index, item.category_id)

    return transaction


def load_budget_categories() -> dict[str, object]:
    categories_file = resources.files("receipts_ai.models").joinpath("budget_categories.json")
    with categories_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise RuntimeError("budget_categories.json must contain a JSON object")
    return cast(dict[str, object], payload)


def _choose_category(*, client: CategoryModelClient, prompt: str, choices: tuple[str, ...]) -> str:
    response = client.complete(prompt)
    choice = _normalize_choice(response, choices)
    if choice is None:
        options = ", ".join(choices)
        raise RuntimeError(
            f"model returned an invalid category {response!r}; expected one of: {options}"
        )
    return choice


def _top_level_prompt(item: ReceiptItem, categories: tuple[str, ...]) -> str:
    return _category_prompt(
        "Choose the best top level budget category.",
        categories=categories,
        search_results=_search_results_text(item),
    )


def _leaf_category_prompt(
    item: ReceiptItem, top_level_category: str, categories: tuple[str, ...]
) -> str:
    return _category_prompt(
        f"Top level category: {top_level_category}\nChoose the best specific budget category.",
        categories=categories,
        search_results=_search_results_text(item),
    )


def _category_prompt(instruction: str, *, categories: tuple[str, ...], search_results: str) -> str:
    options = "\n".join(f"- {category}" for category in categories)
    return (
        f"{instruction}\n"
        "Return only one exact category name from Options.\n"
        "Use only these search result titles and descriptions for the receipt item.\n\n"
        f"Options:\n{options}\n\n"
        f"Search results:\n{search_results}"
    )


def _search_results_text(item: ReceiptItem) -> str:
    if item.brave_search_result is None:
        return "No search results."

    try:
        payload: object = json.loads(item.brave_search_result)
    except json.JSONDecodeError:
        return item.brave_search_result

    if not isinstance(payload, list):
        return "No search results."

    lines: list[str] = []
    for index, result_object in enumerate(cast(list[object], payload), start=1):
        if not isinstance(result_object, dict):
            continue
        result = cast(dict[str, object], result_object)
        title = result.get("title")
        description = result.get("description")
        lines.append(
            "\n".join(
                (
                    f"{index}. Title: {title if isinstance(title, str) else ''}",
                    f"Description: {description if isinstance(description, str) else ''}",
                )
            )
        )

    return "\n".join(lines) if lines else "No search results."


def _leaf_categories(node: object) -> list[str]:
    if not isinstance(node, dict):
        return []
    node_object = cast(dict[str, object], node)
    if not node_object:
        return []

    leaves: list[str] = []
    for category, child in node_object.items():
        if isinstance(child, dict) and child:
            leaves.extend(_leaf_categories(cast(dict[str, object], child)))
        else:
            leaves.append(category)
    return leaves


def _normalize_choice(response: str, choices: tuple[str, ...]) -> str | None:
    cleaned_response = _normalize_text(response)
    for choice in choices:
        if cleaned_response == _normalize_text(choice):
            return choice

    response_lines = [line.strip(" \t-*:\"'`") for line in response.splitlines()]
    for line in response_lines:
        normalized_line = _normalize_text(line)
        for choice in choices:
            if normalized_line == _normalize_text(choice):
                return choice

    for choice in choices:
        if re.search(rf"\b{re.escape(_normalize_text(choice))}\b", cleaned_response):
            return choice
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _ollama_url() -> str:
    for env_var in OLLAMA_URL_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value
    return DEFAULT_OLLAMA_URL


def _ollama_model() -> str:
    for env_var in OLLAMA_MODEL_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value

    env_var_list = ", ".join(OLLAMA_MODEL_ENV_VARS)
    raise RuntimeError(f"Set one of these environment variables: {env_var_list}")
