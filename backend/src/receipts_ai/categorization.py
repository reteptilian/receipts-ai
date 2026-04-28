from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Protocol, cast

from receipts_ai.cache import JsonCallCache
from receipts_ai.models.transaction import ReceiptItem, Transaction

DEFAULT_OLLAMA_URL = "http://localhost:11434"
OLLAMA_URL_ENV_VARS = ("OLLAMA_URL",)
OLLAMA_MODEL_ENV_VARS = ("OLLAMA_MODEL", "OLLAMA_MODEL_NAME")

MAX_SEARCH_RESULTS = 5
MAX_TAXONOMY_LEVELS = 9
PRODUCT_TAXONOMY_FILENAME = "taxonomy.en-US.txt"

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

        logger.debug("Prompt: %s", prompt)
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


class CachedCategoryModelClient:
    def __init__(
        self,
        *,
        cache: JsonCallCache,
        client: CategoryModelClient | None = None,
        client_factory: Callable[[], CategoryModelClient] | None = None,
    ) -> None:
        if client is None and client_factory is None:
            raise ValueError("client or client_factory is required")
        self.cache = cache
        self._client = client
        self._client_factory = client_factory

    def complete(self, prompt: str) -> str:
        request = {"prompt": prompt}
        cached_response = self.cache.get("ollama", request)
        if isinstance(cached_response, str):
            logger.info("Using cached Ollama response: prompt_chars=%s", len(prompt))
            return cached_response

        response = self._active_client().complete(prompt)
        self.cache.set("ollama", request, response)
        logger.info("Cached Ollama response: prompt_chars=%s", len(prompt))
        return response

    def _active_client(self) -> CategoryModelClient:
        if self._client is None:
            if self._client_factory is None:
                raise RuntimeError("client_factory is not configured")
            self._client = self._client_factory()
        return self._client


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


def classify_receipt_items_by_product_taxonomy(
    transaction: Transaction,
    *,
    client: CategoryModelClient | None = None,
    taxonomy: dict[str, object] | None = None,
) -> Transaction:
    if transaction.receipt is None:
        raise ValueError("transaction does not contain a receipt")

    active_client = client if client is not None else create_ollama_category_client()
    active_taxonomy = taxonomy if taxonomy is not None else load_product_taxonomy()

    for index, item in enumerate(transaction.receipt.items, start=1):
        logger.info("Classifying receipt item %s by product taxonomy", index)
        choices = tuple(active_taxonomy.keys())
        node: dict[str, object] = active_taxonomy
        selected_path: list[str] = []

        while choices and len(selected_path) < MAX_TAXONOMY_LEVELS:
            choice = _choose_taxonomy_category(
                client=active_client,
                prompt=_product_taxonomy_prompt(item, selected_path, choices),
                choices=choices,
                selected_path=selected_path,
            )
            if choice is None:
                break
            selected_path.append(choice)
            child = node.get(choice)
            if not isinstance(child, dict) or not child:
                break
            node = cast(dict[str, object], child)
            choices = tuple(node.keys())

        _set_item_taxonomy(item, selected_path)
        logger.info("Classified receipt item %s as %s", index, " > ".join(selected_path))

    return transaction


def load_budget_categories() -> dict[str, object]:
    categories_file = resources.files("receipts_ai.models").joinpath("budget_categories.json")
    with categories_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise RuntimeError("budget_categories.json must contain a JSON object")
    return cast(dict[str, object], payload)


def load_product_taxonomy(path: Path | None = None) -> dict[str, object]:
    taxonomy_path = path if path is not None else _default_product_taxonomy_path()
    taxonomy: dict[str, object] = {}
    with taxonomy_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            parts = tuple(part.strip() for part in stripped_line.split(">") if part.strip())
            if not parts:
                continue
            node = taxonomy
            for part in parts:
                child = node.setdefault(part, {})
                if not isinstance(child, dict):
                    raise RuntimeError(f"taxonomy path conflicts at {part!r}")
                node = cast(dict[str, object], child)
    if not taxonomy:
        raise RuntimeError(f"{taxonomy_path} did not contain product taxonomy entries")
    return taxonomy


def _choose_category(*, client: CategoryModelClient, prompt: str, choices: tuple[str, ...]) -> str:
    response = client.complete(prompt)
    choice = _normalize_choice(response, choices)
    if choice is None:
        options = ", ".join(choices)
        raise RuntimeError(
            f"model returned an invalid category {response!r}; expected one of: {options}"
        )
    return choice


def _choose_taxonomy_category(
    *,
    client: CategoryModelClient,
    prompt: str,
    choices: tuple[str, ...],
    selected_path: list[str],
) -> str | None:
    response = client.complete(prompt)
    choice = _normalize_choice(response, choices)
    if choice is not None:
        return choice

    if selected_path and _normalize_choice(response, tuple(selected_path)) is not None:
        logger.info(
            "Model repeated selected taxonomy path category %r; stopping taxonomy walk", response
        )
        return None

    options = ", ".join(choices)
    raise RuntimeError(
        f"model returned an invalid category {response!r}; expected one of: {options}"
    )


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


def _product_taxonomy_prompt(
    item: ReceiptItem, selected_path: list[str], choices: tuple[str, ...]
) -> str:
    path_text = " > ".join(selected_path)
    instruction = (
        "Based on these search result titles and descriptions, "
        "pick the most appropriate product type from the following choices."
    )
    if path_text:
        instruction = f"Selected product taxonomy path: {path_text}\n{instruction}"
    return _category_prompt(
        instruction,
        categories=choices,
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

    return "\n".join(lines[:MAX_SEARCH_RESULTS]) if lines else "No search results."


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


def _set_item_taxonomy(item: ReceiptItem, selected_path: list[str]) -> None:
    for level in range(1, MAX_TAXONOMY_LEVELS + 1):
        value = selected_path[level - 1] if level <= len(selected_path) else None
        setattr(item, f"taxonomy{level}", value)


def _default_product_taxonomy_path() -> Path:
    package_taxonomy = resources.files("receipts_ai.models").joinpath(PRODUCT_TAXONOMY_FILENAME)
    if package_taxonomy.is_file():
        return Path(str(package_taxonomy))

    repo_taxonomy = Path(__file__).resolve().parents[3] / "models" / PRODUCT_TAXONOMY_FILENAME
    if repo_taxonomy.is_file():
        return repo_taxonomy

    raise RuntimeError(f"could not find {PRODUCT_TAXONOMY_FILENAME}")


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
