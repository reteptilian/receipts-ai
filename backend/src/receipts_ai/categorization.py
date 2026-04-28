from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Protocol, cast

from receipts_ai.cache import JsonCallCache
from receipts_ai.models.transaction import ReceiptItem, Transaction

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 30.0
OLLAMA_URL_ENV_VARS = ("OLLAMA_URL",)
OLLAMA_MODEL_ENV_VARS = ("OLLAMA_MODEL", "OLLAMA_MODEL_NAME")
OLLAMA_TIMEOUT_ENV_VARS = ("OLLAMA_TIMEOUT_SECONDS",)

MAX_SEARCH_RESULTS = 5
MAX_TAXONOMY_SEARCH_PATHS = 5
MAX_TAXONOMY_LEVELS = 9
TAXONOMY_MIN_CHOICE_PROBABILITY = 0.35
TAXONOMY_MIN_SEARCH_PROBABILITY = 0.15
PRODUCT_TAXONOMY_FILENAME = "taxonomy.en-US.txt"

logger = logging.getLogger(__name__)


class CategoryModelClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class CategoryChoiceProbability:
    choice: str
    probability: float


@dataclass(frozen=True)
class CategoryCompletion:
    response: str
    probabilities: tuple[CategoryChoiceProbability, ...]


@dataclass(frozen=True)
class _TaxonomySearchPath:
    path: tuple[str, ...]
    node: dict[str, object]
    probability: float


class UrlLibOllamaClient:
    def __init__(
        self, *, url: str, model: str, timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    ) -> None:
        self.url = url.rstrip("/")
        self.generate_url = (
            self.url if self.url.endswith("/api/generate") else f"{self.url}/api/generate"
        )
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> str:
        return self._generate(
            prompt, options={"temperature": 0, "num_predict": 64, "stop": ["\n"]}
        ).response

    def complete_choice(self, prompt: str, *, choices: tuple[str, ...]) -> str:
        if not choices:
            raise ValueError("choices must not be empty")

        schema = _category_choice_schema(choices)
        schema_prompt = (
            f"{prompt}\n\n"
            "Return only JSON matching this schema. Do not include explanation text.\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        completion = self._generate(
            schema_prompt,
            options={"temperature": 0},
            output_format=schema,
        )
        choice = _category_choice_from_json_response(completion.response, choices)
        if choice is not None:
            return choice
        return completion.response

    def complete_with_probabilities(
        self, prompt: str, *, choices: tuple[str, ...]
    ) -> CategoryCompletion:
        completion = self._generate(
            prompt,
            options={"temperature": 0, "logprobs": True, "top_logprobs": 5},
            logprobs=True,
            top_logprobs=5,
        )
        return CategoryCompletion(
            response=completion.response,
            probabilities=_choice_probabilities_from_ollama_response(
                completion.raw_response,
                response_text=completion.response,
                choices=choices,
            ),
        )

    def _generate(
        self,
        prompt: str,
        *,
        options: dict[str, object],
        output_format: str | dict[str, object] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> _OllamaCompletion:
        if not prompt:
            raise ValueError("prompt must not be empty")

        # logger.debug("Prompt: %s", prompt)
        request_payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }
        if output_format is not None:
            request_payload["format"] = output_format
        if logprobs is not None:
            request_payload["logprobs"] = logprobs
        if top_logprobs is not None:
            request_payload["top_logprobs"] = top_logprobs
        payload = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            self.generate_url,
            data=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )

        logger.info(
            "Sending Ollama generate request: url=%s model=%s prompt_chars=%s "
            "timeout_seconds=%s stream=%s think=%s format=%s logprobs=%s "
            "top_logprobs=%s options=%s payload_bytes=%s",
            self.generate_url,
            self.model,
            len(prompt),
            self.timeout_seconds,
            request_payload["stream"],
            request_payload["think"],
            _format_ollama_request_value(request_payload.get("format")),
            request_payload.get("logprobs", "unset"),
            request_payload.get("top_logprobs", "unset"),
            _format_ollama_request_value(options),
            len(payload),
        )
        logger.debug(
            "Ollama generate curl reproduction command: %s",
            _ollama_curl_command(self.generate_url, request_payload),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = response.read().decode("utf-8")
        except TimeoutError as exc:
            # socket.timeout is an alias of TimeoutError on supported Python versions.
            raise RuntimeError(
                f"Ollama request to {self.generate_url} timed out: "
                f"model={self.model} prompt_chars={len(prompt)} "
                f"timeout_seconds={self.timeout_seconds}"
            ) from exc
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
        logger.debug(
            "Ollama generate response metadata: has_logprobs=%s top_level_keys=%s",
            "logprobs" in result_object,
            ", ".join(sorted(result_object.keys())),
        )
        _log_ollama_generate_stats(
            result_object,
            url=self.generate_url,
            model=self.model,
        )
        return _OllamaCompletion(response=response_text, raw_response=result_object)


@dataclass(frozen=True)
class _OllamaCompletion:
    response: str
    raw_response: dict[str, object]


def _format_ollama_request_value(value: object) -> str:
    if value is None:
        return "unset"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return repr(value)


def _ollama_curl_command(url: str, request_payload: dict[str, object]) -> str:
    payload = json.dumps(request_payload, ensure_ascii=False, sort_keys=True)
    return " ".join(
        (
            "curl",
            "-sS",
            shlex.quote(url),
            "-H",
            shlex.quote("Accept: application/json"),
            "-H",
            shlex.quote("Content-Type: application/json"),
            "-d",
            shlex.quote(payload),
        )
    )


def _log_ollama_generate_stats(response: dict[str, object], *, url: str, model: str) -> None:
    logger.debug(
        "Ollama generate stats: url=%s model=%s total=%s load=%s "
        "prompt_eval=%s eval=%s prompt_tokens=%s eval_tokens=%s eval_tokens_per_second=%s",
        url,
        model,
        _format_ollama_duration(response.get("total_duration")),
        _format_ollama_duration(response.get("load_duration")),
        _format_ollama_duration(response.get("prompt_eval_duration")),
        _format_ollama_duration(response.get("eval_duration")),
        _format_ollama_count(response.get("prompt_eval_count")),
        _format_ollama_count(response.get("eval_count")),
        _format_ollama_eval_rate(response),
    )


def _format_ollama_duration(value: object) -> str:
    if not isinstance(value, int | float):
        return "unknown"
    return f"{float(value) / 1_000_000_000:.3f}s"


def _format_ollama_count(value: object) -> str:
    if not isinstance(value, int | float):
        return "unknown"
    return str(int(value))


def _format_ollama_eval_rate(response: dict[str, object]) -> str:
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    if (
        not isinstance(eval_count, int | float)
        or not isinstance(eval_duration, int | float)
        or eval_duration <= 0
    ):
        return "unknown"
    return f"{float(eval_count) / (float(eval_duration) / 1_000_000_000):.2f}"


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

    def complete_choice(self, prompt: str, *, choices: tuple[str, ...]) -> str:
        request = {
            "prompt": prompt,
            "choices": list(choices),
            "format": "category_choice_schema_v1",
        }
        cached_response = self.cache.get("ollama", request)
        if isinstance(cached_response, str):
            logger.info("Using cached Ollama choice response: prompt_chars=%s", len(prompt))
            return cached_response

        response = _complete_choice(self._active_client(), prompt, choices=choices)
        self.cache.set("ollama", request, response)
        logger.info("Cached Ollama choice response: prompt_chars=%s", len(prompt))
        return response

    def complete_with_probabilities(
        self, prompt: str, *, choices: tuple[str, ...]
    ) -> CategoryCompletion:
        request = {
            "prompt": prompt,
            "logprobs": True,
            "top_logprobs": 5,
        }
        cached_response = self.cache.get("ollama", request)
        if isinstance(cached_response, dict):
            cached_response_object = cast(dict[str, object], cached_response)
            response_text = cached_response_object.get("response")
            probabilities = cached_response_object.get("probabilities")
            if isinstance(response_text, str) and isinstance(probabilities, list):
                stored_probabilities = cast(list[object], probabilities)
                logger.info(
                    "Using cached Ollama probability response: prompt_chars=%s", len(prompt)
                )
                return CategoryCompletion(
                    response=response_text,
                    probabilities=_stored_choice_probabilities(
                        stored_probabilities, choices=choices
                    ),
                )

        active_client = self._active_client()
        completion = _complete_with_probabilities(active_client, prompt, choices=choices)
        self.cache.set(
            "ollama",
            request,
            {
                "response": completion.response,
                "probabilities": [
                    {"choice": result.choice, "probability": result.probability}
                    for result in completion.probabilities
                ],
            },
        )
        logger.info("Cached Ollama probability response: prompt_chars=%s", len(prompt))
        return completion

    def _active_client(self) -> CategoryModelClient:
        if self._client is None:
            if self._client_factory is None:
                raise RuntimeError("client_factory is not configured")
            self._client = self._client_factory()
        return self._client


def create_ollama_category_client() -> CategoryModelClient:
    return UrlLibOllamaClient(
        url=_ollama_url(),
        model=_ollama_model(),
        timeout_seconds=_ollama_timeout_seconds(),
    )


def clean_receipt_item_descriptions(
    transaction: Transaction,
    *,
    client: CategoryModelClient | None = None,
) -> Transaction:
    if transaction.receipt is None:
        raise ValueError("transaction does not contain a receipt")

    active_client = client if client is not None else create_ollama_category_client()

    for index, item in enumerate(transaction.receipt.items, start=1):
        logger.info("Cleaning receipt item description %s", index)
        cleaned_description = _clean_description_response(
            active_client.complete(_description_prompt(item))
        )
        if cleaned_description:
            item.description = cleaned_description
            logger.info(
                "Cleaned receipt item description %s as %s",
                index,
                cleaned_description,
            )
        else:
            logger.warning(
                "Skipping empty cleaned receipt item description for item %s",
                index,
            )

    return transaction


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
        logger.info("Categorizing receipt item %s", item.description)
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
        logger.info("Categorized receipt item %s as %s", item.description, item.category_id)

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
        logger.info("Classifying receipt item %s by product taxonomy", item.description)
        selected_path = _search_taxonomy_path(
            item=item,
            taxonomy=active_taxonomy,
            client=active_client,
        )

        _set_item_taxonomy(item, list(selected_path))
        logger.info("Classified receipt item %s as %s", item.description, " > ".join(selected_path))

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
    response = _complete_choice(client, prompt, choices=choices)
    choice = _normalize_choice(response, choices)
    if choice is None:
        options = ", ".join(choices)
        raise RuntimeError(
            f"model returned an invalid category {response!r}; expected one of: {options}"
        )
    return choice


def _complete_choice(client: CategoryModelClient, prompt: str, *, choices: tuple[str, ...]) -> str:
    complete_choice = getattr(client, "complete_choice", None)
    if callable(complete_choice):
        response = complete_choice(prompt, choices=choices)
        if isinstance(response, str):
            return response

    return client.complete(prompt)


def _search_taxonomy_path(
    *,
    item: ReceiptItem,
    taxonomy: dict[str, object],
    client: CategoryModelClient,
) -> tuple[str, ...]:
    search_paths = (_TaxonomySearchPath(path=(), node=taxonomy, probability=1.0),)
    best_path: tuple[str, ...] = ()
    best_probability = 0.0

    logger.debug(
        "Starting taxonomy search: active_paths=%s",
        _format_taxonomy_search_paths(search_paths),
    )
    while search_paths:
        current, search_paths = search_paths[0], search_paths[1:]
        logger.debug(
            "Expanding taxonomy path: path=%s probability=%.3f remaining_paths=%s",
            _format_taxonomy_path(current.path),
            current.probability,
            _format_taxonomy_search_paths(search_paths),
        )

        choices = tuple(current.node.keys())
        if not choices or len(current.path) >= MAX_TAXONOMY_LEVELS:
            reason = "leaf" if not choices else "max-level"
            best_path, best_probability = _preferred_taxonomy_candidate(
                best_path,
                best_probability,
                current.path,
                current.probability,
            )
            logger.debug(
                "Completed taxonomy path: path=%s probability=%.3f reason=%s "
                "best_path=%s best_probability=%.3f",
                _format_taxonomy_path(current.path),
                current.probability,
                reason,
                _format_taxonomy_path(best_path),
                best_probability,
            )
            continue

        logger.debug(
            "Requesting taxonomy choices: path=%s choices=%s",
            _format_taxonomy_path(current.path),
            ", ".join(choices),
        )
        choice_results = _choose_taxonomy_categories(
            client=client,
            prompt=_product_taxonomy_prompt(item, list(current.path), choices),
            choices=choices,
            selected_path=list(current.path),
        )
        logger.debug(
            "Received taxonomy choices: path=%s results=%s",
            _format_taxonomy_path(current.path),
            _format_category_choice_probabilities(choice_results),
        )
        if not choice_results:
            best_path, best_probability = _preferred_taxonomy_candidate(
                best_path,
                best_probability,
                current.path,
                current.probability,
            )
            logger.debug(
                "Stopped taxonomy path without new choices: path=%s probability=%.3f "
                "best_path=%s best_probability=%.3f",
                _format_taxonomy_path(current.path),
                current.probability,
                _format_taxonomy_path(best_path),
                best_probability,
            )
            continue

        top_result = choice_results[0]
        if top_result.probability < TAXONOMY_MIN_CHOICE_PROBABILITY:
            logger.debug(
                "Stopping taxonomy path %s because top choice %s probability %.3f is below %.3f",
                " > ".join(current.path) if current.path else "<root>",
                top_result.choice,
                top_result.probability,
                TAXONOMY_MIN_CHOICE_PROBABILITY,
            )
            best_path, best_probability = _preferred_taxonomy_candidate(
                best_path,
                best_probability,
                current.path,
                current.probability,
            )
            logger.debug(
                "Stopped low-confidence taxonomy path: path=%s best_path=%s best_probability=%.3f",
                _format_taxonomy_path(current.path),
                _format_taxonomy_path(best_path),
                best_probability,
            )
            continue

        new_paths: list[_TaxonomySearchPath] = []
        for result in choice_results:
            if result.probability < TAXONOMY_MIN_SEARCH_PROBABILITY:
                logger.debug(
                    "Discarding taxonomy choice below search threshold: path=%s choice=%s "
                    "probability=%.3f threshold=%.3f",
                    _format_taxonomy_path(current.path),
                    result.choice,
                    result.probability,
                    TAXONOMY_MIN_SEARCH_PROBABILITY,
                )
                continue

            child = current.node.get(result.choice)
            next_path = (*current.path, result.choice)
            next_probability = current.probability * result.probability
            if not isinstance(child, dict) or not child:
                best_path, best_probability = _preferred_taxonomy_candidate(
                    best_path,
                    best_probability,
                    next_path,
                    next_probability,
                )
                logger.debug(
                    "Reached terminal taxonomy choice: path=%s local_probability=%.3f "
                    "cumulative_probability=%.3f best_path=%s best_probability=%.3f",
                    _format_taxonomy_path(next_path),
                    result.probability,
                    next_probability,
                    _format_taxonomy_path(best_path),
                    best_probability,
                )
                continue

            new_path = _TaxonomySearchPath(
                path=next_path,
                node=cast(dict[str, object], child),
                probability=next_probability,
            )
            new_paths.append(new_path)
            logger.debug(
                "Added taxonomy search path: path=%s local_probability=%.3f "
                "cumulative_probability=%.3f",
                _format_taxonomy_path(next_path),
                result.probability,
                next_probability,
            )

        search_paths = _prune_taxonomy_search_paths((*search_paths, *new_paths))
        logger.debug(
            "Pruned taxonomy search set: active_paths=%s best_path=%s best_probability=%.3f",
            _format_taxonomy_search_paths(search_paths),
            _format_taxonomy_path(best_path),
            best_probability,
        )

    logger.debug(
        "Finished taxonomy search: best_path=%s best_probability=%.3f",
        _format_taxonomy_path(best_path),
        best_probability,
    )
    return best_path


def _preferred_taxonomy_candidate(
    current_path: tuple[str, ...],
    current_probability: float,
    candidate_path: tuple[str, ...],
    candidate_probability: float,
) -> tuple[tuple[str, ...], float]:
    if not candidate_path:
        return current_path, current_probability
    if len(candidate_path) > len(current_path):
        return candidate_path, candidate_probability
    if len(candidate_path) == len(current_path) and candidate_probability > current_probability:
        return candidate_path, candidate_probability
    return current_path, current_probability


def _choose_taxonomy_categories(
    *,
    client: CategoryModelClient,
    prompt: str,
    choices: tuple[str, ...],
    selected_path: list[str],
) -> tuple[CategoryChoiceProbability, ...]:
    completion = _complete_with_probabilities(client, prompt, choices=choices)
    probabilities = tuple(result for result in completion.probabilities if result.choice in choices)
    if probabilities:
        return probabilities

    choice = _normalize_choice(completion.response, choices)
    if choice is not None:
        return (CategoryChoiceProbability(choice=choice, probability=1.0),)

    if selected_path and _normalize_choice(completion.response, tuple(selected_path)) is not None:
        logger.info(
            "Model repeated selected taxonomy path category %r; stopping taxonomy walk",
            completion.response,
        )
        return ()

    options = ", ".join(choices)
    raise RuntimeError(
        f"model returned an invalid category {completion.response!r}; expected one of: {options}"
    )


def _complete_with_probabilities(
    client: CategoryModelClient, prompt: str, *, choices: tuple[str, ...]
) -> CategoryCompletion:
    complete_with_probabilities = getattr(client, "complete_with_probabilities", None)
    if callable(complete_with_probabilities):
        completion = complete_with_probabilities(prompt, choices=choices)
        if isinstance(completion, CategoryCompletion):
            return completion

    return CategoryCompletion(response=client.complete(prompt), probabilities=())


def _prune_taxonomy_search_paths(
    search_paths: tuple[_TaxonomySearchPath, ...],
) -> tuple[_TaxonomySearchPath, ...]:
    deduped: dict[tuple[str, ...], _TaxonomySearchPath] = {}
    for search_path in search_paths:
        existing = deduped.get(search_path.path)
        if existing is None or search_path.probability > existing.probability:
            deduped[search_path.path] = search_path

    return tuple(
        sorted(deduped.values(), key=lambda path: path.probability, reverse=True)[
            :MAX_TAXONOMY_SEARCH_PATHS
        ]
    )


def _format_taxonomy_path(path: tuple[str, ...]) -> str:
    return " > ".join(path) if path else "<root>"


def _format_taxonomy_search_paths(search_paths: tuple[_TaxonomySearchPath, ...]) -> str:
    if not search_paths:
        return "<empty>"
    return "; ".join(
        f"{_format_taxonomy_path(search_path.path)} ({search_path.probability:.3f})"
        for search_path in search_paths
    )


def _format_category_choice_probabilities(
    probabilities: tuple[CategoryChoiceProbability, ...],
) -> str:
    if not probabilities:
        return "<empty>"
    return "; ".join(
        f"{probability.choice} ({probability.probability:.3f})" for probability in probabilities
    )


def _top_level_prompt(item: ReceiptItem, categories: tuple[str, ...]) -> str:
    return _category_prompt(
        "Choose the best top level budget category.",
        categories=categories,
        item_description=item.description,
    )


def _leaf_category_prompt(
    item: ReceiptItem, top_level_category: str, categories: tuple[str, ...]
) -> str:
    return _category_prompt(
        f"Top level category: {top_level_category}\nChoose the best specific budget category.",
        categories=categories,
        item_description=item.description,
    )


def _product_taxonomy_prompt(
    item: ReceiptItem, selected_path: list[str], choices: tuple[str, ...]
) -> str:
    # path_text = " > ".join(selected_path)
    instruction = (
        "Based on this receipt item description, "
        "pick the most appropriate product type from the following choices."
    )
    # if path_text:
    # instruction = f"Selected product taxonomy path: {path_text}\n{instruction}"
    return _category_prompt(
        instruction,
        categories=choices,
        item_description=item.description,
    )


def _description_prompt(item: ReceiptItem) -> str:
    raw_description = item.raw_description or item.description
    return (
        "Convert the raw receipt text into a clean product name using the search results. Remove prices, item numbers, and store info.\n"
        "Return only the cleaned item description, with no quotes, bullets, or explanation.\n\n"
        f"Raw receipt text: {raw_description}\n\n"
        f"Search results:\n{_search_results_text(item)}"
    )
    # return (
    #     "Create a clean, unabbreviated, user-facing description for this receipt item.\n"
    #     "Use the raw receipt text as the primary clue, and use only the search result "
    #     "titles and descriptions below to resolve cryptic abbreviations.\n"
    #     "Note that the search results are ordered with the most relevant results\n"
    #     "first and be aware that some search results may lead you astray.\n"
    #     "Expand likely product and brand abbreviations, remove receipt-only codes, "
    #     "SKU fragments, prices, quantities, and store bookkeeping text.\n"
    #     "Do not invent details that are not supported by the raw text or search results.\n"
    #     "Return only the cleaned item description, with no quotes, bullets, or explanation.\n\n"
    #     f"Raw receipt text: {raw_description}\n\n"
    #     f"Search results:\n{_search_results_text(item)}"
    # )


def _category_prompt(
    instruction: str, *, categories: tuple[str, ...], item_description: str
) -> str:
    options = "\n".join(f"- {category}" for category in categories)
    return (
        f"{instruction}\n"
        "Return only one exact category name from Options.\n\n"
        f"Options:\n{options}\n\n"
        f"Receipt item description: {item_description}"
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


def _clean_description_response(response: str) -> str:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return ""

    cleaned = lines[0].strip(" \t-*:\"'`")
    return re.sub(r"\s+", " ", cleaned).strip()


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


def _category_choice_schema(choices: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(choices),
            }
        },
        "required": ["category"],
        "additionalProperties": False,
    }


def _category_choice_from_json_response(response: str, choices: tuple[str, ...]) -> str | None:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, str):
        return _normalize_choice(payload, choices)
    if not isinstance(payload, dict):
        return None

    category = cast(dict[str, object], payload).get("category")
    if not isinstance(category, str):
        return None
    return _normalize_choice(category, choices)


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


def _choice_probabilities_from_ollama_response(
    response: dict[str, object], *, response_text: str, choices: tuple[str, ...]
) -> tuple[CategoryChoiceProbability, ...]:
    raw_probabilities = _raw_top_logprob_entries(response)
    if not raw_probabilities:
        logger.debug(
            "Ollama response did not include usable logprob entries; falling back to "
            "plain response normalization for %r",
            response_text,
        )
        return ()

    probabilities: dict[str, float] = {}

    for token_text, log_probability in raw_probabilities:
        choice = _normalize_choice(token_text, choices)
        if choice is None:
            choice = _choice_with_token_prefix(token_text, choices)
        if choice is None:
            continue

        probability = _probability_from_logprob(log_probability)
        probabilities[choice] = max(probabilities.get(choice, 0.0), probability)

    return tuple(
        CategoryChoiceProbability(choice=choice, probability=probability)
        for choice, probability in sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        )
    )


def _raw_top_logprob_entries(response: dict[str, object]) -> list[tuple[str, float]]:
    logprobs = response.get("logprobs")
    if isinstance(logprobs, list):
        return _raw_top_logprob_entries_from_tokens(cast(list[object], logprobs))
    if isinstance(logprobs, dict):
        logprobs_object = cast(dict[str, object], logprobs)
        content = logprobs_object.get("content")
        if isinstance(content, list):
            return _raw_top_logprob_entries_from_tokens(cast(list[object], content))

        top_logprobs = logprobs_object.get("top_logprobs")
        if isinstance(top_logprobs, list):
            return _raw_top_logprob_entries_from_top_logprobs(cast(list[object], top_logprobs))

    top_logprobs = response.get("top_logprobs")
    if isinstance(top_logprobs, list):
        return _raw_top_logprob_entries_from_top_logprobs(cast(list[object], top_logprobs))
    return []


def _raw_top_logprob_entries_from_tokens(tokens: list[object]) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    first_token = next((token for token in tokens if isinstance(token, dict)), None)
    if first_token is None:
        return entries

    token = cast(dict[str, object], first_token)
    top_logprobs = token.get("top_logprobs")
    if isinstance(top_logprobs, list):
        entries.extend(_raw_top_logprob_entries_from_top_logprobs(cast(list[object], top_logprobs)))

    token_text = token.get("token")
    log_probability = token.get("logprob")
    if isinstance(token_text, str) and isinstance(log_probability, int | float):
        entries.append((token_text, float(log_probability)))
    return entries


def _raw_top_logprob_entries_from_top_logprobs(
    top_logprobs: list[object],
) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    for entry_object in top_logprobs:
        if isinstance(entry_object, dict):
            entry = cast(dict[str, object], entry_object)
            token_text = entry.get("token")
            log_probability = entry.get("logprob")
            if isinstance(token_text, str) and isinstance(log_probability, int | float):
                entries.append((token_text, float(log_probability)))
            continue

        if (
            isinstance(entry_object, list | tuple)
            and len(entry := cast(list[object] | tuple[object, ...], entry_object)) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], int | float)
        ):
            entries.append((entry[0], float(entry[1])))
    return entries


def _choice_with_token_prefix(token_text: str, choices: tuple[str, ...]) -> str | None:
    normalized_token = _normalize_text(token_text).strip(".,:;-")
    if not normalized_token:
        return None

    matching_choices = [
        choice for choice in choices if _normalize_text(choice).startswith(normalized_token)
    ]
    if len(matching_choices) == 1:
        return matching_choices[0]
    return None


def _probability_from_logprob(log_probability: float) -> float:
    if log_probability > 0:
        return min(log_probability, 1.0)
    return min(math.exp(log_probability), 1.0)


def _stored_choice_probabilities(
    probabilities: list[object], *, choices: tuple[str, ...]
) -> tuple[CategoryChoiceProbability, ...]:
    results: list[CategoryChoiceProbability] = []
    for probability_object in probabilities:
        if not isinstance(probability_object, dict):
            continue
        probability = cast(dict[str, object], probability_object)
        choice = probability.get("choice")
        value = probability.get("probability")
        if isinstance(choice, str) and choice in choices and isinstance(value, int | float):
            results.append(
                CategoryChoiceProbability(
                    choice=choice,
                    probability=max(0.0, min(float(value), 1.0)),
                )
            )
    return tuple(sorted(results, key=lambda result: result.probability, reverse=True))


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


def _ollama_timeout_seconds() -> float:
    for env_var in OLLAMA_TIMEOUT_ENV_VARS:
        value = os.getenv(env_var)
        if not value:
            continue
        try:
            timeout_seconds = float(value)
        except ValueError as exc:
            raise RuntimeError(f"{env_var} must be a number of seconds") from exc
        if timeout_seconds <= 0:
            raise RuntimeError(f"{env_var} must be greater than 0")
        return timeout_seconds

    return DEFAULT_OLLAMA_TIMEOUT_SECONDS
