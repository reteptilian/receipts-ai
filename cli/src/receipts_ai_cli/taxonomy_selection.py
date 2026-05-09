from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache
from threading import Lock
from typing import cast

import sentence_transformers.sentence_transformer.model as sentence_transformer_model
from receipts_ai.categorization import (
    create_taxonomy_embedding_client,
    load_product_taxonomy,
    load_product_taxonomy_embeddings,
    search_product_taxonomy_embeddings,
)
from transformers.utils import logging as transformers_logging

LOGGER = logging.getLogger(__name__)


class _TransformersProgressBarGuard:
    def __enter__(self) -> None:
        self._progress_bar_enabled = transformers_logging.is_progress_bar_enabled()
        transformers_logging.disable_progress_bar()
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._progress_bar_enabled:
            transformers_logging.enable_progress_bar()
        return None


class _SentenceTransformersTrangeGuard:
    def __enter__(self) -> None:
        self._original_trange = sentence_transformer_model.trange
        sentence_transformer_model.trange = _plain_trange
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        sentence_transformer_model.trange = self._original_trange
        return None


def _plain_trange(*args: int, **kwargs: object) -> range:
    return range(*args)


def _format_taxonomy_path(parts: Iterable[str]) -> str:
    return " > ".join(parts)


def _taxonomy_choices_from_tree(
    taxonomy: dict[str, object],
    *,
    prefix: tuple[str, ...] = (),
) -> tuple[str, ...]:
    choices: list[str] = []
    for part, child in taxonomy.items():
        path = (*prefix, part)
        choices.append(_format_taxonomy_path(path))
        if isinstance(child, dict):
            choices.extend(_taxonomy_choices_from_tree(cast(dict[str, object], child), prefix=path))
    return tuple(choices)


class TaxonomySearcher:
    def __init__(self) -> None:
        self._embedding_index = load_product_taxonomy_embeddings()
        self._choices = tuple(_format_taxonomy_path(entry.path) for entry in self._embedding_index.entries)
        if not self._choices:
            taxonomy = load_product_taxonomy()
            self._choices = _taxonomy_choices_from_tree(taxonomy)
        self._embedding_client = None
        self._semantic_error: str | None = None
        self._embedding_client_lock = Lock()

    @property
    def choices(self) -> tuple[str, ...]:
        return self._choices

    @property
    def semantic_error(self) -> str | None:
        return self._semantic_error

    def exact_matches(self, query: str) -> tuple[str, ...]:
        cleaned_query = query.strip().casefold()
        if not cleaned_query:
            return self._choices

        scored_matches: list[tuple[tuple[int, int, int, int], str]] = []
        for index, choice in enumerate(self._choices):
            folded_choice = choice.casefold()
            position = folded_choice.find(cleaned_query)
            if position < 0:
                continue
            score = (
                0 if folded_choice == cleaned_query else 1 if folded_choice.startswith(cleaned_query) else 2,
                position,
                len(choice),
                index,
            )
            scored_matches.append((score, choice))
        scored_matches.sort(key=lambda item: item[0])
        return tuple(choice for _, choice in scored_matches)

    def semantic_matches(self, query: str, *, limit: int = 20) -> tuple[str, ...]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return ()

        try:
            embedding_client = self._embedding_client
            if embedding_client is None:
                with self._embedding_client_lock:
                    embedding_client = self._embedding_client
                    if embedding_client is None:
                        LOGGER.info(
                            "Loading taxonomy embedding model %s",
                            self._embedding_index.embedding_model,
                        )
                        with _TransformersProgressBarGuard():
                            embedding_client = create_taxonomy_embedding_client(
                                self._embedding_index.embedding_model
                            )
                        self._embedding_client = embedding_client
            with _TransformersProgressBarGuard(), _SentenceTransformersTrangeGuard():
                results = search_product_taxonomy_embeddings(
                    cleaned_query,
                    taxonomy_embeddings=self._embedding_index,
                    embedding_client=embedding_client,
                    candidate_count=limit,
                )
        except Exception as exc:
            self._semantic_error = str(exc)
            LOGGER.exception("Taxonomy semantic search failed for query %r", cleaned_query)
            return ()

        self._semantic_error = None
        return tuple(_format_taxonomy_path(result.path) for result in results)

    def combine_matches(
        self,
        query: str,
        *,
        semantic_matches: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return self._choices

        seen: set[str] = set()
        combined: list[str] = []
        for choice in (*self.exact_matches(cleaned_query), *semantic_matches):
            if choice in seen:
                continue
            seen.add(choice)
            combined.append(choice)
        return tuple(combined)


@lru_cache(maxsize=1)
def taxonomy_selector_searcher() -> TaxonomySearcher:
    return TaxonomySearcher()
