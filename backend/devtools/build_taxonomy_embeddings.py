from __future__ import annotations

import argparse
import hashlib
import json
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from rich import print as rprint

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_BATCH_SIZE = 64
ARTIFACT_SCHEMA_VERSION = 1

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "models" / "taxonomy.en-US.txt"
DEFAULT_OUTPUT_PATH = BACKEND_ROOT / "src" / "receipts_ai" / "models" / "taxonomy_embeddings.json"


class TaxonomyEmbedder(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object: ...


@dataclass(frozen=True)
class TaxonomyPath:
    parts: tuple[str, ...]

    @property
    def path(self) -> str:
        return " > ".join(self.parts)

    @property
    def embedding_text(self) -> str:
        return f"Google product category: {self.path}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a JSON embedding artifact for Google product taxonomy leaf paths.",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help=f"Source taxonomy text file. Defaults to {DEFAULT_TAXONOMY_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON artifact path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformers model name. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    return parser


def load_taxonomy_leaf_paths(taxonomy_path: Path) -> list[TaxonomyPath]:
    taxonomy_text = taxonomy_path.read_text(encoding="utf-8")
    all_paths = _taxonomy_paths_from_text(taxonomy_text)
    path_set = {path.parts for path in all_paths}
    leaf_paths = [
        path
        for path in all_paths
        if not any(
            len(candidate) > len(path.parts) and candidate[: len(path.parts)] == path.parts
            for candidate in path_set
        )
    ]
    if not leaf_paths:
        raise RuntimeError(f"{taxonomy_path} did not contain taxonomy leaf paths")
    return leaf_paths


def build_taxonomy_embedding_payload(
    *,
    taxonomy_path: Path,
    model_name: str,
    embedder: TaxonomyEmbedder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    taxonomy_bytes = taxonomy_path.read_bytes()
    leaf_paths = load_taxonomy_leaf_paths(taxonomy_path)
    embedding_texts = [path.embedding_text for path in leaf_paths]
    embeddings = embedder.encode(
        embedding_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = _vectors_as_lists(embeddings)
    if len(vectors) != len(leaf_paths):
        raise RuntimeError(
            f"embedding count mismatch: got {len(vectors)} embeddings for {len(leaf_paths)} paths"
        )

    dimension = len(vectors[0]) if vectors else 0
    if dimension == 0:
        raise RuntimeError("embedding model returned empty vectors")
    if any(len(vector) != dimension for vector in vectors):
        raise RuntimeError("embedding model returned vectors with inconsistent dimensions")

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "taxonomy_source": str(taxonomy_path),
        "taxonomy_sha256": hashlib.sha256(taxonomy_bytes).hexdigest(),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "embedding_normalized": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "entry_count": len(leaf_paths),
        "entries": [
            {
                "path": path.path,
                "parts": list(path.parts),
                "embedding_text": path.embedding_text,
                "embedding": vector,
            }
            for path, vector in zip(leaf_paths, vectors, strict=True)
        ],
    }


def build_taxonomy_embeddings(
    *,
    taxonomy_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"taxonomy file does not exist: {taxonomy_path}")

    from sentence_transformers import SentenceTransformer

    rprint(f"[bold green]Loading embedding model:[/bold green] {model_name}")
    embedder = cast(TaxonomyEmbedder, cast(object, SentenceTransformer(model_name)))
    rprint(f"[bold green]Embedding taxonomy leaves from:[/bold green] {taxonomy_path}")
    payload = build_taxonomy_embedding_payload(
        taxonomy_path=taxonomy_path,
        model_name=model_name,
        embedder=embedder,
        batch_size=batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rprint(
        "[bold green]Wrote "
        f"{payload['entry_count']} taxonomy embeddings to {output_path}[/bold green]"
    )


def _taxonomy_paths_from_text(taxonomy_text: str) -> list[TaxonomyPath]:
    paths: list[TaxonomyPath] = []
    seen: set[tuple[str, ...]] = set()
    for line in taxonomy_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        parts = tuple(part.strip() for part in stripped_line.split(">") if part.strip())
        if not parts or parts in seen:
            continue
        seen.add(parts)
        paths.append(TaxonomyPath(parts=parts))
    return paths


def _vectors_as_lists(embeddings: object) -> list[list[float]]:
    tolist = getattr(embeddings, "tolist", None)
    if callable(tolist):
        embeddings = tolist()
    if not isinstance(embeddings, list):
        raise RuntimeError("embedding model returned an unsupported vector container")

    vectors: list[list[float]] = []
    for embedding in cast(list[object], embeddings):
        if not isinstance(embedding, list):
            raise RuntimeError("embedding model returned a non-list vector")
        vector: list[float] = []
        for value in cast(list[object], embedding):
            if not isinstance(value, numbers.Real):
                raise RuntimeError("embedding model returned a non-numeric vector value")
            vector.append(float(value))
        vectors.append(vector)
    return vectors


def main() -> int:
    args = build_parser().parse_args()
    try:
        build_taxonomy_embeddings(
            taxonomy_path=args.taxonomy_path,
            output_path=args.output,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        rprint(f"[bold red]{error}[/bold red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
