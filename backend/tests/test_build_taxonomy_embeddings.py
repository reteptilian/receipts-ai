from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "devtools" / "build_taxonomy_embeddings.py"
SPEC = importlib.util.spec_from_file_location("build_taxonomy_embeddings", SCRIPT_PATH)
assert SPEC is not None
build_taxonomy_embeddings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = build_taxonomy_embeddings
SPEC.loader.exec_module(build_taxonomy_embeddings)

build_taxonomy_embedding_payload = build_taxonomy_embeddings.build_taxonomy_embedding_payload
load_taxonomy_leaf_paths = build_taxonomy_embeddings.load_taxonomy_leaf_paths


class FakeEmbedder:
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        assert batch_size == 2
        assert normalize_embeddings is True
        assert convert_to_numpy is True
        assert show_progress_bar is True
        return [[float(index), float(index + 1)] for index, _ in enumerate(sentences)]


def test_load_taxonomy_leaf_paths_keeps_only_terminal_paths(tmp_path: Path):
    taxonomy_path = tmp_path / "taxonomy.en-US.txt"
    taxonomy_path.write_text(
        "\n".join(
            (
                "# Comment",
                "Electronics",
                "Electronics > Audio",
                "Electronics > Audio > Headphones",
                "Electronics > Video",
                "Food, Beverages & Tobacco > Food Items > Bakery",
                "Food, Beverages & Tobacco > Food Items > Bakery > Crackers",
            )
        ),
        encoding="utf-8",
    )

    leaf_paths = load_taxonomy_leaf_paths(taxonomy_path)

    assert [path.path for path in leaf_paths] == [
        "Electronics > Audio > Headphones",
        "Electronics > Video",
        "Food, Beverages & Tobacco > Food Items > Bakery > Crackers",
    ]


def test_build_taxonomy_embedding_payload_includes_metadata_and_vectors(tmp_path: Path):
    taxonomy_path = tmp_path / "taxonomy.en-US.txt"
    taxonomy_path.write_text(
        "\n".join(
            (
                "Electronics",
                "Electronics > Audio",
                "Electronics > Audio > Headphones",
                "Electronics > Video",
            )
        ),
        encoding="utf-8",
    )

    payload = build_taxonomy_embedding_payload(
        taxonomy_path=taxonomy_path,
        model_name="test-model",
        embedder=FakeEmbedder(),
        batch_size=2,
    )

    assert payload["schema_version"] == 1
    assert payload["taxonomy_source"] == str(taxonomy_path)
    assert payload["embedding_model"] == "test-model"
    assert payload["embedding_dimension"] == 2
    assert payload["embedding_normalized"] is True
    assert payload["entry_count"] == 2
    assert isinstance(payload["taxonomy_sha256"], str)
    assert isinstance(payload["generated_at"], str)
    assert payload["entries"] == [
        {
            "path": "Electronics > Audio > Headphones",
            "parts": ["Electronics", "Audio", "Headphones"],
            "embedding_text": "Google product category: Electronics > Audio > Headphones",
            "embedding": [0.0, 1.0],
        },
        {
            "path": "Electronics > Video",
            "parts": ["Electronics", "Video"],
            "embedding_text": "Google product category: Electronics > Video",
            "embedding": [1.0, 2.0],
        },
    ]
