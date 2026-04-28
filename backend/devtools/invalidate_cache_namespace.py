from __future__ import annotations

import argparse
from pathlib import Path

from rich import print as rprint

from receipts_ai.cache import JsonCallCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete all JSON cache entries for a namespace.",
    )
    parser.add_argument(
        "cache_file",
        type=Path,
        help="Path to the JSON cache file to update.",
    )
    parser.add_argument(
        "namespace",
        help="Cache namespace to invalidate, for example brave_search or ollama.",
    )
    parser.add_argument(
        "--allow-create",
        action="store_true",
        help="Create the cache file if it does not exist.",
    )
    return parser


def invalidate_cache_namespace(
    cache_file: Path,
    namespace: str,
    *,
    allow_create: bool = False,
) -> int:
    if not allow_create and not cache_file.exists():
        raise FileNotFoundError(f"cache file does not exist: {cache_file}")

    return JsonCallCache(cache_file).invalidate_namespace(namespace)


def main() -> int:
    args = build_parser().parse_args()

    try:
        deleted_count = invalidate_cache_namespace(
            args.cache_file,
            args.namespace,
            allow_create=args.allow_create,
        )
    except FileNotFoundError as error:
        rprint(f"[bold red]{error}[/bold red]")
        return 1
    except RuntimeError as error:
        rprint(f"[bold red]{error}[/bold red]")
        return 1

    noun = "entry" if deleted_count == 1 else "entries"
    rprint(
        f"[bold green]Deleted {deleted_count} cache {noun} from "
        f"{args.namespace!r} in {args.cache_file}[/bold green]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
