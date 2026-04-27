import subprocess
import sys
from pathlib import Path

from rich import print as rprint

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
SCHEMA_PATH = REPO_ROOT / "models" / "transaction.json"
OUTPUT_PATH = BACKEND_ROOT / "src" / "receipts_ai" / "models" / "transaction.py"


def main() -> int:
    if not SCHEMA_PATH.exists():
        rprint(f"[bold red]Schema not found:[/bold red] {SCHEMA_PATH}")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "datamodel_code_generator",
        "--input",
        str(SCHEMA_PATH),
        "--input-file-type",
        "jsonschema",
        "--output",
        str(OUTPUT_PATH),
        "--allow-population-by-field-name",
        "--disable-timestamp",
        "--formatters",
        "ruff-format",
        "ruff-check",
        "--output-model-type",
        "pydantic_v2.BaseModel",
        "--set-default-enum-member",
        "--target-python-version",
        "3.11",
        "--collapse-root-models",
        "--snake-case-field",
        "--use-annotated",
        "--use-default-kwarg",
        "--use-schema-description",
        "--use-standard-collections",
        "--use-type-alias",
        "--use-union-operator",
    ]

    rprint(f"[bold green]Generating Pydantic models from {SCHEMA_PATH}[/bold green]")
    subprocess.run(cmd, check=True)
    rprint(f"[bold green]Wrote {OUTPUT_PATH}[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
