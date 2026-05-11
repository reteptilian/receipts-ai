from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, NoReturn, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from receipts_ai.categorization import (
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_PROMPT_LOG_ENV_VARS,
    UrlLibOllamaClient,
)
from receipts_ai.config import add_config_file_argument, configure_config_file
from receipts_ai.ingest_receipts import (
    DEFAULT_RECEIPT_OLLAMA_MODEL,
    _ollama_timeout_seconds,
    _ollama_url,
    _receipt_data_from_ollama_response,
    _receipt_ollama_model,
    _receipt_ollama_output_schema,
    _receipt_ollama_think,
    _visionkit_ollama_receipt_prompt,
    _visionkit_text_lines,
    _visionkit_text_observations,
)

DEFAULT_PROMPT = """
Extract merchant, items, and total.
Address: 123 Apple Way, Cupertino
Item: M4 Mac Mini - $599
Total: $599.00
""".strip()
SCHEMA_PRESETS = ("full", "simple", "total")


class _SimpleReceiptItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    description: Annotated[str, Field(min_length=1)]
    amount: str


class _SimpleReceiptDataExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    merchant_name: Annotated[str, Field(alias="merchantName", min_length=1)]
    transaction_date: Annotated[date | None, Field(alias="transactionDate")] = None
    total: str
    items: Annotated[list[_SimpleReceiptItem], Field(min_length=1)]


class _ReceiptTotalExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    merchant_name: Annotated[str, Field(alias="merchantName", min_length=1)]
    transaction_date: Annotated[date | None, Field(alias="transactionDate")] = None
    total: str
    currency: Annotated[str | None, Field(pattern="^[A-Z]{3}$")] = None


SchemaPreset = Literal["full", "simple", "total"]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    prompt_input = _read_prompt_input(args)
    prompt = (
        prompt_input
        if args.raw_prompt
        else _visionkit_ollama_receipt_prompt(prompt_input.splitlines())
    )
    schema = _output_schema(args)
    if args.schema_in_prompt:
        prompt = _prompt_with_schema(prompt, schema)
    options = _ollama_options(args)

    if args.show_prompt:
        print(prompt)
        return 0
    if args.show_schema:
        print(json.dumps(schema, indent=2, sort_keys=True))
        return 0

    client = UrlLibOllamaClient(
        url=args.host,
        model=args.model,
        timeout_seconds=args.timeout,
        prompt_log_path=args.prompt_log,
        think=args.think,
    )

    completion = client._generate(
        prompt,
        options=options,
        output_format=None if args.no_schema else schema,
    )
    response = completion.response
    thinking = completion.raw_response.get("thinking")

    if args.response_file is not None:
        args.response_file.parent.mkdir(parents=True, exist_ok=True)
        args.response_file.write_text(response, encoding="utf-8")
    if args.thinking_file is not None:
        args.thinking_file.parent.mkdir(parents=True, exist_ok=True)
        args.thinking_file.write_text(_thinking_text(thinking), encoding="utf-8")
    if args.show_thinking:
        _print_thinking(thinking)

    if args.raw_response or args.no_schema:
        print(response)
        return 0

    return _print_structured_response(
        response,
        raw_text=prompt_input,
        validate=not args.no_validate,
        schema_preset=args.schema_preset if args.schema_file is None else None,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    early_parser = argparse.ArgumentParser(add_help=False)
    add_config_file_argument(early_parser)
    early_args, _ = early_parser.parse_known_args(argv)
    configure_config_file(early_args.config_file)

    parser = argparse.ArgumentParser(
        description=(
            "Call Ollama with the same constrained JSON schema used by the "
            "visionkit_ollama receipt pipeline."
        )
    )
    add_config_file_argument(parser)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument(
        "--prompt",
        help="Receipt OCR text or a complete prompt. Defaults to the example in this file.",
    )
    prompt_source.add_argument(
        "--prompt-file",
        type=Path,
        help="Read receipt OCR text or a complete prompt from this file.",
    )
    prompt_source.add_argument(
        "--stdin",
        action="store_true",
        help="Read receipt OCR text or a complete prompt from standard input.",
    )
    prompt_source.add_argument(
        "--receipt-image",
        type=Path,
        help="Run VisionKit OCR on this receipt image/PDF and use the grouped OCR lines.",
    )
    parser.add_argument(
        "--host",
        default=_ollama_url(),
        help="Ollama host or /api/generate URL. Default follows receipt pipeline config.",
    )
    parser.add_argument(
        "--model",
        default=_receipt_ollama_model(),
        help=f"Ollama model. Default: env/config value or {DEFAULT_RECEIPT_OLLAMA_MODEL}.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=_ollama_timeout_seconds(),
        help=f"Request timeout in seconds. Default: env/config value or {DEFAULT_OLLAMA_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=_receipt_ollama_think(),
        help="Enable or disable Ollama thinking. Default follows receipt pipeline config.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ollama temperature option. Default: %(default)s.",
    )
    parser.add_argument("--seed", type=int, help="Ollama seed option for repeatable tests.")
    parser.add_argument("--num-predict", type=int, help="Ollama num_predict option.")
    parser.add_argument("--top-k", type=int, help="Ollama top_k option.")
    parser.add_argument("--top-p", type=float, help="Ollama top_p option.")
    parser.add_argument("--repeat-penalty", type=float, help="Ollama repeat_penalty option.")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help=(
            "Additional Ollama option. Values are parsed as JSON when possible, "
            "for example --option num_ctx=8192 or --option stop='[\"\\n\\n\"]'."
        ),
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send the prompt text exactly as provided instead of wrapping it as OCR lines.",
    )
    parser.add_argument(
        "--schema-preset",
        choices=SCHEMA_PRESETS,
        default="full",
        help=(
            "Output schema to request. full matches the production pipeline; simple keeps "
            "merchantName, transactionDate, total, and item description/amount; total keeps "
            "only merchantName, transactionDate, total, and currency. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--schema-file",
        type=Path,
        help="Read a custom Ollama JSON schema from this file instead of using --schema-preset.",
    )
    parser.add_argument(
        "--schema-in-prompt",
        action="store_true",
        help=(
            "Append the selected schema to the prompt as instructions. Combine with "
            "--no-schema to avoid Ollama structured output while still showing the model "
            "the desired formal schema."
        ),
    )
    parser.add_argument(
        "--ocr-debug-image",
        type=Path,
        help="When using --receipt-image, write a PNG showing VisionKit OCR boxes.",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Do not request the receipt JSON schema. Useful for prompt debugging.",
    )
    parser.add_argument(
        "--raw-response",
        action="store_true",
        help="Print the model response string without parsing or pretty formatting.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Print Ollama's thinking field before the final response when the model returns one.",
    )
    parser.add_argument(
        "--thinking-file",
        type=Path,
        help="Write Ollama's thinking field here when the model returns one.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Pretty-print JSON without validating it as receipt extraction data.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the exact prompt that would be sent and exit.",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Print the receipt extraction JSON schema and exit.",
    )
    parser.add_argument(
        "--prompt-log",
        type=Path,
        help=(
            "Write a human-readable prompt log for this call. The shared client also "
            f"honors {', '.join(OLLAMA_PROMPT_LOG_ENV_VARS)}."
        ),
    )
    parser.add_argument("--response-file", type=Path, help="Also write the raw response here.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)
    if args.no_schema and args.schema_file is not None and not args.schema_in_prompt:
        parser.error("--schema-file with --no-schema requires --schema-in-prompt")
    return args


def _read_prompt_input(args: argparse.Namespace) -> str:
    if args.stdin:
        prompt = sys.stdin.read()
    elif args.receipt_image is not None:
        observations = _visionkit_text_observations(
            args.receipt_image,
            debug_image_path=args.ocr_debug_image,
        )
        prompt = "\n".join(_visionkit_text_lines(observations))
    elif args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT

    prompt = prompt.strip()
    if not prompt:
        _die("prompt must not be empty")
    return prompt


def _ollama_options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {"temperature": args.temperature}
    for option_name in ("seed", "num_predict", "top_k", "top_p", "repeat_penalty"):
        value = getattr(args, option_name)
        if value is not None:
            options[option_name] = value
    for option in args.option:
        key, separator, value = option.partition("=")
        if not separator or not key.strip():
            _die(f"--option must use KEY=JSON format: {option!r}")
        options[key.strip()] = _parse_jsonish_value(value)
    return options


def _output_schema(args: argparse.Namespace) -> dict[str, object]:
    if args.schema_file is not None:
        schema = json.loads(args.schema_file.read_text(encoding="utf-8"))
        if not isinstance(schema, dict):
            _die("--schema-file must contain a JSON object schema")
        return cast(dict[str, object], schema)
    return _schema_preset(args.schema_preset)


def _schema_preset(schema_preset: SchemaPreset) -> dict[str, object]:
    if schema_preset == "full":
        return _receipt_ollama_output_schema()
    if schema_preset == "simple":
        return _SimpleReceiptDataExtraction.model_json_schema(by_alias=True)
    if schema_preset == "total":
        return _ReceiptTotalExtraction.model_json_schema(by_alias=True)
    _die(f"Unknown schema preset: {schema_preset}")


def _prompt_with_schema(prompt: str, schema: dict[str, object]) -> str:
    schema_json = json.dumps(schema, indent=2, sort_keys=True)
    return (
        f"{prompt}\n\n"
        "Return only JSON matching this JSON Schema. Do not include markdown, "
        "comments, or explanation text.\n"
        f"{schema_json}"
    )


def _parse_jsonish_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _thinking_text(thinking: object) -> str:
    if thinking is None:
        return ""
    if isinstance(thinking, str):
        return thinking
    return json.dumps(thinking, indent=2, sort_keys=True)


def _print_thinking(thinking: object) -> None:
    text = _thinking_text(thinking).strip()
    if not text:
        print("Thinking: <not returned by Ollama>", file=sys.stderr)
        return
    print("Thinking:", file=sys.stderr)
    print(text, file=sys.stderr)
    print(file=sys.stderr)


def _print_structured_response(
    response: str,
    *,
    raw_text: str,
    validate: bool,
    schema_preset: SchemaPreset | None,
) -> int:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        print(response)
        print(f"\nERROR: response was not JSON: {error}", file=sys.stderr)
        return 1

    if not validate or schema_preset is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    try:
        formatted_response = _validate_response_payload(
            response,
            payload=payload,
            raw_text=raw_text,
            schema_preset=schema_preset,
        )
    except (ValidationError, ValueError) as error:
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(
            f"\nERROR: response did not validate as {schema_preset!r} receipt data: {error}",
            file=sys.stderr,
        )
        return 1

    print(formatted_response)
    return 0


def _validate_response_payload(
    response: str,
    *,
    payload: object,
    raw_text: str,
    schema_preset: SchemaPreset,
) -> str:
    if schema_preset == "full":
        receipt_data = _receipt_data_from_ollama_response(response, raw_text=raw_text)
        return receipt_data.model_dump_json(by_alias=True, indent=2)
    if schema_preset == "simple":
        receipt_data = TypeAdapter(_SimpleReceiptDataExtraction).validate_python(payload)
        return receipt_data.model_dump_json(by_alias=True, indent=2)
    if schema_preset == "total":
        receipt_data = TypeAdapter(_ReceiptTotalExtraction).validate_python(payload)
        return receipt_data.model_dump_json(by_alias=True, indent=2)
    _die(f"Unknown schema preset: {schema_preset}")


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


if __name__ == "__main__":
    raise SystemExit(main())
