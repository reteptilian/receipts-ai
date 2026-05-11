from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
import argparse
import shlex
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from receipts_ai.cache import SqliteCallCache
from receipts_ai.ingest_receipts import _visionkit_ocr_image_path
from receipts_ai.models.receipt_data_extraction import (
    ExtractedReceiptItem,
    ReceiptDataExtraction,
    ReceiptDataExtractionMetadata,
)
from receipts_ai.models.transaction import LineType
from receipts_ai.review_models import (
    ReceiptExtractionRecord,
    ReceiptReviewRecord,
    ReceiptReviewStatus,
    ReceiptSourceRecord,
)
from receipts_ai.review_service import (
    compare_reviewed_receipt_to_pipeline,
    import_receipt_for_review,
    save_review,
    write_training_jsonl,
)
from receipts_ai.review_store import DEFAULT_REVIEW_DB_PATH, ReceiptReviewStore


@dataclass(frozen=True)
class _ReceiptQueueEntry:
    sha256_hex: str
    label: str
    transaction_date: date | None
    reviewed: bool


def main() -> None:
    args = _parse_args()
    store = ReceiptReviewStore(args.db)

    st.set_page_config(page_title="Receipt Review", layout="wide")
    st.title("Receipt Review")

    _render_sidebar(store, args.db)
    selected_sha = _selected_receipt_sha(store)
    if selected_sha is None:
        st.info("Import a receipt to begin.")
        return

    source = store.source(selected_sha)
    if source is None:
        st.error(f"Receipt source {selected_sha} no longer exists.")
        return

    review = store.review(selected_sha)
    extraction = store.latest_extraction(selected_sha, pipeline="azure") or store.latest_extraction(
        selected_sha
    )
    if review is not None:
        receipt_data = review.corrected_receipt_data
        source_extraction_id = review.source_extraction_id
    elif extraction is not None:
        receipt_data = extraction.receipt_data
        source_extraction_id = extraction.id
    else:
        st.error("This receipt has no extraction yet.")
        return

    image_column, review_column, context_column = st.columns([0.75, 1.35, 0.9], gap="large")
    with image_column:
        _render_receipt_source(source.image_path, source.sha256_hex)
    with review_column:
        _render_review_form(
            store,
            receipt_sha256_hex=selected_sha,
            receipt_data=receipt_data,
            source_extraction_id=source_extraction_id,
            status=review.status if review is not None else ReceiptReviewStatus.draft,
            notes=review.notes if review is not None else None,
        )
    with context_column:
        _render_extraction_metadata(store, selected_sha)
        st.divider()
        _render_comparison_tools(store, selected_sha, Path(source.image_path))


def _render_sidebar(store: ReceiptReviewStore, db_path: Path) -> None:
    st.sidebar.header("Workspace")
    st.sidebar.caption(str(db_path))

    import_form = st.sidebar.form("import_receipt")
    with import_form:
        receipt_path_text = import_form.text_input("Receipt path")
        pipeline = import_form.selectbox(
            "Baseline pipeline", ["azure", "visionkit_ollama"], index=0
        )
        cache_file_text = import_form.text_input("Pipeline cache DB", value="")
        import_form.caption(
            "Optional. Use the same SQLite cache path you pass to ingestion with --cache-file."
        )
        force = import_form.checkbox("Force new extraction", value=False)
        submitted = import_form.form_submit_button("Import")
    if submitted:
        try:
            receipt_path = _path_from_text(receipt_path_text)
            cache = SqliteCallCache(_path_from_text(cache_file_text)) if cache_file_text else None
            extraction = import_receipt_for_review(
                receipt_path,
                store=store,
                pipeline=pipeline,
                cache=cache,
                force=force,
            )
            st.session_state["selected_receipt_sha"] = extraction.receipt_sha256_hex
            st.sidebar.success(f"Stored extraction {extraction.id}")
            st.rerun()
        except Exception as error:
            st.sidebar.error(str(error))

    st.sidebar.divider()
    export_path_text = st.sidebar.text_input(
        "Training JSONL",
        value="receipt-training.jsonl",
    )
    if st.sidebar.button("Export reviewed examples"):
        try:
            output_path = _path_from_text(export_path_text)
            with output_path.open("w", encoding="utf-8") as file:
                write_training_jsonl(store, file)
            st.sidebar.success(f"Wrote {output_path}")
        except Exception as error:
            st.sidebar.error(str(error))


def _selected_receipt_sha(store: ReceiptReviewStore) -> str | None:
    entries = _receipt_queue_entries(store)
    if not entries:
        return None

    st.sidebar.subheader("Receipts")
    _render_receipt_queue_progress(entries)

    current = st.session_state.get("selected_receipt_sha")
    next_unreviewed_sha = _next_unreviewed_sha(
        entries,
        current if isinstance(current, str) else None,
    )
    if st.sidebar.button(
        "Next unreviewed",
        disabled=next_unreviewed_sha is None,
        use_container_width=True,
    ):
        st.session_state["selected_receipt_sha"] = next_unreviewed_sha
        st.rerun()

    options = [entry.sha256_hex for entry in entries]
    labels = {entry.sha256_hex: entry.label for entry in entries}
    index = options.index(current) if isinstance(current, str) and current in options else 0
    with st.sidebar.container(height=420):
        selected = st.radio(
            "Receipts",
            options,
            index=index,
            format_func=lambda value: labels[value],
            label_visibility="collapsed",
        )
    st.session_state["selected_receipt_sha"] = selected
    return selected


def _receipt_queue_entries(store: ReceiptReviewStore) -> list[_ReceiptQueueEntry]:
    reviews_by_sha = {review.receipt_sha256_hex: review for review in store.reviews()}
    entries = [
        _receipt_queue_entry(
            source,
            review=reviews_by_sha.get(source.sha256_hex),
            extraction=store.latest_extraction(source.sha256_hex, pipeline="azure")
            or store.latest_extraction(source.sha256_hex),
        )
        for source in store.sources()
    ]
    return sorted(
        entries,
        key=lambda entry: (
            entry.transaction_date is None,
            entry.transaction_date or date.max,
            entry.label.casefold(),
        ),
    )


def _receipt_queue_entry(
    source: ReceiptSourceRecord,
    *,
    review: ReceiptReviewRecord | None,
    extraction: ReceiptExtractionRecord | None,
) -> _ReceiptQueueEntry:
    receipt_data = (
        review.corrected_receipt_data
        if review is not None
        else extraction.receipt_data
        if extraction is not None
        else None
    )
    transaction_date = receipt_data.transaction_date if receipt_data is not None else None
    merchant_name = (
        receipt_data.merchant_name if receipt_data is not None else Path(source.image_path).stem
    )
    reviewed = review is not None and review.status == ReceiptReviewStatus.reviewed
    date_text = transaction_date.isoformat() if transaction_date is not None else "no date"
    merchant_text = _truncate_label_part(
        merchant_name or Path(source.image_path).stem,
        max_length=28,
    )
    marker_text = "" if reviewed else "* "
    return _ReceiptQueueEntry(
        sha256_hex=source.sha256_hex,
        label=f"{marker_text}{date_text}  {merchant_text}  {source.sha256_hex[:12]}",
        transaction_date=transaction_date,
        reviewed=reviewed,
    )


def _truncate_label_part(value: str, *, max_length: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1]}..."


def _render_receipt_queue_progress(entries: list[_ReceiptQueueEntry]) -> None:
    reviewed_count = sum(1 for entry in entries if entry.reviewed)
    st.sidebar.caption(f"Reviewed {reviewed_count} / {len(entries)}")


def _next_unreviewed_sha(
    entries: list[_ReceiptQueueEntry],
    current_sha256_hex: str | None,
) -> str | None:
    unreviewed = [entry.sha256_hex for entry in entries if not entry.reviewed]
    if not unreviewed:
        return None
    if current_sha256_hex is None:
        return unreviewed[0]

    options = [entry.sha256_hex for entry in entries]
    if current_sha256_hex not in options:
        return unreviewed[0]

    current_index = options.index(current_sha256_hex)
    ordered_candidates = entries[current_index + 1 :] + entries[:current_index]
    for entry in ordered_candidates:
        if not entry.reviewed:
            return entry.sha256_hex
    return None


def _render_receipt_source(image_path_text: str, sha256_hex: str) -> None:
    image_path = Path(image_path_text)
    st.subheader("Receipt Image")
    with st.container(height=780, border=True):
        preview = _receipt_preview_image(image_path)
        if preview is not None:
            st.image(preview, use_container_width=True)
        else:
            st.code(str(image_path))
    st.caption(sha256_hex)


def _receipt_preview_image(image_path: Path) -> bytes | str | None:
    if not image_path.exists():
        return None
    if image_path.suffix.casefold() != ".pdf":
        return str(image_path)
    with _visionkit_ocr_image_path(image_path) as preview_path:
        return preview_path.read_bytes()


def _render_extraction_metadata(store: ReceiptReviewStore, receipt_sha256_hex: str) -> None:
    st.subheader("Runs")
    rows: list[dict[str, object | None]] = []
    for pipeline in ["azure", "visionkit_ollama"]:
        extraction = store.latest_extraction(receipt_sha256_hex, pipeline=pipeline)
        if extraction is None:
            continue
        rows.append(
            {
                "id": extraction.id,
                "pipeline": extraction.pipeline,
                "model": extraction.model,
                "created_at": extraction.created_at.isoformat(),
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_review_form(
    store: ReceiptReviewStore,
    *,
    receipt_sha256_hex: str,
    receipt_data: ReceiptDataExtraction,
    source_extraction_id: int | None,
    status: ReceiptReviewStatus,
    notes: str | None,
) -> None:
    st.subheader("Corrected Data")
    status_label = "reviewed" if status == ReceiptReviewStatus.reviewed else "draft"
    st.caption(f"Current status: {status_label}")

    merchant_name = st.text_input("Merchant", value=receipt_data.merchant_name)
    transaction_date = st.date_input("Transaction date", value=receipt_data.transaction_date)
    currency = st.text_input("Currency", value=receipt_data.currency)
    receipt_number = st.text_input("Receipt number", value=receipt_data.receipt_number or "")

    amount_columns = st.columns(3)
    with amount_columns[0]:
        subtotal = st.text_input("Subtotal", value=receipt_data.subtotal or "")
    with amount_columns[1]:
        total_tax = st.text_input("Tax", value=receipt_data.total_tax or "")
    with amount_columns[2]:
        total = st.text_input("Total", value=receipt_data.total)

    item_rows = _item_rows(receipt_data)
    edited_items = st.data_editor(
        item_rows,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "line_type": st.column_config.SelectboxColumn(
                "line_type",
                options=[line_type.value for line_type in LineType],
            )
        },
        key=f"items_{receipt_sha256_hex}",
    )
    _render_sanity_checks(
        subtotal=subtotal,
        total_tax=total_tax,
        total=total,
        items=edited_items,
    )
    notes_value = st.text_area("Notes", value=notes or "")

    draft_col, reviewed_col = st.columns(2)
    with draft_col:
        save_draft = st.button("Save draft", use_container_width=True)
    with reviewed_col:
        mark_reviewed = st.button("Mark reviewed", type="primary", use_container_width=True)

    if save_draft or mark_reviewed:
        try:
            corrected = _receipt_data_from_form(
                merchant_name=merchant_name,
                transaction_date=transaction_date,
                currency=currency,
                receipt_number=receipt_number,
                subtotal=subtotal,
                total_tax=total_tax,
                total=total,
                items=edited_items,
                original=receipt_data,
            )
            save_review(
                store,
                receipt_sha256_hex=receipt_sha256_hex,
                corrected_receipt_data=corrected,
                status=ReceiptReviewStatus.reviewed if mark_reviewed else ReceiptReviewStatus.draft,
                source_extraction_id=source_extraction_id,
                notes=notes_value or None,
            )
            st.success("Saved")
            st.rerun()
        except (ValidationError, ValueError) as error:
            st.error(str(error))


def _render_comparison_tools(
    store: ReceiptReviewStore,
    receipt_sha256_hex: str,
    receipt_path: Path,
) -> None:
    st.subheader("Candidate Comparison")
    pipeline = st.selectbox("Candidate pipeline", ["visionkit_ollama"], index=0)
    cache_file_text = st.text_input("Candidate pipeline cache DB", value="", key="comparison_cache")
    st.caption("Optional. Reuses cached OCR/model calls from the same --cache-file database.")
    if st.button("Run comparison"):
        try:
            cache = SqliteCallCache(_path_from_text(cache_file_text)) if cache_file_text else None
            result = compare_reviewed_receipt_to_pipeline(
                receipt_path,
                store=store,
                candidate_pipeline=pipeline,
                cache=cache,
            )
            st.success(f"Score {result.score:.1%}, mismatches {result.mismatch_count}")
            st.rerun()
        except Exception as error:
            st.error(str(error))

    comparisons = store.latest_comparisons(receipt_sha256_hex)
    if not comparisons:
        return
    latest = comparisons[0]
    st.metric("Latest score", f"{latest.score:.1%}", f"{latest.mismatch_count} mismatches")
    diff_rows = [
        {
            "path": field.path,
            "expected": field.expected,
            "actual": field.actual,
            "similarity": field.similarity,
            "matches": field.matches,
        }
        for field in latest.fields
        if not field.matches
    ]
    if diff_rows:
        st.dataframe(diff_rows, hide_index=True, use_container_width=True)


def _item_rows(receipt_data: ReceiptDataExtraction) -> list[dict[str, object | None]]:
    return [
        {
            "description": item.description,
            "quantity": item.quantity,
            "unit_price": item.unit_price,
            "amount": item.amount,
            "discount_amount": item.discount_amount,
            "discount_description": item.discount_description,
            "line_type": item.line_type.value,
        }
        for item in receipt_data.items
    ]


def _receipt_data_from_form(
    *,
    merchant_name: str,
    transaction_date: date,
    currency: str,
    receipt_number: str,
    subtotal: str,
    total_tax: str,
    total: str,
    items: Any,
    original: ReceiptDataExtraction,
) -> ReceiptDataExtraction:
    return ReceiptDataExtraction(
        merchant_name=merchant_name,
        transaction_date=transaction_date,
        currency=currency.upper(),
        receipt_number=receipt_number or None,
        subtotal=subtotal or None,
        total_tax=total_tax or None,
        total=total,
        items=_items_from_editor(items),
        extraction=ReceiptDataExtractionMetadata(
            pipeline=original.extraction.pipeline,
            model=original.extraction.model,
            confidence=original.extraction.confidence,
            raw_text=original.extraction.raw_text,
        ),
    )


def _items_from_editor(items: Any) -> list[ExtractedReceiptItem]:
    records = _editor_records(items)
    extracted_items: list[ExtractedReceiptItem] = []
    for record in records:
        description = _clean_string(record.get("description"))
        amount = _clean_string(record.get("amount"))
        if not description and not amount:
            continue
        if not description or not amount:
            raise ValueError("Each item row needs both description and amount.")
        extracted_items.append(
            ExtractedReceiptItem(
                description=description,
                quantity=_float_or_none(record.get("quantity")),
                unit_price=_clean_string(record.get("unit_price")) or None,
                amount=amount,
                discount_amount=_clean_string(record.get("discount_amount")) or None,
                discount_description=_clean_string(record.get("discount_description")) or None,
                line_type=LineType(_clean_string(record.get("line_type")) or LineType.item.value),
            )
        )
    if not extracted_items:
        raise ValueError("Receipt needs at least one item.")
    return extracted_items


def _editor_records(items: Any) -> list[dict[str, Any]]:
    if isinstance(items, pd.DataFrame):
        return cast(list[dict[str, Any]], items.to_dict(orient="records"))
    if isinstance(items, list):
        return cast(list[dict[str, Any]], items)
    raise ValueError("Unexpected item editor value.")


def _render_sanity_checks(
    *,
    subtotal: str,
    total_tax: str,
    total: str,
    items: Any,
) -> None:
    failures = _sanity_check_failures(
        subtotal=subtotal,
        total_tax=total_tax,
        total=total,
        items=items,
    )
    st.subheader("Sanity Checks")
    if not failures:
        st.success("All checks pass.")
        return

    st.warning(f"{len(failures)} failing check(s)")
    st.dataframe(failures, hide_index=True, use_container_width=True)


def _sanity_check_failures(
    *,
    subtotal: str | None,
    total_tax: str | None,
    total: str | None,
    items: Any,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    parsed_subtotal = _money_or_failure("subtotal", subtotal, failures)
    parsed_tax = _money_or_failure("tax", total_tax, failures, default=Decimal("0"))
    parsed_total = _money_or_failure("total", total, failures)

    if parsed_subtotal is not None and parsed_tax is not None and parsed_total is not None:
        expected_total = parsed_subtotal + parsed_tax
        if _money_mismatch(parsed_total, expected_total):
            failures.append(
                {
                    "check": "total = subtotal + tax",
                    "expected": _money_text(expected_total),
                    "actual": _money_text(parsed_total),
                }
            )

    item_net_total = _item_net_total_or_failure(items, failures)
    if parsed_subtotal is not None and item_net_total is not None:
        if _money_mismatch(parsed_subtotal, item_net_total):
            failures.append(
                {
                    "check": "subtotal = sum(item net amounts)",
                    "expected": _money_text(item_net_total),
                    "actual": _money_text(parsed_subtotal),
                }
            )

    tax_line_total = _tax_line_total_or_failure(items, failures)
    if parsed_tax is not None and tax_line_total is not None:
        if _money_mismatch(parsed_tax, tax_line_total):
            failures.append(
                {
                    "check": "tax = sum(tax line amounts)",
                    "expected": _money_text(tax_line_total),
                    "actual": _money_text(parsed_tax),
                }
            )

    return failures


def _money_or_failure(
    field: str,
    value: object | None,
    failures: list[dict[str, str]],
    *,
    default: Decimal | None = None,
) -> Decimal | None:
    text = _clean_string(value)
    if text is None:
        return default
    try:
        return Decimal(text)
    except InvalidOperation:
        failures.append(
            {
                "check": f"{field} is a valid amount",
                "expected": "decimal amount",
                "actual": text,
            }
        )
        return None


def _item_net_total_or_failure(
    items: Any,
    failures: list[dict[str, str]],
) -> Decimal | None:
    try:
        records = _editor_records(items)
    except ValueError as error:
        failures.append(
            {
                "check": "items can be read",
                "expected": "editable item rows",
                "actual": str(error),
            }
        )
        return None

    total = Decimal("0")
    for index, record in enumerate(records):
        if (
            _clean_string(record.get("description")) is None
            and _clean_string(record.get("amount")) is None
            and _clean_string(record.get("discount_amount")) is None
        ):
            continue

        line_type = _clean_string(record.get("line_type")) or LineType.item.value
        if line_type != LineType.item.value:
            continue

        amount = _money_or_failure(f"items[{index}].amount", record.get("amount"), failures)
        discount = _money_or_failure(
            f"items[{index}].discount_amount",
            record.get("discount_amount"),
            failures,
            default=Decimal("0"),
        )
        if amount is None or discount is None:
            return None
        if discount > 0:
            failures.append(
                {
                    "check": f"items[{index}].discount_amount is negative",
                    "expected": "negative discount amount",
                    "actual": _money_text(discount),
                }
            )
        total += amount + discount
    return total


def _tax_line_total_or_failure(
    items: Any,
    failures: list[dict[str, str]],
) -> Decimal | None:
    try:
        records = _editor_records(items)
    except ValueError as error:
        failures.append(
            {
                "check": "items can be read",
                "expected": "editable item rows",
                "actual": str(error),
            }
        )
        return None

    total = Decimal("0")
    for index, record in enumerate(records):
        if (
            _clean_string(record.get("description")) is None
            and _clean_string(record.get("amount")) is None
            and _clean_string(record.get("discount_amount")) is None
        ):
            continue

        line_type = _clean_string(record.get("line_type")) or LineType.item.value
        if line_type != LineType.tax.value:
            continue

        amount = _money_or_failure(f"items[{index}].amount", record.get("amount"), failures)
        if amount is None:
            return None
        total += amount
    return total


def _money_mismatch(actual: Decimal, expected: Decimal) -> bool:
    return actual.quantize(Decimal("0.01")) != expected.quantize(Decimal("0.01"))


def _money_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def _clean_string(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: object | None) -> float | None:
    text = _clean_string(value)
    if text is None:
        return None
    return float(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_REVIEW_DB_PATH)
    return parser.parse_args()


def _path_from_text(value: str) -> Path:
    text = value.strip()
    if not text:
        raise ValueError("Path must not be empty.")
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = []
    if len(parts) == 1:
        text = parts[0]
    return Path(text).expanduser()


if __name__ == "__main__":
    main()
