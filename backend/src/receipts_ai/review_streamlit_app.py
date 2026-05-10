from __future__ import annotations

# pyright: reportUnknownMemberType=false
import argparse
import shlex
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from receipts_ai.cache import SqliteCallCache
from receipts_ai.models.receipt_data_extraction import (
    ExtractedReceiptItem,
    ReceiptDataExtraction,
    ReceiptDataExtractionMetadata,
)
from receipts_ai.models.transaction import LineType
from receipts_ai.review_models import ReceiptReviewStatus
from receipts_ai.review_service import (
    compare_reviewed_receipt_to_pipeline,
    import_receipt_for_review,
    save_review,
    write_training_jsonl,
)
from receipts_ai.review_store import DEFAULT_REVIEW_DB_PATH, ReceiptReviewStore


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

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        _render_receipt_source(source.image_path, source.sha256_hex)
        _render_extraction_metadata(store, selected_sha)
    with right:
        _render_review_form(
            store,
            receipt_sha256_hex=selected_sha,
            receipt_data=receipt_data,
            source_extraction_id=source_extraction_id,
            status=review.status if review is not None else ReceiptReviewStatus.draft,
            notes=review.notes if review is not None else None,
        )
        st.divider()
        _render_comparison_tools(store, selected_sha, Path(source.image_path))


def _render_sidebar(store: ReceiptReviewStore, db_path: Path) -> None:
    st.sidebar.header("Workspace")
    st.sidebar.caption(str(db_path))

    with st.sidebar.form("import_receipt"):
        receipt_path_text = st.text_input("Receipt path")
        pipeline = st.selectbox("Baseline pipeline", ["azure", "visionkit_ollama"], index=0)
        cache_file_text = st.text_input("Pipeline cache DB", value="")
        st.caption(
            "Optional. Use the same SQLite cache path you pass to ingestion with --cache-file."
        )
        force = st.checkbox("Force new extraction", value=False)
        submitted = st.form_submit_button("Import")
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
    sources = store.sources()
    if not sources:
        return None
    options = [source.sha256_hex for source in sources]
    labels = {
        source.sha256_hex: f"{Path(source.image_path).name}  {source.sha256_hex[:12]}"
        for source in sources
    }
    current = st.session_state.get("selected_receipt_sha")
    index = options.index(current) if isinstance(current, str) and current in options else 0
    selected = st.sidebar.selectbox(
        "Receipts",
        options,
        index=index,
        format_func=lambda value: labels[value],
    )
    st.session_state["selected_receipt_sha"] = selected
    return selected


def _render_receipt_source(image_path_text: str, sha256_hex: str) -> None:
    image_path = Path(image_path_text)
    st.subheader("Image")
    if image_path.exists() and image_path.suffix.casefold() != ".pdf":
        st.image(str(image_path), use_container_width=True)
    else:
        st.code(str(image_path))
    st.caption(sha256_hex)


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
