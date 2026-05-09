from __future__ import annotations

from collections.abc import Iterable

from receipts_ai.models.transaction import ReceiptItem, Transaction

MAX_TAXONOMY_LEVELS = 9
TAXONOMY_DELIMITER = " > "


def flatten_taxonomy_parts(parts: Iterable[str | None]) -> str | None:
    cleaned_parts = [part.strip() for part in parts if part is not None and part.strip()]
    if not cleaned_parts:
        return None
    return TAXONOMY_DELIMITER.join(cleaned_parts)


def split_taxonomy_path(
    taxonomy: str | None,
    *,
    levels: int = MAX_TAXONOMY_LEVELS,
) -> tuple[str | None, ...]:
    if taxonomy is None:
        return (None,) * levels

    cleaned_parts = tuple(part.strip() for part in taxonomy.split(">") if part.strip())
    if len(cleaned_parts) >= levels:
        return cleaned_parts[:levels]
    return cleaned_parts + (None,) * (levels - len(cleaned_parts))


def effective_receipt_item_taxonomy(item: ReceiptItem) -> str | None:
    overrides = item.user_overrides
    if overrides is not None and overrides.taxonomy is not None:
        return overrides.taxonomy or None
    return item.taxonomy


def effective_transaction_taxonomy(transaction: Transaction) -> str | None:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.taxonomy is not None:
        return overrides.taxonomy or None
    return transaction.taxonomy
