from __future__ import annotations

# pyright: reportUnusedFunction=false
import os
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import NamedTuple, cast

from pydantic import ValidationError
from receipts_ai.models.transaction import (
    CategoryAllocation,
    LineType,
    ReceiptItem,
    Transaction,
    UserCategoryAllocation,
)
from textual.widgets import Input

TransactionLoader = Callable[[], Sequence[Transaction]]

ReceiptItemParser = Callable[[str], object | None]
ReceiptItemFormatter = Callable[[ReceiptItem], str]


class TransactionTableColumn(NamedTuple):
    key: str
    label: str


class TransactionTableFlexColumn(NamedTuple):
    key: str
    min_width: int
    max_width: int
    weight: int


TRANSACTION_TABLE_COLUMNS = (
    TransactionTableColumn(key="date", label="Date"),
    TransactionTableColumn(key="payee", label="Payee"),
    TransactionTableColumn(key="description", label="Description"),
    TransactionTableColumn(key="category", label="Category"),
    TransactionTableColumn(key="ingestion_filename", label="Ingestion file"),
    TransactionTableColumn(key="receipt", label="Receipt?"),
    TransactionTableColumn(key="amount", label="Amount"),
)

TRANSACTION_TABLE_FIXED_WIDTHS = {
    "date": 10,
    "receipt": 8,
    "amount": 12,
}

TRANSACTION_TABLE_FLEX_COLUMNS = (
    TransactionTableFlexColumn(key="payee", min_width=16, max_width=40, weight=2),
    TransactionTableFlexColumn(key="description", min_width=20, max_width=80, weight=5),
    TransactionTableFlexColumn(key="category", min_width=16, max_width=40, weight=2),
    TransactionTableFlexColumn(key="ingestion_filename", min_width=14, max_width=30, weight=1),
)


@dataclass(frozen=True)
class ReceiptItemColumn:
    label: str
    field_name: str
    parser: ReceiptItemParser
    formatter: ReceiptItemFormatter
    direct_edit: bool = False


RECEIPT_ITEM_COLUMNS = (
    ReceiptItemColumn(
        "Description",
        "description",
        lambda value: _parse_required_text(value, "Description"),
        lambda item: cast(str, _effective_receipt_item_value(item, "description")),
    ),
    ReceiptItemColumn(
        "Quantity",
        "quantity",
        lambda value: _parse_optional_float(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "quantity")),
    ),
    ReceiptItemColumn(
        "Unit price",
        "unit_price",
        lambda value: _parse_optional_text(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "unit_price")),
    ),
    ReceiptItemColumn(
        "Amount",
        "amount",
        lambda value: _parse_required_text(value, "Amount"),
        lambda item: str(_effective_receipt_item_value(item, "amount")),
    ),
    ReceiptItemColumn(
        "Discount amount",
        "discount_amount",
        lambda value: _parse_optional_text(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "discount_amount")),
    ),
    ReceiptItemColumn(
        "Discount description",
        "discount_description",
        lambda value: _parse_optional_text(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "discount_description")),
    ),
    ReceiptItemColumn(
        "Net amount",
        "net_amount",
        lambda value: _parse_required_text(value, "Net amount"),
        lambda item: str(_effective_receipt_item_value(item, "net_amount")),
    ),
    ReceiptItemColumn(
        "Line type",
        "line_type",
        lambda value: _parse_optional_line_type(value),
        lambda item: _format_optional(
            _format_line_type(_effective_receipt_item_value(item, "line_type"))
        ),
    ),
    ReceiptItemColumn(
        "Category ID",
        "category_id",
        lambda value: _parse_optional_text(value),
        lambda item: _format_optional(_effective_receipt_item_value(item, "category_id")),
    ),
    *(
        ReceiptItemColumn(
            f"Taxonomy {index}",
            f"taxonomy{index}",
            lambda value: _parse_optional_text(value),
            lambda item, field_name=f"taxonomy{index}": _format_optional(
                _effective_receipt_item_value(item, field_name)
            ),
        )
        for index in range(1, 10)
    ),
)

def _open_file_in_external_viewer(filepath: str) -> None:
    """Opens a file in the system's default external viewer."""
    system_platform = platform.system()
    try:
        if system_platform == "Darwin":  # macOS
            subprocess.run(["open", filepath], check=True)
        elif system_platform == "Windows":  # Windows
            os.startfile(filepath)
        else:  # Linux (and other Unix-like systems)
            subprocess.run(["xdg-open", filepath], check=True)
    except Exception:
        # Ignore errors when opening external viewer to avoid crashing the app
        pass

def _transaction_sort_key(transaction: Transaction) -> tuple[object, ...]:
    try:
        amount = Decimal(_effective_transaction_amount(transaction))
    except InvalidOperation:
        return (
            _effective_transaction_date(transaction),
            1,
            _effective_transaction_amount(transaction),
        )

    return (_effective_transaction_date(transaction), 0, amount)


def _transaction_table_column_widths(terminal_width: int) -> dict[str, int]:
    available_width = max(0, terminal_width - len(TRANSACTION_TABLE_COLUMNS) - 1)
    fixed_width = sum(TRANSACTION_TABLE_FIXED_WIDTHS.values())
    flexible_width = max(0, available_width - fixed_width)
    flex_min_width = sum(column.min_width for column in TRANSACTION_TABLE_FLEX_COLUMNS)
    extra_width = max(0, flexible_width - flex_min_width)
    total_weight = sum(column.weight for column in TRANSACTION_TABLE_FLEX_COLUMNS)

    column_widths = dict(TRANSACTION_TABLE_FIXED_WIDTHS)
    remaining_extra_width = extra_width
    remaining_weight = total_weight
    for column in TRANSACTION_TABLE_FLEX_COLUMNS:
        if remaining_weight <= 0:
            column_width = column.min_width
        else:
            weighted_extra_width = remaining_extra_width * column.weight // remaining_weight
            column_width = min(column.max_width, column.min_width + weighted_extra_width)
        column_widths[column.key] = column_width
        remaining_extra_width -= column_width - column.min_width
        remaining_weight -= column.weight

    return column_widths


def _format_amount(amount: str, currency: str) -> str:
    try:
        decimal_amount = Decimal(amount)
    except InvalidOperation:
        formatted_amount = amount
    else:
        formatted_amount = f"{decimal_amount:,.2f}"

    return f"{formatted_amount} {currency}"


def _display_transactions(transactions: Sequence[Transaction]) -> list[Transaction]:
    linked_receipt_ids = {
        transaction.linked_receipt_based_transaction_id
        for transaction in transactions
        if _is_bank_statement_transaction(transaction)
        and transaction.linked_receipt_based_transaction_id is not None
    }
    return [transaction for transaction in transactions if transaction.id not in linked_receipt_ids]


def _receipt_transactions_by_display_id(
    display_transactions: Sequence[Transaction],
    transactions_by_id: dict[str, Transaction],
) -> dict[str, Transaction]:
    receipt_transactions: dict[str, Transaction] = {}
    for transaction in display_transactions:
        linked_receipt_id = transaction.linked_receipt_based_transaction_id
        if linked_receipt_id is not None and linked_receipt_id in transactions_by_id:
            receipt_transactions[transaction.id] = transactions_by_id[linked_receipt_id]
        elif _has_receipt_items(transaction):
            receipt_transactions[transaction.id] = transaction
    return receipt_transactions


def _format_receipt_indicator(transaction: Transaction, *, selected: bool = False) -> str:
    indicator = "Y" if _has_receipt_items(transaction) else ""
    if selected:
        return f"{indicator}*" if indicator else "*"
    return indicator


def _format_transaction_category(transaction: Transaction) -> str:
    allocations = _effective_category_allocations(transaction)
    if not allocations:
        return ""

    if len(allocations) == 1:
        return allocations[0].category_id

    largest_allocation = max(
        allocations,
        key=lambda allocation: abs(_decimal_amount(allocation.amount, "Category allocation amount")),
    )
    return f"{largest_allocation.category_id}, ..."


def _has_receipt_items(transaction: Transaction) -> bool:
    receipt = transaction.receipt
    return receipt is not None and bool(receipt.items)


def _is_bank_statement_transaction(transaction: Transaction) -> bool:
    record_type = getattr(transaction, "record_type", None)
    return record_type == "bank_statement" or transaction.source == "bank_statement"


def _is_receipt_based_transaction(transaction: Transaction) -> bool:
    record_type = getattr(transaction, "record_type", None)
    return record_type == "receipt_based" or transaction.source == "receipt"


def _receipt_item_row(item: ReceiptItem) -> list[str]:
    return [column.formatter(item) for column in RECEIPT_ITEM_COLUMNS]


def _format_optional(value: object | None) -> str:
    return "" if value is None else str(value)


def _format_line_type(value: object | None) -> str | None:
    if isinstance(value, LineType):
        return value.value
    return None if value is None else str(value)


def _effective_transaction_date(transaction: Transaction) -> date:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.transaction_date is not None:
        return overrides.transaction_date

    return transaction.transaction_date


def _effective_transaction_payee(transaction: Transaction) -> str:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.payee is not None:
        return overrides.payee

    return transaction.payee or ""


def _effective_transaction_description(transaction: Transaction) -> str:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.description is not None:
        return overrides.description

    return transaction.description or ""


def _effective_transaction_amount(transaction: Transaction) -> str:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.amount is not None:
        return overrides.amount

    return transaction.amount


def _effective_category_allocations(
    transaction: Transaction,
) -> list[CategoryAllocation | UserCategoryAllocation]:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.category_allocations is not None:
        return list(overrides.category_allocations)

    return list(transaction.category_allocations or [])


def _effective_receipt_item_value(item: ReceiptItem, field_name: str) -> object | None:
    overrides = item.user_overrides
    if overrides is not None:
        override_value = getattr(overrides, field_name)
        if override_value is not None:
            return override_value

    return getattr(item, field_name)


def _parse_required_text(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} must not be empty")

    return value


def _parse_required_decimal_text(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} must not be empty")
    _decimal_amount(value, label)
    return value


def _parse_optional_text(value: str) -> str | None:
    return value or None


def _parse_optional_float(value: str) -> float | None:
    if not value:
        return None

    return float(value)


def _parse_optional_date(value: str) -> date | None:
    if not value:
        return None

    return date.fromisoformat(value)


def _parse_optional_line_type(value: str) -> LineType | None:
    if not value:
        return None

    return LineType(value)


def _decimal_amount(value: str, label: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a decimal amount") from exc


def _header_input_label(input_widget: Input) -> str:
    match input_widget.id:
        case "receipt-date":
            return "transaction date"
        case "receipt-amount":
            return "transaction amount"
        case "receipt-payee":
            return "payee"
        case "receipt-description":
            return "description"
        case _:
            return "field"


def _field_edit_error_message(label: str, exc: ValueError | ValidationError) -> str:
    return f"Could not update {label}: {_validation_error_explanation(exc)}"


def _validation_error_explanation(exc: ValueError | ValidationError) -> str:
    if isinstance(exc, ValidationError):
        for error in exc.errors():
            error_type = str(error.get("type", ""))
            location = ".".join(str(part) for part in error.get("loc", ()))
            if error_type == "string_pattern_mismatch" and (
                location.endswith("amount")
                or location.endswith("unit_price")
                or location.endswith("discount_amount")
                or location.endswith("net_amount")
            ):
                return "enter a decimal amount like -12.34, with up to 4 decimal places."
            if error_type == "string_too_short":
                return "this value cannot be blank."
            if error_type == "greater_than":
                return "enter a number greater than zero."
            if error_type == "enum":
                return "enter one of the supported values for this field."
        return "the value does not match the expected format."

    message = str(exc)
    if "Invalid isoformat string" in message:
        return "enter a date as YYYY-MM-DD, or leave it blank to clear the override."
    if "must be a decimal amount" in message:
        return "enter a decimal amount like -12.34, with up to 4 decimal places."
    if message:
        return message
    return "the value does not match the expected format."
