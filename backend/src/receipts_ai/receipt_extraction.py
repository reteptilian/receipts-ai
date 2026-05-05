from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from receipts_ai.document_intelligence import to_jsonable
from receipts_ai.models.transaction import (
    ExtractionMetadata,
    LineType,
    Receipt,
    ReceiptItem,
    Source,
    Transaction,
)

__all__ = (
    "receipt_from_document_intelligence_result",
    "transaction_from_document_intelligence_result",
)


def receipt_from_document_intelligence_result(result: Any) -> Receipt:
    transaction = transaction_from_document_intelligence_result(result)
    if transaction.receipt is None:
        raise ValueError("transaction does not contain a receipt")
    return transaction.receipt


def transaction_from_document_intelligence_result(result: Any) -> Transaction:
    payload = cast(dict[str, Any], to_jsonable(result))
    documents = _list_value(payload.get("documents"))
    if not documents:
        raise ValueError("document intelligence result does not contain any documents")

    document = _dict_value(documents[0])
    fields = _dict_value(document.get("fields"))
    payee = _field_text(fields.get("MerchantName"))
    if payee is None:
        raise ValueError("document intelligence result does not contain a merchant name")

    transaction_date = _field_date(fields.get("TransactionDate"))
    if transaction_date is None:
        raise ValueError("document intelligence result does not contain a transaction date")

    total_field = fields.get("Total")
    total = _currency_amount(total_field)
    if total is None:
        raise ValueError("document intelligence result does not contain a receipt total")

    items = _receipt_items(fields.get("Items"), payee=payee)
    tax_amount = _currency_amount(fields.get("TotalTax"))
    if tax_amount is not None and Decimal(tax_amount) != 0:
        items.append(
            ReceiptItem(
                description="Sales tax",
                amount=tax_amount,
                net_amount=tax_amount,
                line_type=LineType.tax,
            )
        )
    currency = _currency_code(total_field) or "USD"

    receipt = Receipt(
        subtotal=_currency_amount(fields.get("Subtotal")),
        total=total,
        items=items,
        extraction=ExtractionMetadata(
            model=_string_value(payload.get("modelId")) or _string_value(payload.get("model_id")),
            confidence=_float_value(document.get("confidence")),
            raw_text=_string_value(payload.get("content")),
        ),
    )
    return Transaction(
        id=_transaction_id(payee, transaction_date, total, _string_value(payload.get("content"))),
        source=Source.receipt,
        transaction_date=transaction_date,
        payee=payee,
        amount=_expense_amount(total),
        currency=currency,
        receipt=receipt,
    )


def _receipt_items(items_field: Any, *, payee: str) -> list[ReceiptItem]:
    items: list[ReceiptItem] = []
    for item_field in _list_value(_dict_value(items_field).get("valueArray")):
        item = _dict_value(item_field)
        item_object = _dict_value(item.get("valueObject"))
        description = _field_text(item_object.get("Description"))
        amount = _currency_amount(item_object.get("TotalPrice"))

        if description is None or amount is None:
            continue

        items.append(
            ReceiptItem(
                description=description,
                raw_description=description,
                quantity=_float_value(_nested_field(item_object, "Quantity", "valueNumber")),
                unit_price=_currency_amount(item_object.get("Price")),
                amount=amount,
                net_amount=amount,
                confidence=_float_value(item.get("confidence")),
            )
        )

    if not items:
        raise ValueError("document intelligence result does not contain receipt line items")
    return _merge_vendor_discount_items(items, payee=payee)


def _merge_vendor_discount_items(items: list[ReceiptItem], *, payee: str) -> list[ReceiptItem]:
    if payee.strip().upper() != "COSTCO":
        return items

    merged_items: list[ReceiptItem] = []
    for item in items:
        if _is_costco_coupon_discount(item) and merged_items:
            _apply_discount(merged_items[-1], item)
            continue
        merged_items.append(item)
    return merged_items


def _is_costco_coupon_discount(item: ReceiptItem) -> bool:
    if re.fullmatch(r"/\d+", item.description.strip()) is None:
        return False

    try:
        return Decimal(item.amount) < 0
    except InvalidOperation:
        return False


def _apply_discount(item: ReceiptItem, discount: ReceiptItem) -> None:
    discount_amount = Decimal(discount.amount)
    if item.discount_amount is not None:
        discount_amount += Decimal(item.discount_amount)

    item.discount_amount = format(discount_amount, "f")
    item.discount_description = _combined_discount_description(
        item.discount_description,
        discount.raw_description or discount.description,
    )
    item.net_amount = format(Decimal(item.amount) + discount_amount, "f")


def _combined_discount_description(existing: str | None, new: str) -> str:
    if existing is None:
        return new
    return f"{existing}; {new}"


def _nested_field(parent: dict[str, Any], field_name: str, value_name: str) -> Any:
    return _dict_value(parent.get(field_name)).get(value_name)


def _field_text(field: Any) -> str | None:
    field_dict = _dict_value(field)
    return _string_value(field_dict.get("valueString")) or _string_value(field_dict.get("content"))


def _field_date(field: Any) -> date | None:
    field_dict = _dict_value(field)
    value_date = _string_value(field_dict.get("valueDate"))
    if value_date is not None:
        try:
            return date.fromisoformat(value_date)
        except ValueError:
            pass

    content = _string_value(field_dict.get("content"))
    if content is None:
        return None

    for date_format in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(content, date_format).date()
        except ValueError:
            continue
    return None


def _currency_amount(field: Any) -> str | None:
    field_dict = _dict_value(field)
    currency = _dict_value(field_dict.get("valueCurrency"))
    content = _string_value(field_dict.get("content"))

    if content is not None:
        normalized_content = _decimal_string(content)
        if normalized_content is not None:
            return normalized_content

    amount = currency.get("amount")
    if amount is None:
        return None
    return _decimal_string(amount)


def _currency_code(field: Any) -> str | None:
    currency = _dict_value(_dict_value(field).get("valueCurrency"))
    code = _string_value(currency.get("currencyCode"))
    if code is None:
        return None
    return code.upper()


def _expense_amount(total: str) -> str:
    amount = Decimal(total)
    if amount > 0:
        amount = -amount
    return format(amount, "f")


def _transaction_id(payee: str, transaction_date: date, total: str, raw_text: str | None) -> str:
    key = "|".join((payee, transaction_date.isoformat(), total, raw_text or ""))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"receipt_{digest}"


def _decimal_string(value: object) -> str | None:
    try:
        decimal = Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, ValueError):
        return None

    return format(decimal, "f")


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return cast(list[Any], value)
    return []


def _string_value(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _float_value(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
