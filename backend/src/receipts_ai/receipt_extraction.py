from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, cast

from receipts_ai.document_intelligence import to_jsonable
from receipts_ai.models.transaction import ExtractionMetadata, Receipt, ReceiptItem

__all__ = ("receipt_from_document_intelligence_result",)


def receipt_from_document_intelligence_result(result: Any) -> Receipt:
    payload = cast(dict[str, Any], to_jsonable(result))
    documents = _list_value(payload.get("documents"))
    if not documents:
        raise ValueError("document intelligence result does not contain any documents")

    document = _dict_value(documents[0])
    fields = _dict_value(document.get("fields"))
    items = _receipt_items(fields.get("Items"))

    return Receipt(
        subtotal=_currency_amount(fields.get("Subtotal")),
        total=_currency_amount(fields.get("Total")),
        items=items,
        extraction=ExtractionMetadata(
            model=_string_value(payload.get("modelId")) or _string_value(payload.get("model_id")),
            confidence=_float_value(document.get("confidence")),
            raw_text=_string_value(payload.get("content")),
        ),
    )


def _receipt_items(items_field: Any) -> list[ReceiptItem]:
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
                confidence=_float_value(item.get("confidence")),
            )
        )

    if not items:
        raise ValueError("document intelligence result does not contain receipt line items")
    return items


def _nested_field(parent: dict[str, Any], field_name: str, value_name: str) -> Any:
    return _dict_value(parent.get(field_name)).get(value_name)


def _field_text(field: Any) -> str | None:
    field_dict = _dict_value(field)
    return _string_value(field_dict.get("valueString")) or _string_value(field_dict.get("content"))


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
