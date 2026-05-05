import pytest
from pydantic import ValidationError

from receipts_ai.models import Transaction
from receipts_ai.models.transaction import ReceiptItem


def test_transaction_accepts_json_aliases():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "bank_statement",
            "transactionDate": "2026-04-27",
            "payee": "Costco",
            "mcc": "5411",
            "mccDescription": "Grocery Stores, Supermarkets",
            "braveSearchResult": "Costco Wholesale - search result",
            "receiptImageExtractionResults": '{"azure-doc-intelligence":{"status":"ok"},"gemini":{"status":"ok"}}',
            "amount": "-42.19",
            "currency": "USD",
            "linkedTransactionIds": ["receipt_1"],
        }
    )

    assert transaction.transaction_date.isoformat() == "2026-04-27"
    assert transaction.mcc == "5411"
    assert transaction.mcc_description == "Grocery Stores, Supermarkets"
    assert transaction.brave_search_result == "Costco Wholesale - search result"
    assert (
        transaction.receipt_image_extraction_results
        == '{"azure-doc-intelligence":{"status":"ok"},"gemini":{"status":"ok"}}'
    )
    assert transaction.linked_transaction_ids is not None
    assert transaction.linked_transaction_ids[0] == "receipt_1"


def test_transaction_accepts_python_field_names():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "manual",
            "transaction_date": "2026-04-27",
            "payee": "Manual Adjustment",
            "receipt_image_extraction_results": '{"gemini":{"status":"ok"}}',
            "amount": "12.00",
            "currency": "USD",
        }
    )

    assert transaction.transaction_date.isoformat() == "2026-04-27"
    assert transaction.receipt_image_extraction_results == '{"gemini":{"status":"ok"}}'


def test_transaction_payee_is_optional():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "bank_statement",
            "transaction_date": "2026-04-27",
            "amount": "-42.19",
            "currency": "USD",
        }
    )

    assert transaction.payee is None


def test_receipt_item_accepts_raw_description_alias():
    item = ReceiptItem.model_validate(
        {
            "description": "Saltine Crackers",
            "rawDescription": "NBSC SALTINE",
            "braveSearchResult": "Nabisco Premium Saltine Crackers - search result",
            "amount": "4.49",
            "discountAmount": "-1.50",
            "discountDescription": "/1779212",
            "netAmount": "2.99",
        }
    )

    assert item.description == "Saltine Crackers"
    assert item.raw_description == "NBSC SALTINE"
    assert item.brave_search_result == "Nabisco Premium Saltine Crackers - search result"
    assert item.discount_amount == "-1.50"
    assert item.discount_description == "/1779212"
    assert item.net_amount == "2.99"


def test_receipt_item_requires_net_amount():
    with pytest.raises(ValidationError):
        ReceiptItem.model_validate(
            {
                "description": "Bagel",
                "amount": "3.00",
            }
        )


def test_transaction_rejects_invalid_currency():
    with pytest.raises(ValidationError):
        Transaction.model_validate(
            {
                "id": "txn_1",
                "source": "manual",
                "transaction_date": "2026-04-27",
                "payee": "Manual Adjustment",
                "amount": "12.00",
                "currency": "usd",
            }
        )


def test_transaction_rejects_invalid_mcc():
    with pytest.raises(ValidationError):
        Transaction.model_validate(
            {
                "id": "txn_1",
                "source": "bank_statement",
                "transaction_date": "2026-04-27",
                "amount": "-42.19",
                "currency": "USD",
                "mcc": "541",
            }
        )
