import pytest
from pydantic import ValidationError

from receipts_ai.models import Transaction


def test_transaction_accepts_json_aliases():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "bank_statement",
            "transactionDate": "2026-04-27",
            "payee": "Costco",
            "amount": "-42.19",
            "currency": "USD",
            "linkedTransactionIds": ["receipt_1"],
        }
    )

    assert transaction.transaction_date.isoformat() == "2026-04-27"
    assert transaction.linked_transaction_ids is not None
    assert transaction.linked_transaction_ids[0] == "receipt_1"


def test_transaction_accepts_python_field_names():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "manual",
            "transaction_date": "2026-04-27",
            "payee": "Manual Adjustment",
            "amount": "12.00",
            "currency": "USD",
        }
    )

    assert transaction.transaction_date.isoformat() == "2026-04-27"


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
