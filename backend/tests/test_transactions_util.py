from receipts_ai.models import Transaction
from receipts_ai.transactions import transaction_combined_description


def test_transaction_combined_description_falls_back_to_description():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "receipt",
            "transaction_date": "2026-04-27",
            "payee": "Costco",
            "description": "Item purchase",
            "amount": "-42.19",
            "currency": "USD",
        }
    )
    assert transaction_combined_description(transaction) == "Item purchase"


def test_transaction_combined_description_blank_description_fallback():
    # Desired behavior: if description is empty, it should fall back to payee
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "receipt",
            "transaction_date": "2026-04-27",
            "payee": "Costco",
            "description": "",
            "amount": "-42.19",
            "currency": "USD",
        }
    )
    assert transaction_combined_description(transaction) == "Costco"


def test_transaction_combined_description_none_description_fallback():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "receipt",
            "transaction_date": "2026-04-27",
            "payee": "Costco",
            "description": None,
            "amount": "-42.19",
            "currency": "USD",
        }
    )
    assert transaction_combined_description(transaction) == "Costco"


def test_transaction_combined_description_bank_statement_returns_payee():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "bank_statement",
            "accountId": "acc_1",
            "transaction_date": "2026-04-27",
            "payee": "Costco",
            "description": "COSTCO WHSE #1234",
            "amount": "-42.19",
            "currency": "USD",
        }
    )
    assert transaction_combined_description(transaction) == "Costco"
