import pytest
from pydantic import ValidationError

from receipts_ai.models import Transaction
from receipts_ai.models.transaction import ReceiptItem


def test_transaction_accepts_json_aliases():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "bank_statement",
            "ingestionFileUrl": "file:///tmp/statement.ofx",
            "transactionDate": "2026-04-27",
            "payee": "Costco",
            "mcc": "5411",
            "mccDescription": "Grocery Stores, Supermarkets",
            "braveSearchResult": "Costco Wholesale - search result",
            "receiptDataExtractionService": "azure-doc-intelligence",
            "receiptImageExtractionResults": '{"azure-doc-intelligence":{"status":"ok"},"gemini":{"status":"ok"}}',
            "amount": "-42.19",
            "currency": "USD",
            "linkedTransactionIds": ["receipt_1"],
            "transactionGroupId": "rtx_1",
            "groupRole": "primary",
            "matchStatus": "confirmed",
            "matchSource": "user",
            "matchConfidence": 1.0,
        }
    )

    assert transaction.transaction_date.isoformat() == "2026-04-27"
    assert transaction.ingestion_file_url == "file:///tmp/statement.ofx"
    assert transaction.mcc == "5411"
    assert transaction.mcc_description == "Grocery Stores, Supermarkets"
    assert transaction.brave_search_result == "Costco Wholesale - search result"
    assert transaction.receipt_data_extraction_service == "azure-doc-intelligence"
    assert (
        transaction.receipt_image_extraction_results
        == '{"azure-doc-intelligence":{"status":"ok"},"gemini":{"status":"ok"}}'
    )
    assert transaction.linked_transaction_ids is not None
    assert transaction.linked_transaction_ids[0] == "receipt_1"
    assert transaction.transaction_group_id == "rtx_1"
    assert transaction.group_role == "primary"
    assert transaction.match_status == "confirmed"
    assert transaction.match_source == "user"
    assert transaction.match_confidence == 1.0


def test_transaction_accepts_python_field_names():
    transaction = Transaction.model_validate(
        {
            "id": "txn_1",
            "source": "manual",
            "ingestion_file_url": "file:///tmp/manual.json",
            "transaction_date": "2026-04-27",
            "payee": "Manual Adjustment",
            "receipt_data_extraction_service": "gemini-3-flash-lite",
            "receipt_image_extraction_results": '{"gemini":{"status":"ok"}}',
            "amount": "12.00",
            "currency": "USD",
        }
    )

    assert transaction.transaction_date.isoformat() == "2026-04-27"
    assert transaction.ingestion_file_url == "file:///tmp/manual.json"
    assert transaction.receipt_data_extraction_service == "gemini-3-flash-lite"
    assert transaction.receipt_image_extraction_results == '{"gemini":{"status":"ok"}}'
    assert transaction.match_status == "unmatched"


def test_transaction_accepts_match_metadata_python_field_names():
    transaction = Transaction.model_validate(
        {
            "id": "receipt_1",
            "source": "receipt",
            "transaction_date": "2026-04-27",
            "amount": "42.19",
            "currency": "USD",
            "transaction_group_id": "rtx_1",
            "group_role": "supporting",
            "match_status": "candidate",
            "match_source": "model",
            "match_confidence": 0.78,
        }
    )

    assert transaction.transaction_group_id == "rtx_1"
    assert transaction.group_role == "supporting"
    assert transaction.match_status == "candidate"
    assert transaction.match_source == "model"
    assert transaction.match_confidence == 0.78


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


def test_receipt_item_user_overrides_description():
    item = ReceiptItem.model_validate(
        {
            "description": "Original",
            "amount": "1.00",
            "netAmount": "1.00",
            "userOverrides": {"description": "Overridden"},
        }
    )

    assert item.description == "Original"
    assert item.user_overrides is not None
    assert item.user_overrides.description == "Overridden"


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
