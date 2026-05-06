from __future__ import annotations

from receipts_ai.models.transaction import Source, Transaction

__all__ = ("transaction_combined_description",)


def transaction_combined_description(transaction: Transaction) -> str | None:
    """Return the display description used for transaction-level rows."""
    if (
        transaction.source == Source.bank_statement
        and transaction.payee is not None
        and transaction.account_id is not None
        and ":" not in transaction.account_id
    ):
        return transaction.payee
    return transaction.description or transaction.payee
