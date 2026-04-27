# RECEIPTS-AI

This is the monorepo for an expense tracking app.

This app works by assigning budget categories to transactions and receipt items.

## Bank statement transactions

If the user uploads a bank statement, say from a checking account or a credit card, each line will represent a transaction with a date, a payee and an amount. This transaction can be assigned to a budget category like Groceries. A transaction can also be broken out into more than one budget category. For example, a single $5 transaction could have three categories: $3 of Groceries, $1 of Electronics and $1 of Entertainment.

A transaction may have a receipt associated with it, in which case its budget categories can be deduced from the receipt line item categorizations. Statement amounts should be stored as signed amounts from the account perspective, while category allocations and receipt item amounts should preserve their original signs so refunds, discounts and returns can be represented.

## Receipt items

If the user uploads an itemized receipt, that will also represent a transaction with a date, a payee and a total amount, but it will also contain a set of items that each have an amount and description and can be assigned their own budget category. So a receipt from Costco might have a few items categorized as Groceries and one item categorized as Electronics. A receipt may also have items like Service Charges and Sales Tax.

Receipt subtotals, taxes, tips, discounts and fees are modeled as receipt line items so they can be categorized, prorated or ignored explicitly instead of being hidden in the transaction total.

## Bank statement / Receipt correlation

As noted above, a bank statement transaction can be associated with a receipt. This association will be made by the user. Matched bank statement transactions and receipt transactions should remain separate imported records, linked together after matching, so the app preserves the original source data while still allowing the UI and backend to treat them as the same real-world purchase when needed.

### Project Layout

The `backend` directory contains python code for ingesting and processing data, as well as serving it to the UI.

The `ui` directory contains a flutter app for viewing the transactions.

The `models` directory contains the JSON Schema for the project data.
