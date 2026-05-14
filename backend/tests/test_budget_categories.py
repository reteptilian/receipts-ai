from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from receipts_ai import budget_categories
from receipts_ai.budget_categories import (
    BUDGET_CATEGORY_GROUP_EXPENSE,
    BUDGET_CATEGORY_GROUP_INCOME,
    BUDGET_CATEGORY_GROUP_INTERNAL_TRANSFER,
    BudgetCategory,
    BudgetCategoryCatalog,
    catalog_from_legacy_tree,
    delete_category_from_transaction,
    load_budget_category_catalog,
    summarize_delete_category_impact,
    validate_budget_category_catalog,
)
from receipts_ai.models.transaction import (
    CategoryAllocation,
    Receipt,
    ReceiptItem,
    ReceiptItemUserOverrides,
    Source,
    Transaction,
    TransactionUserOverrides,
    UserCategoryAllocation,
)


def test_catalog_from_legacy_tree_generates_stable_leaf_ids():
    catalog = catalog_from_legacy_tree(
        {
            "Food & Dining": {
                "Groceries": {},
                "Restaurants & Dining Out": {},
            }
        }
    )

    choices = catalog.choices()

    assert [(choice.category_id, choice.path_text) for choice in choices] == [
        ("food_dining.groceries", "Food & Dining > Groceries"),
        (
            "food_dining.restaurants_dining_out",
            "Food & Dining > Restaurants & Dining Out",
        ),
    ]
    assert catalog.legacy_path_aliases()["Food & Dining > Groceries"] == "food_dining.groceries"


def test_load_budget_category_catalog_accepts_record_format(tmp_path: Path):
    categories_path = tmp_path / "budget_categories.json"
    categories_path.write_text(
        json.dumps(
            {
                "id": "user-budget",
                "version": 3,
                "maxDepth": 4,
                "categories": [
                    {"id": "expense", "name": "Expense", "sortOrder": 0, "system": True},
                    {"id": "income", "name": "Income", "sortOrder": 1, "system": True},
                    {
                        "id": "internal_transfer",
                        "name": "Internal Transfers",
                        "sortOrder": 2,
                        "system": True,
                    },
                    {"id": "food", "name": "Food", "parentId": "expense", "sortOrder": 10},
                    {
                        "id": "food.groceries",
                        "name": "Groceries",
                        "parentId": "food",
                        "sortOrder": 11,
                        "aliases": ["Supermarket"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog = load_budget_category_catalog(categories_path)

    assert catalog.id == "user-budget"
    assert catalog.version == 3
    assert catalog.choices()[0].category_id == "food.groceries"
    assert catalog.choices()[0].aliases == ("Supermarket",)


def test_budget_category_catalog_resolves_system_groups():
    catalog = load_budget_category_catalog()

    assert catalog.category_group("food_dining.groceries") == BUDGET_CATEGORY_GROUP_EXPENSE
    assert catalog.category_group("income.wages") == BUDGET_CATEGORY_GROUP_INCOME
    assert (
        catalog.category_group("internal_transfer.bank_transfers")
        == BUDGET_CATEGORY_GROUP_INTERNAL_TRANSFER
    )
    assert catalog.category_group(None) is None


def test_validate_budget_category_catalog_requires_system_groups():
    catalog = catalog_from_legacy_tree({"Food": {"Groceries": {}}})

    result = validate_budget_category_catalog(catalog, max_leaf_categories=None)

    assert "missing required system budget category group 'expense'" in result.errors


def test_validate_budget_category_catalog_rejects_too_many_leaf_categories():
    catalog = catalog_from_legacy_tree(
        {"Budget": {f"Category {index}": {} for index in range(1, 84)}}
    )

    result = validate_budget_category_catalog(
        catalog, max_leaf_categories=82, require_system_groups=False
    )

    assert result.errors == (
        "too many budget category leaves for single-token alias prompt: got 83, maximum is 82",
    )


def test_validate_budget_category_catalog_rejects_more_than_four_levels():
    catalog = BudgetCategoryCatalog(
        id="test",
        version=1,
        max_depth=4,
        categories=(
            BudgetCategory(id="a", name="A"),
            BudgetCategory(id="b", name="B", parent_id="a"),
            BudgetCategory(id="c", name="C", parent_id="b"),
            BudgetCategory(id="d", name="D", parent_id="c"),
            BudgetCategory(id="e", name="E", parent_id="d"),
        ),
    )

    result = validate_budget_category_catalog(catalog, require_system_groups=False)

    assert "category 'e' depth 5 exceeds maxDepth 4" in result.errors


def test_delete_category_impact_counts_allocations_and_review_state():
    catalog = catalog_from_legacy_tree({"Food": {"Groceries": {}, "Restaurants": {}}})
    transactions = [
        Transaction(
            id="t1",
            source=Source.bank_statement,
            transaction_date=date(2026, 5, 1),
            amount="-10.00",
            currency="USD",
            reviewed=True,
            category_allocations=[
                CategoryAllocation(category_id="food.groceries", amount="-10.00")
            ],
        ),
        Transaction(
            id="t2",
            source=Source.receipt,
            transaction_date=date(2026, 5, 2),
            amount="-5.00",
            currency="USD",
            reviewed=False,
            receipt=Receipt(
                items=[
                    ReceiptItem(
                        description="Apple",
                        amount="5.00",
                        net_amount="5.00",
                        category_id="food.groceries",
                    )
                ]
            ),
        ),
    ]

    impact = summarize_delete_category_impact(transactions, catalog, "food.groceries")

    assert impact.affected_transactions == 2
    assert impact.affected_reviewed_transactions == 1
    assert impact.affected_unreviewed_transactions == 1
    assert impact.affected_transaction_allocations == 1
    assert impact.affected_receipt_items == 1


def test_delete_category_impact_counts_legacy_path_references():
    catalog = catalog_from_legacy_tree({"Food": {"Groceries": {}}})
    transactions = [
        Transaction(
            id="t1",
            source=Source.bank_statement,
            transaction_date=date(2026, 5, 1),
            amount="-10.00",
            currency="USD",
            reviewed=True,
            category_allocations=[
                CategoryAllocation(category_id="Food > Groceries", amount="-10.00")
            ],
        )
    ]

    impact = summarize_delete_category_impact(transactions, catalog, "food.groceries")

    assert impact.affected_transactions == 1
    assert impact.affected_transaction_allocations == 1


def test_delete_category_from_transaction_clears_references_and_marks_unreviewed():
    catalog = catalog_from_legacy_tree({"Food": {"Groceries": {}, "Restaurants": {}}})
    transaction = Transaction(
        id="t1",
        source=Source.receipt,
        transaction_date=date(2026, 5, 1),
        amount="-10.00",
        currency="USD",
        reviewed=True,
        category_allocations=[CategoryAllocation(category_id="food.groceries", amount="-5.00")],
        user_overrides=TransactionUserOverrides(
            category_allocations=[
                UserCategoryAllocation(category_id="food.groceries", amount="-5.00")
            ]
        ),
        receipt=Receipt(
            items=[
                ReceiptItem(
                    description="Apple",
                    amount="5.00",
                    net_amount="5.00",
                    category_id="food.groceries",
                    user_overrides=ReceiptItemUserOverrides(category_id="food.groceries"),
                ),
                ReceiptItem(
                    description="Burger",
                    amount="5.00",
                    net_amount="5.00",
                    category_id="food.restaurants",
                ),
            ]
        ),
    )

    changed = delete_category_from_transaction(transaction, catalog, "food.groceries")

    assert changed is True
    assert transaction.reviewed is False
    assert transaction.category_allocations == []
    assert transaction.user_overrides is not None
    assert transaction.user_overrides.category_allocations == []
    assert transaction.receipt is not None
    assert transaction.receipt.items[0].category_id is None
    assert transaction.receipt.items[0].user_overrides is not None
    assert transaction.receipt.items[0].user_overrides.category_id is None
    assert transaction.receipt.items[1].category_id == "food.restaurants"


def test_catalog_descendants_raises_for_unknown_category():
    catalog = catalog_from_legacy_tree({"Food": {"Groceries": {}}})

    with pytest.raises(ValueError, match="unknown budget category id"):
        catalog.descendants("missing")


def test_main_export_default_writes_record_style_catalog(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    output_path = tmp_path / "budget_categories.json"

    budget_categories.main(["export-default", "--output", output_path.as_posix()])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["id"] == "default-budget-categories"
    assert payload["maxDepth"] == 4
    assert isinstance(payload["categories"], list)
    assert {
        "id": "food_dining.groceries",
        "name": "Groceries",
        "parentId": "food_dining",
        "status": "active",
        "sortOrder": 31,
    } in payload["categories"]
    assert {
        "id": "expense",
        "name": "Expense",
        "status": "active",
        "sortOrder": 0,
        "system": True,
    } in payload["categories"]
    assert f"Wrote {output_path}" in capsys.readouterr().out
