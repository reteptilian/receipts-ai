from __future__ import annotations

from datetime import date

from receipts_ai.automation_rules import (
    AutomationRule,
    CategoryAllocationPercent,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionField,
    RuleScope,
    apply_automation_rules,
    generate_rule_suggestions,
    rule_invalid_reason,
)
from receipts_ai.budget_categories import BudgetCategory, BudgetCategoryCatalog
from receipts_ai.models.transaction import (
    Receipt,
    ReceiptItem,
    ReceiptItemUserOverrides,
    Source,
    Source1,
    Transaction,
    TransactionUserOverrides,
    UserCategoryAllocation,
)


def _catalog() -> BudgetCategoryCatalog:
    return BudgetCategoryCatalog(
        id="test",
        version=1,
        max_depth=4,
        categories=(
            BudgetCategory(id="transportation", name="Transportation"),
            BudgetCategory(
                id="transportation.gas_fuel",
                name="Gas & Fuel",
                parent_id="transportation",
            ),
            BudgetCategory(id="food", name="Food"),
            BudgetCategory(id="food.groceries", name="Groceries", parent_id="food"),
        ),
    )


def _transaction() -> Transaction:
    return Transaction(
        id="txn_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 5, 6),
        payee="COSTCO #101 GAS STATION",
        amount="-50.00",
        currency="USD",
    )


def test_generates_category_rule_suggestion_from_reviewed_edit() -> None:
    original = _transaction()
    edited = original.model_copy(deep=True)
    edited.reviewed = True
    edited.user_overrides = TransactionUserOverrides(
        category_allocations=[
            UserCategoryAllocation(category_id="transportation.gas_fuel", amount="-50.00")
        ]
    )

    suggestions = generate_rule_suggestions(original, edited, category_catalog=_catalog())

    assert len(suggestions) == 1
    rule = suggestions[0].rule
    assert rule.scope == RuleScope.transaction
    assert rule.conditions == [
        RuleCondition(
            field=RuleConditionField.payee,
            value="COSTCO #101 GAS STATION",
        )
    ]
    assert rule.actions[0].type == RuleActionType.set_category_allocations
    assert rule.actions[0].allocations == [
        CategoryAllocationPercent.model_validate(
            {"categoryId": "transportation.gas_fuel", "percent": "100.0000"}
        )
    ]


def test_no_rule_suggestion_when_reviewed_transaction_has_no_new_edit_delta() -> None:
    original = _transaction()
    original.reviewed = True
    original.user_overrides = TransactionUserOverrides(
        category_allocations=[
            UserCategoryAllocation(category_id="transportation.gas_fuel", amount="-50.00")
        ]
    )
    edited = original.model_copy(deep=True)

    assert generate_rule_suggestions(original, edited, category_catalog=_catalog()) == ()


def test_apply_payee_rule_sets_rule_sourced_category_allocations() -> None:
    transaction = _transaction()
    rule = AutomationRule(
        id="rule_1",
        name="Costco gas",
        scope=RuleScope.transaction,
        conditions=[
            RuleCondition(
                field=RuleConditionField.payee,
                value="COSTCO #101 GAS STATION",
            )
        ],
        actions=[
            RuleAction(
                type=RuleActionType.set_category_allocations,
                allocations=[
                    CategoryAllocationPercent.model_validate(
                        {
                            "categoryId": "transportation.gas_fuel",
                            "percent": "100",
                        }
                    )
                ],
            )
        ],
    )

    apply_automation_rules(transaction, [rule], category_catalog=_catalog())

    assert transaction.category_allocations is not None
    assert transaction.category_allocations[0].category_id == "transportation.gas_fuel"
    assert transaction.category_allocations[0].amount == "-50.00"
    assert transaction.category_allocations[0].source == Source1.rule


def test_generates_transaction_taxonomy_clear_rule_suggestion() -> None:
    original = _transaction()
    original.taxonomy = "Vehicles & Parts > Vehicle Fuels & Lubricants"
    edited = original.model_copy(deep=True)
    edited.reviewed = True
    edited.user_overrides = TransactionUserOverrides(taxonomy="")

    suggestions = generate_rule_suggestions(original, edited, category_catalog=_catalog())

    assert len(suggestions) == 1
    assert suggestions[0].prompt == (
        "Always leave taxonomy unassigned for payee 'COSTCO #101 GAS STATION'?"
    )
    assert suggestions[0].rule.actions[0].type == RuleActionType.set_transaction_taxonomy
    assert suggestions[0].rule.actions[0].value == ""


def test_apply_transaction_taxonomy_clear_rule_sets_taxonomy_to_none() -> None:
    transaction = _transaction()
    transaction.taxonomy = "Vehicles & Parts > Vehicle Fuels & Lubricants"
    rule = AutomationRule(
        id="rule_1",
        name="No Costco gas taxonomy",
        scope=RuleScope.transaction,
        conditions=[
            RuleCondition(
                field=RuleConditionField.payee,
                value="COSTCO #101 GAS STATION",
            )
        ],
        actions=[
            RuleAction(
                type=RuleActionType.set_transaction_taxonomy,
                value="",
            )
        ],
    )

    apply_automation_rules(transaction, [rule], category_catalog=_catalog())

    assert transaction.taxonomy is None


def test_rule_with_deleted_category_is_invalid_and_skipped() -> None:
    catalog = BudgetCategoryCatalog(
        id="test",
        version=1,
        max_depth=4,
        categories=(
            BudgetCategory(id="transportation", name="Transportation"),
            BudgetCategory(
                id="transportation.gas_fuel",
                name="Gas & Fuel",
                parent_id="transportation",
                status="deleted",
            ),
        ),
    )
    rule = AutomationRule(
        id="rule_1",
        name="Costco gas",
        scope=RuleScope.transaction,
        conditions=[
            RuleCondition(
                field=RuleConditionField.payee,
                value="COSTCO #101 GAS STATION",
            )
        ],
        actions=[
            RuleAction(
                type=RuleActionType.set_category_allocations,
                allocations=[
                    CategoryAllocationPercent.model_validate(
                        {
                            "categoryId": "transportation.gas_fuel",
                            "percent": "100",
                        }
                    )
                ],
            )
        ],
    )
    transaction = _transaction()

    assert rule_invalid_reason(rule, category_catalog=catalog) == (
        "category 'transportation.gas_fuel' is not an active leaf category"
    )
    apply_automation_rules(transaction, [rule], category_catalog=catalog)

    assert transaction.category_allocations == []


def test_generates_receipt_item_taxonomy_rule_suggestion() -> None:
    original = _transaction()
    original.receipt = Receipt(
        items=[ReceiptItem(description="KIRKLAND BATTERIES", amount="12.00", net_amount="12.00")]
    )
    edited = original.model_copy(deep=True)
    edited.reviewed = True
    assert edited.receipt is not None
    edited.receipt.items[0].user_overrides = ReceiptItemUserOverrides(
        taxonomy="Electronics > Batteries"
    )

    suggestions = generate_rule_suggestions(original, edited, category_catalog=_catalog())

    assert len(suggestions) == 1
    assert suggestions[0].prompt == (
        "Always classify receipt item 'KIRKLAND BATTERIES' as 'Electronics > Batteries'?"
    )
    assert suggestions[0].rule.scope == RuleScope.receipt_item
    assert suggestions[0].rule.actions[0].type == RuleActionType.set_receipt_item_taxonomy
