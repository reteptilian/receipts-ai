from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any, Protocol, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from receipts_ai.budget_categories import BudgetCategoryCatalog, load_budget_category_catalog
from receipts_ai.firestore_client import create_firestore_client
from receipts_ai.models.transaction import (
    CategoryAllocation,
    ReceiptItem,
    Source1,
    Transaction,
)

DEFAULT_AUTOMATION_RULES_COLLECTION = "automationRules"

LOGGER = logging.getLogger(__name__)


class FirestoreDocumentSnapshot(Protocol):
    id: str

    def to_dict(self) -> dict[str, Any] | None: ...


class FirestoreDocumentReference(Protocol):
    def set(self, document_data: dict[str, Any], *, merge: bool = False) -> object: ...


class FirestoreCollectionReference(Protocol):
    def document(self, document_id: str) -> FirestoreDocumentReference: ...

    def stream(self) -> Iterable[FirestoreDocumentSnapshot]: ...


class FirestoreRuleClient(Protocol):
    def collection(self, collection_path: str) -> FirestoreCollectionReference: ...


class RuleScope(StrEnum):
    transaction = "transaction"
    receipt_item = "receipt_item"


class RuleConditionField(StrEnum):
    payee = "payee"
    receipt_item_description = "receipt_item_description"


class RuleConditionOperator(StrEnum):
    equals = "equals"


class RuleActionType(StrEnum):
    set_payee = "set_payee"
    set_category_allocations = "set_category_allocations"
    set_receipt_item_category = "set_receipt_item_category"
    set_receipt_item_taxonomy = "set_receipt_item_taxonomy"
    set_transaction_taxonomy = "set_transaction_taxonomy"


class AutomationRuleStatus(StrEnum):
    active = "active"
    deleted = "deleted"


class RuleCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    field: RuleConditionField
    operator: RuleConditionOperator = RuleConditionOperator.equals
    value: str = Field(min_length=1)


class CategoryAllocationPercent(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    category_id: str = Field(alias="categoryId", min_length=1)
    percent: str = Field(pattern="^(0|[1-9][0-9]*)(\\.[0-9]{1,4})?$")


class RuleAction(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: RuleActionType
    value: str | None = None
    category_id: str | None = Field(default=None, alias="categoryId")
    allocations: list[CategoryAllocationPercent] | None = None


class AutomationRule(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    scope: RuleScope
    enabled: bool = True
    status: AutomationRuleStatus = AutomationRuleStatus.active
    priority: int = 100
    conditions: list[RuleCondition] = Field(min_length=1)
    actions: list[RuleAction] = Field(min_length=1)
    created_from_transaction_id: str | None = Field(default=None, alias="createdFromTransactionId")
    created_at: AwareDatetime | None = Field(default=None, alias="createdAt")
    updated_at: AwareDatetime | None = Field(default=None, alias="updatedAt")
    deleted_at: AwareDatetime | None = Field(default=None, alias="deletedAt")


class RuleSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    key: str
    prompt: str
    rule: AutomationRule


def stream_automation_rules(
    *,
    client: FirestoreRuleClient | None = None,
    collection: str = DEFAULT_AUTOMATION_RULES_COLLECTION,
) -> Iterable[AutomationRule]:
    if not collection:
        raise ValueError("collection must not be empty")
    firestore_client = _firestore_rule_client(client)
    for snapshot in firestore_client.collection(collection).stream():
        document = snapshot.to_dict()
        if document is None:
            continue
        yield AutomationRule.model_validate(document)


def automation_rules_from_firestore(
    *,
    client: FirestoreRuleClient | None = None,
    collection: str = DEFAULT_AUTOMATION_RULES_COLLECTION,
) -> list[AutomationRule]:
    return list(stream_automation_rules(client=client, collection=collection))


def save_automation_rule(
    rule: AutomationRule,
    *,
    client: FirestoreRuleClient | None = None,
    collection: str = DEFAULT_AUTOMATION_RULES_COLLECTION,
    updated_at: datetime | None = None,
) -> None:
    if not collection:
        raise ValueError("collection must not be empty")
    timestamp = updated_at or datetime.now(UTC)
    document_rule = rule.model_copy(
        update={
            "created_at": rule.created_at or timestamp,
            "updated_at": timestamp,
        }
    )
    _firestore_rule_client(client).collection(collection).document(rule.id).set(
        document_rule.model_dump(mode="json", by_alias=True, exclude_none=True),
        merge=True,
    )


def delete_automation_rule(
    rule_id: str,
    *,
    client: FirestoreRuleClient | None = None,
    collection: str = DEFAULT_AUTOMATION_RULES_COLLECTION,
    updated_at: datetime | None = None,
) -> None:
    if not rule_id:
        raise ValueError("rule_id must not be empty")
    if not collection:
        raise ValueError("collection must not be empty")
    timestamp = updated_at or datetime.now(UTC)
    _firestore_rule_client(client).collection(collection).document(rule_id).set(
        {
            "enabled": False,
            "status": AutomationRuleStatus.deleted.value,
            "deletedAt": _json_datetime(timestamp),
            "updatedAt": _json_datetime(timestamp),
        },
        merge=True,
    )


def rule_invalid_reason(
    rule: AutomationRule,
    *,
    category_catalog: BudgetCategoryCatalog | None = None,
) -> str | None:
    catalog = category_catalog or load_budget_category_catalog()
    active_leaf_ids = {choice.category_id for choice in catalog.choices()}
    for action in rule.actions:
        category_ids = _action_category_ids(action)
        for category_id in category_ids:
            if category_id not in active_leaf_ids:
                return f"category {category_id!r} is not an active leaf category"
    return None


def apply_automation_rules(
    transaction: Transaction,
    rules: Sequence[AutomationRule],
    *,
    category_catalog: BudgetCategoryCatalog | None = None,
) -> Transaction:
    catalog = category_catalog or load_budget_category_catalog()
    for rule in sorted(rules, key=lambda candidate: candidate.priority, reverse=True):
        if not rule.enabled or rule.status != AutomationRuleStatus.active:
            continue
        invalid_reason = rule_invalid_reason(rule, category_catalog=catalog)
        if invalid_reason is not None:
            LOGGER.warning("Skipping invalid automation rule %s: %s", rule.id, invalid_reason)
            continue
        if rule.scope == RuleScope.transaction and _transaction_rule_matches(rule, transaction):
            _apply_transaction_actions(transaction, rule.actions)
        elif rule.scope == RuleScope.receipt_item and transaction.receipt is not None:
            for item in transaction.receipt.items:
                if _receipt_item_rule_matches(rule, item):
                    _apply_receipt_item_actions(item, rule.actions)
    return transaction


def apply_automation_rules_from_firestore(
    transactions: Sequence[Transaction],
    *,
    client: FirestoreRuleClient | None = None,
) -> None:
    try:
        rules = automation_rules_from_firestore(client=client)
    except Exception:
        LOGGER.exception("Skipping automation rules because rules could not be loaded")
        return

    for transaction in transactions:
        try:
            apply_automation_rules(transaction, rules)
        except Exception:
            LOGGER.exception("Skipping automation rules for transaction %s", transaction.id)


def generate_rule_suggestions(
    original_transaction: Transaction,
    edited_transaction: Transaction,
    *,
    original_receipt_transaction: Transaction | None = None,
    edited_receipt_transaction: Transaction | None = None,
    category_catalog: BudgetCategoryCatalog | None = None,
) -> tuple[RuleSuggestion, ...]:
    if edited_transaction.reviewed is not True:
        return ()

    catalog = category_catalog or load_budget_category_catalog()
    suggestions: list[RuleSuggestion] = []
    original_payee = _effective_transaction_payee(original_transaction)
    edited_payee = _effective_transaction_payee(edited_transaction)

    if original_payee and edited_payee and original_payee != edited_payee:
        rule = _transaction_rule(
            name=f"Rename {original_payee}",
            payee=original_payee,
            action=RuleAction(type=RuleActionType.set_payee, value=edited_payee),
            transaction_id=original_transaction.id,
        )
        suggestions.append(
            _suggestion(
                f"Always rename payee {original_payee!r} to {edited_payee!r}?",
                rule,
            )
        )

    original_allocations = _effective_category_allocations(original_transaction)
    edited_allocations = _effective_category_allocations(edited_transaction)
    if (
        original_payee
        and edited_allocations
        and _allocation_signature(original_allocations) != _allocation_signature(edited_allocations)
    ):
        try:
            allocations = _allocation_percents(
                edited_allocations,
                transaction_amount=Decimal(_effective_transaction_amount(edited_transaction)),
            )
        except ValueError:
            allocations = ()
        if allocations:
            action = RuleAction(
                type=RuleActionType.set_category_allocations,
                allocations=list(allocations),
            )
            rule = _transaction_rule(
                name=f"Categorize {original_payee}",
                payee=original_payee,
                action=action,
                transaction_id=original_transaction.id,
            )
            if rule_invalid_reason(rule, category_catalog=catalog) is None:
                suggestions.append(
                    _suggestion(
                        f"Always categorize payee {original_payee!r} this way?",
                        rule,
                    )
                )

    original_taxonomy = _effective_transaction_taxonomy(original_transaction)
    edited_taxonomy = _effective_transaction_taxonomy(edited_transaction)
    if original_payee and original_taxonomy != edited_taxonomy:
        taxonomy_value = edited_taxonomy or ""
        rule = _transaction_rule(
            name=f"Set taxonomy for {original_payee}",
            payee=original_payee,
            action=RuleAction(
                type=RuleActionType.set_transaction_taxonomy,
                value=taxonomy_value,
            ),
            transaction_id=original_transaction.id,
        )
        prompt = (
            f"Always leave taxonomy unassigned for payee {original_payee!r}?"
            if not taxonomy_value
            else f"Always set taxonomy for payee {original_payee!r} to {taxonomy_value!r}?"
        )
        suggestions.append(_suggestion(prompt, rule))

    original_receipt = original_receipt_transaction or original_transaction
    edited_receipt = edited_receipt_transaction or edited_transaction
    if original_receipt.receipt is None or edited_receipt.receipt is None:
        return tuple(suggestions)

    for index, original_item in enumerate(original_receipt.receipt.items):
        if index >= len(edited_receipt.receipt.items):
            continue
        edited_item = edited_receipt.receipt.items[index]
        item_description = _effective_receipt_item_description(original_item)
        if not item_description:
            continue
        original_category = _effective_receipt_item_category(original_item)
        edited_category = _effective_receipt_item_category(edited_item)
        if edited_category and original_category != edited_category:
            rule = _receipt_item_rule(
                name=f"Categorize {item_description}",
                description=item_description,
                action=RuleAction.model_validate(
                    {
                        "type": RuleActionType.set_receipt_item_category,
                        "categoryId": edited_category,
                    }
                ),
                transaction_id=original_transaction.id,
            )
            if rule_invalid_reason(rule, category_catalog=catalog) is None:
                suggestions.append(
                    _suggestion(
                        f"Always assign receipt item {item_description!r} to this category?",
                        rule,
                    )
                )
        original_taxonomy = _effective_receipt_item_taxonomy(original_item)
        edited_taxonomy = _effective_receipt_item_taxonomy(edited_item)
        if edited_taxonomy and original_taxonomy != edited_taxonomy:
            rule = _receipt_item_rule(
                name=f"Classify {item_description}",
                description=item_description,
                action=RuleAction(
                    type=RuleActionType.set_receipt_item_taxonomy,
                    value=edited_taxonomy,
                ),
                transaction_id=original_transaction.id,
            )
            suggestions.append(
                _suggestion(
                    f"Always classify receipt item {item_description!r} as {edited_taxonomy!r}?",
                    rule,
                )
            )
    return tuple(suggestions)


def _firestore_rule_client(client: FirestoreRuleClient | None) -> FirestoreRuleClient:
    return (
        client
        if client is not None
        else cast(FirestoreRuleClient, cast(object, create_firestore_client()))
    )


def _transaction_rule(
    *,
    name: str,
    payee: str,
    action: RuleAction,
    transaction_id: str,
) -> AutomationRule:
    return _rule(
        name=name,
        scope=RuleScope.transaction,
        condition=RuleCondition(field=RuleConditionField.payee, value=payee),
        action=action,
        transaction_id=transaction_id,
    )


def _receipt_item_rule(
    *,
    name: str,
    description: str,
    action: RuleAction,
    transaction_id: str,
) -> AutomationRule:
    return _rule(
        name=name,
        scope=RuleScope.receipt_item,
        condition=RuleCondition(
            field=RuleConditionField.receipt_item_description,
            value=description,
        ),
        action=action,
        transaction_id=transaction_id,
    )


def _rule(
    *,
    name: str,
    scope: RuleScope,
    condition: RuleCondition,
    action: RuleAction,
    transaction_id: str,
) -> AutomationRule:
    payload = {
        "scope": scope.value,
        "conditions": [condition.model_dump(mode="json")],
        "actions": [action.model_dump(mode="json", by_alias=True, exclude_none=True)],
    }
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]
    return AutomationRule.model_validate(
        {
            "id": f"rule_{digest}",
            "name": name,
            "scope": scope,
            "conditions": [condition],
            "actions": [action],
            "createdFromTransactionId": transaction_id,
        }
    )


def _suggestion(prompt: str, rule: AutomationRule) -> RuleSuggestion:
    return RuleSuggestion(key=rule.id, prompt=prompt, rule=rule)


def _transaction_rule_matches(rule: AutomationRule, transaction: Transaction) -> bool:
    for condition in rule.conditions:
        if condition.field == RuleConditionField.payee:
            value = transaction.payee or ""
        else:
            return False
        if condition.operator == RuleConditionOperator.equals and value != condition.value:
            return False
    return True


def _receipt_item_rule_matches(rule: AutomationRule, item: ReceiptItem) -> bool:
    for condition in rule.conditions:
        if condition.field == RuleConditionField.receipt_item_description:
            value = item.description
        else:
            return False
        if condition.operator == RuleConditionOperator.equals and value != condition.value:
            return False
    return True


def _apply_transaction_actions(transaction: Transaction, actions: Sequence[RuleAction]) -> None:
    for action in actions:
        if action.type == RuleActionType.set_payee and action.value:
            transaction.payee = action.value
        elif action.type == RuleActionType.set_transaction_taxonomy and action.value is not None:
            transaction.taxonomy = action.value or None
        elif action.type == RuleActionType.set_category_allocations and action.allocations:
            transaction.category_allocations = [
                CategoryAllocation(
                    category_id=allocation.category_id,
                    amount=amount,
                    confidence=1.0,
                    source=Source1.rule,
                )
                for allocation, amount in zip(
                    action.allocations,
                    _amounts_from_percents(transaction.amount, action.allocations),
                    strict=True,
                )
            ]


def _apply_receipt_item_actions(item: ReceiptItem, actions: Sequence[RuleAction]) -> None:
    for action in actions:
        if action.type == RuleActionType.set_receipt_item_category and action.category_id:
            item.category_id = action.category_id
            item.confidence = 1.0
        elif action.type == RuleActionType.set_receipt_item_taxonomy and action.value:
            item.taxonomy = action.value
            item.confidence = 1.0


def _action_category_ids(action: RuleAction) -> tuple[str, ...]:
    if action.category_id:
        return (action.category_id,)
    if action.allocations:
        return tuple(allocation.category_id for allocation in action.allocations)
    return ()


def _effective_transaction_payee(transaction: Transaction) -> str | None:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.payee is not None:
        return overrides.payee
    return transaction.payee


def _effective_transaction_amount(transaction: Transaction) -> str:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.amount is not None:
        return overrides.amount
    return transaction.amount


def _effective_transaction_taxonomy(transaction: Transaction) -> str | None:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.taxonomy is not None:
        return overrides.taxonomy or None
    return transaction.taxonomy


def _effective_category_allocations(transaction: Transaction) -> list[CategoryAllocation]:
    overrides = transaction.user_overrides
    if overrides is not None and overrides.category_allocations is not None:
        return [
            CategoryAllocation(category_id=allocation.category_id, amount=allocation.amount)
            for allocation in overrides.category_allocations
        ]
    return list(transaction.category_allocations or [])


def _effective_receipt_item_description(item: ReceiptItem) -> str:
    overrides = item.user_overrides
    if overrides is not None and overrides.description is not None:
        return overrides.description
    return item.description


def _effective_receipt_item_category(item: ReceiptItem) -> str | None:
    overrides = item.user_overrides
    if overrides is not None and overrides.category_id is not None:
        return overrides.category_id
    return item.category_id


def _effective_receipt_item_taxonomy(item: ReceiptItem) -> str | None:
    overrides = item.user_overrides
    if overrides is not None and overrides.taxonomy is not None:
        return overrides.taxonomy or None
    return item.taxonomy


def _allocation_signature(allocations: Sequence[CategoryAllocation]) -> tuple[tuple[str, str], ...]:
    return tuple((allocation.category_id, allocation.amount) for allocation in allocations)


def _allocation_percents(
    allocations: Sequence[CategoryAllocation],
    *,
    transaction_amount: Decimal,
) -> tuple[CategoryAllocationPercent, ...]:
    if transaction_amount == 0:
        raise ValueError("cannot build percentage allocation rule from zero amount")
    return tuple(
        CategoryAllocationPercent.model_validate(
            {
                "categoryId": allocation.category_id,
                "percent": str(
                    (Decimal(allocation.amount) / transaction_amount * Decimal("100")).quantize(
                        Decimal("0.0001")
                    )
                ),
            }
        )
        for allocation in allocations
    )


def _amounts_from_percents(
    transaction_amount: str,
    allocations: Sequence[CategoryAllocationPercent],
) -> tuple[str, ...]:
    total = Decimal(transaction_amount)
    amounts: list[Decimal] = []
    allocated = Decimal("0.00")
    for index, allocation in enumerate(allocations):
        if index == len(allocations) - 1:
            amount = total - allocated
        else:
            amount = (total * Decimal(allocation.percent) / Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            allocated += amount
        amounts.append(amount)
    return tuple(format(amount, "f") for amount in amounts)


def _json_datetime(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
