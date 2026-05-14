from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import cast

from receipts_ai.config import add_config_file_argument, config_value, configure_config_file
from receipts_ai.firestore_transactions import transactions_from_firestore
from receipts_ai.models.transaction import Transaction

BUDGET_CATEGORIES_FILE_ENV_VAR = "RECEIPTS_AI_BUDGET_CATEGORIES_FILE"
DEFAULT_BUDGET_CATEGORY_CATALOG_ID = "default-budget-categories"
DEFAULT_BUDGET_CATEGORY_CATALOG_VERSION = 1
DEFAULT_MAX_BUDGET_CATEGORY_DEPTH = 4
DEFAULT_MAX_BUDGET_CATEGORY_LEAVES = 82
BUDGET_CATEGORY_GROUP_EXPENSE = "expense"
BUDGET_CATEGORY_GROUP_INCOME = "income"
BUDGET_CATEGORY_GROUP_INTERNAL_TRANSFER = "internal_transfer"
SYSTEM_BUDGET_CATEGORY_GROUPS: tuple[str, ...] = (
    BUDGET_CATEGORY_GROUP_EXPENSE,
    BUDGET_CATEGORY_GROUP_INCOME,
    BUDGET_CATEGORY_GROUP_INTERNAL_TRANSFER,
)


@dataclass(frozen=True)
class BudgetCategory:
    id: str
    name: str
    parent_id: str | None = None
    status: str = "active"
    sort_order: int = 0
    aliases: tuple[str, ...] = ()
    system: bool = False


@dataclass(frozen=True)
class BudgetCategoryChoice:
    category_id: str
    path: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    @property
    def path_text(self) -> str:
        return " > ".join(self.path)


@dataclass(frozen=True)
class BudgetCategoryCatalog:
    id: str
    version: int
    max_depth: int
    categories: tuple[BudgetCategory, ...]

    @classmethod
    def from_json_payload(cls, payload: object) -> BudgetCategoryCatalog:
        if _is_catalog_payload(payload):
            return _catalog_from_record_payload(cast(Mapping[str, object], payload))
        if isinstance(payload, dict):
            return catalog_from_legacy_tree(cast(dict[str, object], payload))
        raise ValueError("budget categories file must contain a JSON object")

    @property
    def categories_by_id(self) -> dict[str, BudgetCategory]:
        return {category.id: category for category in self.categories}

    def choices(self) -> tuple[BudgetCategoryChoice, ...]:
        categories_by_id = self.categories_by_id
        child_counts = Counter(
            category.parent_id
            for category in self.categories
            if category.status == "active" and category.parent_id is not None
        )
        choices: list[BudgetCategoryChoice] = []
        for category in self.categories:
            if category.system or category.status != "active" or child_counts[category.id]:
                continue
            choices.append(
                BudgetCategoryChoice(
                    category_id=category.id,
                    path=_category_path(category.id, categories_by_id),
                    aliases=category.aliases,
                )
            )
        if not choices:
            raise RuntimeError("budget category catalog must contain at least one active leaf")
        return tuple(choices)

    def legacy_path_aliases(self) -> dict[str, str]:
        return {choice.path_text: choice.category_id for choice in self.choices()}

    def descendants(self, category_id: str) -> tuple[str, ...]:
        categories_by_id = self.categories_by_id
        if category_id not in categories_by_id:
            raise ValueError(f"unknown budget category id: {category_id}")
        children_by_parent: dict[str, list[str]] = {}
        for category in self.categories:
            if category.parent_id is not None:
                children_by_parent.setdefault(category.parent_id, []).append(category.id)
        descendants: list[str] = []
        pending = [category_id]
        while pending:
            current = pending.pop()
            descendants.append(current)
            pending.extend(children_by_parent.get(current, ()))
        return tuple(descendants)

    def category_group(self, category_id: str | None) -> str | None:
        if not category_id:
            return None
        categories_by_id = self.categories_by_id
        resolved_category_id = category_id
        if resolved_category_id not in categories_by_id:
            resolved_category_id = self.legacy_path_aliases().get(category_id, category_id)
        category = categories_by_id.get(resolved_category_id)
        while category is not None:
            if category.parent_id is None:
                return category.id if category.id in SYSTEM_BUDGET_CATEGORY_GROUPS else None
            category = categories_by_id.get(category.parent_id)
        return None

    def to_json_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": self.version,
            "maxDepth": self.max_depth,
            "categories": [_category_json_payload(category) for category in self.categories],
        }


@dataclass(frozen=True)
class BudgetCategoryCatalogValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("invalid budget category catalog:\n" + "\n".join(self.errors))


@dataclass(frozen=True)
class CategoryDeleteImpact:
    category_ids: tuple[str, ...]
    affected_transactions: int
    affected_reviewed_transactions: int
    affected_unreviewed_transactions: int
    affected_transaction_allocations: int
    affected_user_override_allocations: int
    affected_receipt_items: int
    affected_receipt_item_overrides: int


def load_budget_category_catalog(path: Path | str | None = None) -> BudgetCategoryCatalog:
    category_path = Path(path).expanduser() if path is not None else _configured_catalog_path()
    if category_path is not None:
        payload = json.loads(category_path.read_text(encoding="utf-8"))
    else:
        categories_file = resources.files("receipts_ai.models").joinpath("budget_categories.json")
        with categories_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    catalog = BudgetCategoryCatalog.from_json_payload(payload)
    validate_budget_category_catalog(catalog, max_leaf_categories=None).raise_for_errors()
    return catalog


def load_default_budget_category_catalog() -> BudgetCategoryCatalog:
    categories_file = resources.files("receipts_ai.models").joinpath("budget_categories.json")
    with categories_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    catalog = BudgetCategoryCatalog.from_json_payload(payload)
    validate_budget_category_catalog(catalog, max_leaf_categories=None).raise_for_errors()
    return catalog


def load_budget_categories(path: Path | str | None = None) -> dict[str, object]:
    category_path = Path(path).expanduser() if path is not None else _configured_catalog_path()
    if category_path is not None:
        payload = json.loads(category_path.read_text(encoding="utf-8"))
    else:
        categories_file = resources.files("receipts_ai.models").joinpath("budget_categories.json")
        with categories_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    if not isinstance(payload, dict):
        raise RuntimeError("budget categories file must contain a JSON object")
    return cast(dict[str, object], payload)


def load_budget_category_choices(
    categories: dict[str, object] | BudgetCategoryCatalog | None = None,
) -> tuple[str, ...]:
    catalog = _catalog_from_categories_argument(categories)
    return tuple(choice.path_text for choice in catalog.choices())


def load_budget_category_options(
    categories: dict[str, object] | BudgetCategoryCatalog | None = None,
) -> tuple[BudgetCategoryChoice, ...]:
    return _catalog_from_categories_argument(categories).choices()


def catalog_from_legacy_tree(
    categories: dict[str, object],
    *,
    catalog_id: str = DEFAULT_BUDGET_CATEGORY_CATALOG_ID,
    version: int = DEFAULT_BUDGET_CATEGORY_CATALOG_VERSION,
    max_depth: int = DEFAULT_MAX_BUDGET_CATEGORY_DEPTH,
) -> BudgetCategoryCatalog:
    records: list[BudgetCategory] = []

    def append_node(path: tuple[str, ...], node: object, parent_id: str | None) -> None:
        category_id = _category_id_from_path(path)
        records.append(
            BudgetCategory(
                id=category_id,
                name=path[-1],
                parent_id=parent_id,
                sort_order=len(records),
            )
        )
        if isinstance(node, dict):
            for child_name, child_node in cast(dict[str, object], node).items():
                append_node((*path, child_name), child_node, category_id)

    for category_name, child_node in categories.items():
        append_node((category_name,), child_node, None)

    return BudgetCategoryCatalog(
        id=catalog_id,
        version=version,
        max_depth=max_depth,
        categories=tuple(records),
    )


def validate_budget_category_catalog(
    catalog: BudgetCategoryCatalog,
    *,
    max_leaf_categories: int | None = DEFAULT_MAX_BUDGET_CATEGORY_LEAVES,
    require_system_groups: bool = True,
) -> BudgetCategoryCatalogValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not catalog.id:
        errors.append("catalog id must not be empty")
    if catalog.version < 1:
        errors.append("catalog version must be >= 1")
    if catalog.max_depth < 1:
        errors.append("catalog maxDepth must be >= 1")
    category_ids = [category.id for category in catalog.categories]
    duplicate_ids = sorted(
        category_id for category_id, count in Counter(category_ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate category ids: {', '.join(duplicate_ids)}")
    if not catalog.categories:
        errors.append("catalog must contain at least one category")

    categories_by_id = catalog.categories_by_id
    for category in catalog.categories:
        if not category.id:
            errors.append("category id must not be empty")
        if not category.name:
            errors.append(f"category {category.id!r} name must not be empty")
        if category.status not in {"active", "deleted"}:
            errors.append(f"category {category.id!r} has unsupported status {category.status!r}")
        if category.parent_id is not None and category.parent_id not in categories_by_id:
            errors.append(
                f"category {category.id!r} references missing parent {category.parent_id!r}"
            )

    if require_system_groups:
        for group in SYSTEM_BUDGET_CATEGORY_GROUPS:
            category = categories_by_id.get(group)
            if category is None:
                errors.append(f"missing required system budget category group {group!r}")
                continue
            if category.parent_id is not None:
                errors.append(f"system budget category group {group!r} must be a root category")
            if category.status != "active":
                errors.append(f"system budget category group {group!r} must be active")
            if not category.system:
                errors.append(f"system budget category group {group!r} must have system=true")

    if errors:
        return BudgetCategoryCatalogValidationResult(tuple(errors), tuple(warnings))

    for category in catalog.categories:
        seen: set[str] = set()
        current: BudgetCategory | None = category
        depth = 0
        while current is not None:
            if current.id in seen:
                errors.append(f"category {category.id!r} has a parent cycle")
                break
            seen.add(current.id)
            depth += 1
            current = (
                categories_by_id.get(current.parent_id) if current.parent_id is not None else None
            )
        if depth > catalog.max_depth:
            errors.append(
                f"category {category.id!r} depth {depth} exceeds maxDepth {catalog.max_depth}"
            )

    if not errors:
        leaf_count = len(catalog.choices())
        if max_leaf_categories is not None and leaf_count > max_leaf_categories:
            errors.append(
                "too many budget category leaves for single-token alias prompt: "
                f"got {leaf_count}, maximum is {max_leaf_categories}"
            )
        deleted_parents = {
            category.id for category in catalog.categories if category.status == "deleted"
        }
        for category in catalog.categories:
            if category.status == "active" and category.parent_id in deleted_parents:
                warnings.append(
                    f"active category {category.id!r} is under deleted parent {category.parent_id!r}"
                )
            if require_system_groups and category.status == "active" and not category.system:
                root_group = catalog.category_group(category.id)
                if root_group not in SYSTEM_BUDGET_CATEGORY_GROUPS:
                    errors.append(
                        f"active category {category.id!r} must be under a system budget category group"
                    )

    return BudgetCategoryCatalogValidationResult(tuple(errors), tuple(warnings))


def summarize_delete_category_impact(
    transactions: Sequence[Transaction],
    catalog: BudgetCategoryCatalog,
    category_id: str,
) -> CategoryDeleteImpact:
    affected_category_ids = catalog.descendants(category_id)
    affected = _affected_category_references(catalog, affected_category_ids)
    affected_transactions = 0
    reviewed = 0
    transaction_allocations = 0
    user_override_allocations = 0
    receipt_items = 0
    receipt_item_overrides = 0

    for transaction in transactions:
        transaction_hit = False
        for allocation in transaction.category_allocations or []:
            if allocation.category_id in affected:
                transaction_allocations += 1
                transaction_hit = True
        overrides = transaction.user_overrides
        if overrides is not None and overrides.category_allocations is not None:
            for allocation in overrides.category_allocations:
                if allocation.category_id in affected:
                    user_override_allocations += 1
                    transaction_hit = True
        if transaction.receipt is not None:
            for item in transaction.receipt.items:
                if item.category_id in affected:
                    receipt_items += 1
                    transaction_hit = True
                if item.user_overrides is not None and item.user_overrides.category_id in affected:
                    receipt_item_overrides += 1
                    transaction_hit = True
        if transaction_hit:
            affected_transactions += 1
            if transaction.reviewed:
                reviewed += 1

    return CategoryDeleteImpact(
        category_ids=affected_category_ids,
        affected_transactions=affected_transactions,
        affected_reviewed_transactions=reviewed,
        affected_unreviewed_transactions=affected_transactions - reviewed,
        affected_transaction_allocations=transaction_allocations,
        affected_user_override_allocations=user_override_allocations,
        affected_receipt_items=receipt_items,
        affected_receipt_item_overrides=receipt_item_overrides,
    )


def delete_category_from_transaction(
    transaction: Transaction,
    catalog: BudgetCategoryCatalog,
    category_id: str,
) -> bool:
    affected = _affected_category_references(catalog, catalog.descendants(category_id))
    changed = False
    allocations = [
        allocation
        for allocation in transaction.category_allocations or []
        if allocation.category_id not in affected
    ]
    if len(allocations) != len(transaction.category_allocations or []):
        transaction.category_allocations = allocations
        changed = True

    if (
        transaction.user_overrides is not None
        and transaction.user_overrides.category_allocations is not None
    ):
        user_allocations = [
            allocation
            for allocation in transaction.user_overrides.category_allocations
            if allocation.category_id not in affected
        ]
        if len(user_allocations) != len(transaction.user_overrides.category_allocations):
            transaction.user_overrides = transaction.user_overrides.model_copy(
                update={"category_allocations": user_allocations}
            )
            changed = True

    if transaction.receipt is not None:
        for item in transaction.receipt.items:
            if item.category_id in affected:
                item.category_id = None
                changed = True
            if item.user_overrides is not None and item.user_overrides.category_id in affected:
                item.user_overrides = item.user_overrides.model_copy(update={"category_id": None})
                changed = True

    if changed:
        transaction.reviewed = False
    return changed


def category_id_from_legacy_path(
    path: str,
    catalog: BudgetCategoryCatalog | None = None,
) -> str | None:
    active_catalog = catalog if catalog is not None else load_budget_category_catalog()
    return active_catalog.legacy_path_aliases().get(path)


def _category_json_payload(category: BudgetCategory) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": category.id,
        "name": category.name,
        "status": category.status,
        "sortOrder": category.sort_order,
    }
    if category.parent_id is not None:
        payload["parentId"] = category.parent_id
    if category.aliases:
        payload["aliases"] = list(category.aliases)
    if category.system:
        payload["system"] = category.system
    return payload


def _affected_category_references(
    catalog: BudgetCategoryCatalog, category_ids: Iterable[str]
) -> set[str]:
    affected = set(category_ids)
    for choice in catalog.choices():
        if choice.category_id in affected:
            affected.add(choice.path_text)
    return affected


def _catalog_from_categories_argument(
    categories: dict[str, object] | BudgetCategoryCatalog | None,
) -> BudgetCategoryCatalog:
    if isinstance(categories, BudgetCategoryCatalog):
        return categories
    if categories is not None:
        if _is_catalog_payload(categories):
            catalog = BudgetCategoryCatalog.from_json_payload(categories)
            validate_budget_category_catalog(catalog, max_leaf_categories=None).raise_for_errors()
            return catalog
        return catalog_from_legacy_tree(categories)
    return load_budget_category_catalog()


def _is_catalog_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    payload_mapping = cast(Mapping[str, object], payload)
    return isinstance(payload_mapping.get("categories"), list)


def _catalog_from_record_payload(payload: Mapping[str, object]) -> BudgetCategoryCatalog:
    records: list[BudgetCategory] = []
    raw_categories_payload = payload.get("categories", [])
    raw_categories: list[object]
    if isinstance(raw_categories_payload, list):
        raw_categories = cast(list[object], raw_categories_payload)
    else:
        raise ValueError("categories must be a list")
    for index, raw_category_payload in enumerate(raw_categories):
        if not isinstance(raw_category_payload, dict):
            raise ValueError(f"categories[{index}] must be a JSON object")
        raw_category = cast(Mapping[str, object], raw_category_payload)
        aliases = raw_category.get("aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) for alias in cast(list[object], aliases)
        ):
            raise ValueError(f"categories[{index}].aliases must be a list of strings")
        records.append(
            BudgetCategory(
                id=_required_string(raw_category, "id", index),
                name=_required_string(raw_category, "name", index),
                parent_id=_optional_string(raw_category, "parentId", index),
                status=_optional_string(raw_category, "status", index) or "active",
                sort_order=_optional_int(raw_category, "sortOrder", index) or index,
                aliases=tuple(cast(list[str], aliases)),
                system=_optional_bool(raw_category, "system", index) or False,
            )
        )
    return BudgetCategoryCatalog(
        id=_optional_payload_string(payload, "id") or DEFAULT_BUDGET_CATEGORY_CATALOG_ID,
        version=_optional_payload_int(payload, "version")
        or DEFAULT_BUDGET_CATEGORY_CATALOG_VERSION,
        max_depth=_optional_payload_int(payload, "maxDepth") or DEFAULT_MAX_BUDGET_CATEGORY_DEPTH,
        categories=tuple(records),
    )


def _required_string(raw_category: Mapping[str, object], key: str, index: int) -> str:
    value = raw_category.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"categories[{index}].{key} must be a non-empty string")
    return value


def _optional_string(raw_category: Mapping[str, object], key: str, index: int) -> str | None:
    value = raw_category.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"categories[{index}].{key} must be null or a non-empty string")
    return value


def _optional_int(raw_category: Mapping[str, object], key: str, index: int) -> int | None:
    value = raw_category.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"categories[{index}].{key} must be an integer")
    return value


def _optional_bool(raw_category: Mapping[str, object], key: str, index: int) -> bool | None:
    value = raw_category.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"categories[{index}].{key} must be a boolean")
    return value


def _optional_payload_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _optional_payload_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _category_path(
    category_id: str, categories_by_id: Mapping[str, BudgetCategory]
) -> tuple[str, ...]:
    parts: list[str] = []
    current = categories_by_id[category_id]
    while True:
        parts.append(current.name)
        if current.parent_id is None:
            break
        current = categories_by_id[current.parent_id]
    return tuple(reversed(parts))


def _category_id_from_path(path: tuple[str, ...]) -> str:
    return ".".join(_slugify_path_part(part) for part in path)


def _slugify_path_part(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot create category id from empty path component {value!r}")
    return slug


def _configured_catalog_path() -> Path | None:
    value = config_value(BUDGET_CATEGORIES_FILE_ENV_VAR)
    return Path(value).expanduser() if value else None


def _transactions_from_json_file(path: Path) -> list[Transaction]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if isinstance(payload, list):
        return [Transaction.model_validate(item) for item in cast(list[object], payload)]
    if isinstance(payload, dict):
        return [Transaction.model_validate(payload)]
    raise ValueError("transactions JSON file must contain one transaction object or a list")


def _transactions_from_jsonl_file(path: Path) -> list[Transaction]:
    transactions: list[Transaction] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        transactions.append(Transaction.model_validate(payload))
    return transactions


def _load_transactions_argument(path: Path | None) -> list[Transaction]:
    if path is None:
        return transactions_from_firestore()
    if path.suffix.lower() == ".jsonl":
        return _transactions_from_jsonl_file(path)
    return _transactions_from_json_file(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and inspect budget category catalogs.")
    add_config_file_argument(parser)
    parser.add_argument(
        "--categories",
        type=Path,
        help=(
            "Budget category JSON file. Defaults to "
            f"{BUDGET_CATEGORIES_FILE_ENV_VAR} or the packaged prototype catalog."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a category catalog.")
    validate_parser.add_argument(
        "--allow-too-many-leaves",
        action="store_true",
        help="Do not enforce the current single-token alias leaf limit.",
    )

    impact_parser = subparsers.add_parser(
        "impact-delete", help="Summarize the transaction impact of deleting a category."
    )
    impact_parser.add_argument("category_id")
    impact_parser.add_argument(
        "--transactions-json",
        type=Path,
        help="Transaction JSON or JSONL file. Defaults to Firestore.",
    )

    export_parser = subparsers.add_parser(
        "export-default", help="Print the packaged default catalog in record-style JSON."
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        help="Write to this file instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    configure_config_file(args.config_file)
    if args.command == "export-default":
        default_catalog = load_default_budget_category_catalog()
        text = json.dumps(default_catalog.to_json_payload(), indent=2) + "\n"
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
            print(f"Wrote {args.output}")
            return
        print(text, end="")
        return

    catalog = load_budget_category_catalog(args.categories)
    if args.command == "validate":
        max_leaves = None if args.allow_too_many_leaves else DEFAULT_MAX_BUDGET_CATEGORY_LEAVES
        result = validate_budget_category_catalog(catalog, max_leaf_categories=max_leaves)
        if result.errors:
            for error in result.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            raise SystemExit(1)
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        print(
            f"Catalog {catalog.id} v{catalog.version}: {len(catalog.choices())} leaf categories OK"
        )
        return

    if args.command == "impact-delete":
        transactions = _load_transactions_argument(args.transactions_json)
        impact = summarize_delete_category_impact(transactions, catalog, args.category_id)
        print(f"Deleting {args.category_id} affects {impact.affected_transactions} transaction(s)")
        print(f"Reviewed affected: {impact.affected_reviewed_transactions}")
        print(f"Unreviewed affected: {impact.affected_unreviewed_transactions}")
        print(f"Transaction allocations cleared: {impact.affected_transaction_allocations}")
        print(f"User override allocations cleared: {impact.affected_user_override_allocations}")
        print(f"Receipt item categories cleared: {impact.affected_receipt_items}")
        print(f"Receipt item override categories cleared: {impact.affected_receipt_item_overrides}")
        return


if __name__ == "__main__":
    main()
