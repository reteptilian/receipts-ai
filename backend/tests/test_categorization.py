from __future__ import annotations

import json
import logging
import math
import shlex
import urllib.error
import urllib.request
from datetime import date
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import TypedDict, Unpack

import pytest

from receipts_ai.cache import SqliteCallCache
from receipts_ai.categorization import (
    DEFAULT_OLLAMA_URL,
    MAX_TAXONOMY_ALIAS_CHOICES,
    MAX_TRANSACTION_CATEGORY_ALIAS_CHOICES,
    CachedCategoryModelClient,
    CategoryChoiceProbability,
    CategoryCompletion,
    TaxonomyEmbeddingEntry,
    TaxonomyEmbeddingIndex,
    UrlLibOllamaClient,
    categorize_receipt_items,
    categorize_transactions,
    classify_receipt_items_by_product_taxonomy,
    classify_receipt_items_by_product_taxonomy_vector_search,
    clean_receipt_item_descriptions,
    create_ollama_category_client,
    load_budget_categories,
    load_budget_category_choices,
    load_product_taxonomy,
    load_product_taxonomy_embeddings,
    search_product_taxonomy_embeddings,
)
from receipts_ai.models.transaction import LineType, Receipt, Source, Transaction
from receipts_ai.models.transaction import ReceiptItem as GeneratedReceiptItem


class ReceiptItemKwargs(TypedDict, total=False):
    id: str | None
    description: str
    raw_description: str | None
    brave_search_result: str | None
    quantity: float | None
    unit_price: str | None
    amount: str
    discount_amount: str | None
    discount_description: str | None
    net_amount: str
    line_type: LineType | None
    category_id: str | None
    taxonomy1: str | None
    taxonomy2: str | None
    taxonomy3: str | None
    taxonomy4: str | None
    taxonomy5: str | None
    taxonomy6: str | None
    taxonomy7: str | None
    taxonomy8: str | None
    taxonomy9: str | None
    confidence: float | None


def ReceiptItem(**kwargs: Unpack[ReceiptItemKwargs]) -> GeneratedReceiptItem:  # noqa: N802
    if "amount" in kwargs and "net_amount" not in kwargs:
        kwargs["net_amount"] = kwargs["amount"]
    return GeneratedReceiptItem(**kwargs)


class FakeCategoryClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FakeProbabilityCategoryClient:
    def __init__(self, responses: list[CategoryCompletion]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0).response

    def complete_with_probabilities(
        self, prompt: str, *, choices: tuple[str, ...]
    ) -> CategoryCompletion:
        _ = choices
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FakeTaxonomyEmbeddingClient:
    def __init__(self, embedding: tuple[float, ...]) -> None:
        self.embedding = embedding
        self.texts: list[str] = []

    def embed(self, text: str) -> tuple[float, ...]:
        self.texts.append(text)
        return self.embedding


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_load_budget_categories_exposes_nested_categories():
    categories = load_budget_categories()

    assert "Food & Dining" in categories
    food_categories = categories["Food & Dining"]
    assert isinstance(food_categories, dict)
    assert "Groceries" in food_categories


def test_load_budget_category_choices_flattens_leaf_paths():
    choices = load_budget_category_choices(
        {
            "Taxes": {
                "Income Taxes": {},
                "Sales Taxes": {},
            },
            "Miscellaneous": {
                "Uncategorized": {},
            },
        }
    )

    assert choices == (
        "Taxes > Income Taxes",
        "Taxes > Sales Taxes",
        "Miscellaneous > Uncategorized",
    )


def test_load_product_taxonomy_parses_greater_than_levels(tmp_path: Path):
    taxonomy_path = tmp_path / "taxonomy.en-US.txt"
    taxonomy_path.write_text(
        "\n".join(
            (
                "# Comment",
                "Food, Beverages & Tobacco",
                "Food, Beverages & Tobacco > Food Items",
                "Food, Beverages & Tobacco > Food Items > Bakery",
                "Food, Beverages & Tobacco > Food Items > Bakery > Crackers",
            )
        ),
        encoding="utf-8",
    )

    taxonomy = load_product_taxonomy(taxonomy_path)

    assert taxonomy == {
        "Food, Beverages & Tobacco": {
            "Food Items": {
                "Bakery": {
                    "Crackers": {},
                },
            },
        },
    }


def test_load_product_taxonomy_embeddings_parses_vector_artifact(tmp_path: Path):
    embeddings_path = tmp_path / "taxonomy_embeddings.json"
    embeddings_path.write_text(
        json.dumps(
            {
                "embedding_model": "test-model",
                "embedding_dimension": 2,
                "entries": [
                    {
                        "path": "Electronics > Audio > Headphones",
                        "parts": ["Electronics", "Audio", "Headphones"],
                        "embedding": [1.0, 0.0],
                    },
                    {
                        "path": "Food, Beverages & Tobacco > Food Items > Bakery > Crackers",
                        "parts": ["Food, Beverages & Tobacco", "Food Items", "Bakery", "Crackers"],
                        "embedding": [0.0, 1.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    taxonomy_embeddings = load_product_taxonomy_embeddings(embeddings_path)

    assert taxonomy_embeddings.embedding_model == "test-model"
    assert taxonomy_embeddings.embedding_dimension == 2
    assert taxonomy_embeddings.entries == (
        TaxonomyEmbeddingEntry(
            path=("Electronics", "Audio", "Headphones"),
            embedding=(1.0, 0.0),
        ),
        TaxonomyEmbeddingEntry(
            path=("Food, Beverages & Tobacco", "Food Items", "Bakery", "Crackers"),
            embedding=(0.0, 1.0),
        ),
    )


def test_search_product_taxonomy_embeddings_returns_nearest_paths():
    taxonomy_embeddings = TaxonomyEmbeddingIndex(
        embedding_model="test-model",
        embedding_dimension=2,
        entries=(
            TaxonomyEmbeddingEntry(
                path=("Food, Beverages & Tobacco", "Food Items", "Bakery", "Crackers"),
                embedding=(0.0, 1.0),
            ),
            TaxonomyEmbeddingEntry(
                path=("Electronics", "Audio", "Speakers"),
                embedding=(0.8, 0.2),
            ),
            TaxonomyEmbeddingEntry(
                path=("Electronics", "Audio", "Headphones"),
                embedding=(1.0, 0.0),
            ),
        ),
    )
    embedding_client = FakeTaxonomyEmbeddingClient((1.0, 0.0))

    results = search_product_taxonomy_embeddings(
        "Apple AirPods Pro 3",
        taxonomy_embeddings=taxonomy_embeddings,
        embedding_client=embedding_client,
        candidate_count=2,
    )

    assert embedding_client.texts == ["Apple AirPods Pro 3"]
    assert [result.path for result in results] == [
        ("Electronics", "Audio", "Headphones"),
        ("Electronics", "Audio", "Speakers"),
    ]
    assert [result.score for result in results] == [1.0, 0.8]


def test_clean_receipt_item_descriptions_uses_raw_text_and_top_five_search_results():
    item = ReceiptItem(
        description="Confusing receipt text",
        raw_description="NBSC SALTINE",
        amount="4.49",
        brave_search_result=json.dumps(
            [
                {
                    "title": "Nabisco Premium Saltine Crackers",
                    "description": "Original saltine crackers in a family-size box.",
                },
                {"title": "Result 2", "description": "Description 2"},
                {"title": "Result 3", "description": "Description 3"},
                {"title": "Result 4", "description": "Description 4"},
                {"title": "Result 5", "description": "Description 5"},
                {"title": "Result 6", "description": "Description 6"},
            ]
        ),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    client = FakeCategoryClient(['"Nabisco Premium Saltine Crackers"\nExplanation omitted'])

    result = clean_receipt_item_descriptions(transaction, client=client)

    assert result is transaction
    assert item.description == "Nabisco Premium Saltine Crackers"
    assert len(client.prompts) == 1
    assert "Raw receipt text: NBSC SALTINE" in client.prompts[0]
    assert "Nabisco Premium Saltine Crackers" in client.prompts[0]
    assert "Original saltine crackers in a family-size box." in client.prompts[0]
    assert "Result 5" in client.prompts[0]
    assert "Result 6" not in client.prompts[0]


def test_categorize_receipt_items_uses_clean_description_and_flattened_category():
    item = ReceiptItem(
        description="Nabisco Premium Saltine Crackers",
        raw_description="RAW CONFUSING CODE",
        amount="4.49",
        brave_search_result=json.dumps(
            [
                {
                    "title": "Premium Saltines",
                    "description": "Crisp crackers sold in grocery stores.",
                }
            ]
        ),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    client = FakeCategoryClient(["Food & Dining > Groceries"])

    result = categorize_receipt_items(transaction, client=client)

    assert result is transaction
    assert item.category_id == "Food & Dining > Groceries"
    assert len(client.prompts) == 1
    assert "Food & Dining" in client.prompts[0]
    assert "Groceries" in client.prompts[0]
    assert "Receipt item description: Nabisco Premium Saltine Crackers" in client.prompts[0]
    assert "Premium Saltines" not in "\n".join(client.prompts)
    assert "Crisp crackers sold in grocery stores." not in "\n".join(client.prompts)
    assert "RAW CONFUSING CODE" not in "\n".join(client.prompts)


def test_categorize_receipt_items_can_choose_from_flattened_categories():
    item = ReceiptItem(
        description="Nabisco Premium Saltine Crackers",
        amount="4.49",
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    client = FakeCategoryClient(["Food & Dining > Groceries"])

    result = categorize_receipt_items(
        transaction,
        client=client,
        categories={
            "Housing & Utilities": {"Mortgage & Rent": {}},
            "Food & Dining": {"Groceries": {}, "Restaurants & Dining Out": {}},
        },
    )

    assert result is transaction
    assert item.category_id == "Food & Dining > Groceries"
    assert len(client.prompts) == 1
    assert "Choose the best budget category path." in client.prompts[0]
    assert "- Housing & Utilities > Mortgage & Rent" in client.prompts[0]
    assert "- Food & Dining > Groceries" in client.prompts[0]


def test_categorize_transactions_sets_single_model_allocation_from_flattened_categories():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="COSTCO WHSE",
        description="POS PURCHASE COSTCO WHSE #123",
        brave_search_result=json.dumps(
            [
                {
                    "title": "Costco Wholesale",
                    "description": "Warehouse store with groceries and household goods.",
                }
            ]
        ),
        amount="-42.19",
        currency="USD",
    )
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="1",
                probabilities=(CategoryChoiceProbability("1", 0.78),),
            ),
        ]
    )

    result = categorize_transactions(
        [transaction],
        client=client,
        categories={"Food & Dining": {"Groceries": {}, "Restaurants & Dining Out": {}}},
    )

    assert result == [transaction]
    allocations = transaction.category_allocations
    assert allocations is not None
    assert len(allocations) == 1
    allocation = allocations[0]
    assert allocation.category_id == "Food & Dining > Groceries"
    assert allocation.amount == "-42.19"
    assert allocation.confidence == 0.78
    assert allocation.source == "model"
    assert (
        "Raw transaction description: COSTCO WHSE POS PURCHASE COSTCO WHSE #123"
        in client.prompts[0]
    )
    assert "Merchant category:" not in client.prompts[0]
    # assert "1. Title: Costco Wholesale" in client.prompts[0]
    assert "1: Food & Dining > Groceries" in client.prompts[0]
    assert client.prompts[0].endswith("Label: ")


def test_categorize_transactions_includes_mcc_description_in_budget_prompt():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="COSTCO WHSE",
        description="POS PURCHASE COSTCO WHSE #123",
        mcc="5411",
        mcc_description="Grocery Stores, Supermarkets",
        amount="-42.19",
        currency="USD",
    )
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="1",
                probabilities=(CategoryChoiceProbability("1", 0.82),),
            ),
        ]
    )

    categorize_transactions(
        [transaction],
        client=client,
        categories={"Food & Dining": {"Groceries": {}}},
    )

    assert "Merchant category: Grocery Stores, Supermarkets" in client.prompts[0]


def test_categorize_transactions_can_choose_from_flattened_categories():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="COSTCO WHSE",
        description="POS PURCHASE COSTCO WHSE #123",
        amount="-42.19",
        currency="USD",
    )
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="1",
                probabilities=(CategoryChoiceProbability("1", 0.82),),
            ),
        ]
    )

    categorize_transactions(
        [transaction],
        client=client,
        categories={
            "Food & Dining": {"Groceries": {}, "Restaurants & Dining Out": {}},
            "Housing & Utilities": {"Mortgage & Rent": {}},
        },
    )

    allocations = transaction.category_allocations
    assert allocations is not None
    assert len(allocations) == 1
    assert allocations[0].category_id == "Food & Dining > Groceries"
    assert allocations[0].confidence == 0.82
    assert len(client.prompts) == 1
    assert "Choose the best budget category path." in client.prompts[0]
    assert "1: Food & Dining > Groceries" in client.prompts[0]
    assert "2: Food & Dining > Restaurants & Dining Out" in client.prompts[0]


def test_categorize_transactions_strips_bank_transfer_noise_from_prompt():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        description="Descriptive Withdrawal P2P Transfer  Maia April lessons",
        amount="-200.00",
        currency="USD",
    )
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="1",
                probabilities=(CategoryChoiceProbability("1", 0.86),),
            ),
        ]
    )

    categorize_transactions(
        [transaction],
        client=client,
        categories={
            "Education": {"Classes & Lessons": {}},
            "Transfers": {"Bank Transfers": {}},
        },
    )

    assert "Raw transaction description:   Maia April lessons" in client.prompts[0]
    assert "Descriptive Withdrawal P2P Transfer" not in client.prompts[0]


def test_categorize_transactions_strips_ach_debit_from_prompt():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        description="ACH Debit Electric Company",
        amount="-92.14",
        currency="USD",
    )
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="1",
                probabilities=(CategoryChoiceProbability("1", 0.86),),
            ),
        ]
    )

    categorize_transactions(
        [transaction],
        client=client,
        categories={
            "Housing & Utilities": {"Electricity": {}},
            "Transfers": {"Bank Transfers": {}},
        },
    )

    assert "Raw transaction description:  Electric Company" in client.prompts[0]
    assert "ACH Debit" not in client.prompts[0]


def test_categorize_transactions_uses_single_character_aliases_for_budget_categories():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        description="Kacey Dogsitting",
        amount="-125.00",
        currency="USD",
    )
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="f",
                probabilities=(CategoryChoiceProbability("f", 0.91),),
            ),
        ]
    )

    categorize_transactions(
        [transaction],
        client=client,
        categories=load_budget_categories(),
    )

    allocations = transaction.category_allocations
    assert allocations is not None
    assert len(allocations) == 1
    assert allocations[0].category_id == "Pets & Family > Childcare & Daycare"
    assert "f: Pets & Family > Childcare & Daycare" in client.prompts[0]
    assert "41: Pets & Family > Childcare & Daycare" not in client.prompts[0]


def test_categorize_transactions_rejects_too_many_single_token_alias_choices():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        description="Kacey Dogsitting",
        amount="-125.00",
        currency="USD",
    )
    categories: dict[str, object] = {
        "Budget": {
            f"Category {index}": {}
            for index in range(1, MAX_TRANSACTION_CATEGORY_ALIAS_CHOICES + 2)
        }
    }
    client = FakeProbabilityCategoryClient([])

    with pytest.raises(
        RuntimeError,
        match=(
            "too many budget category choices for single-token alias prompt: "
            f"got {MAX_TRANSACTION_CATEGORY_ALIAS_CHOICES + 1}, "
            f"maximum is {MAX_TRANSACTION_CATEGORY_ALIAS_CHOICES}"
        ),
    ):
        categorize_transactions([transaction], client=client, categories=categories)

    assert client.prompts == []


def test_categorize_transactions_skips_low_confidence_response(caplog: pytest.LogCaptureFixture):
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="UNKNOWN PAYEE",
        description="ACH WEB PMT 123",
        amount="-42.19",
        currency="USD",
    )
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="1",
                probabilities=(CategoryChoiceProbability("1", 0.2),),
            ),
        ]
    )

    caplog.set_level(logging.INFO, logger="receipts_ai.categorization")

    categorize_transactions(
        [transaction],
        client=client,
        categories={"Food & Dining": {"Groceries": {}}},
    )

    assert transaction.category_allocations == []
    assert len(client.prompts) == 1
    assert (
        "Skipping transaction bank_statement_1 because category confidence was low: "
        "description='ACH WEB PMT 123' ollama_response='1' "
        "ollama_choices=Food & Dining > Groceries (0.200)"
    ) in caplog.text


def test_categorize_transactions_skips_invalid_single_character_response():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="PUGET SOUND ENER",
        description="ACH Debit PUGET SOUND ENER BILLPAY",
        amount="-96.00",
        currency="USD",
    )
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="H",
                probabilities=(CategoryChoiceProbability("H", 1.0),),
            ),
        ]
    )

    categorize_transactions(
        [transaction],
        client=client,
        categories={"Housing & Utilities": {"Electricity": {}}, "Food & Dining": {"Groceries": {}}},
    )

    assert transaction.category_allocations == []
    assert "1: Housing & Utilities > Electricity" in client.prompts[0]


def test_classify_receipt_items_by_product_taxonomy_walks_each_level():
    item = ReceiptItem(
        description="Nabisco Premium Saltine Crackers",
        raw_description="RAW CONFUSING CODE",
        amount="4.49",
        taxonomy4="Stale Value",
        brave_search_result=json.dumps(
            [
                {
                    "title": "Premium Saltines",
                    "description": "Crisp crackers sold in grocery stores.",
                },
                {"title": "Result 2", "description": "Description 2"},
                {"title": "Result 3", "description": "Description 3"},
                {"title": "Result 4", "description": "Description 4"},
                {"title": "Result 5", "description": "Description 5"},
                {"title": "Result 6", "description": "Description 6"},
            ]
        ),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="FredMeyer",
        amount="-4.49",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    taxonomy: dict[str, object] = {
        "Animals & Pet Supplies": {},
        "Food, Beverages & Tobacco": {
            "Food Items": {
                "Bakery": {
                    "Crackers": {},
                    "Cakes": {},
                },
                "Candy": {},
            },
            "Beverages": {},
        },
    }
    client = FakeCategoryClient(["Food, Beverages & Tobacco", "Food Items", "Bakery", "Crackers"])

    result = classify_receipt_items_by_product_taxonomy(
        transaction, client=client, taxonomy=taxonomy
    )

    assert result is transaction
    assert item.taxonomy1 == "Food, Beverages & Tobacco"
    assert item.taxonomy2 == "Food Items"
    assert item.taxonomy3 == "Bakery"
    assert item.taxonomy4 == "Crackers"
    assert item.taxonomy5 is None
    assert len(client.prompts) == 4
    assert "pick the most appropriate product type" in client.prompts[0]
    assert "Animals & Pet Supplies" in client.prompts[0]
    assert "Food Items" not in client.prompts[0]
    assert "Selected product taxonomy path: Food, Beverages & Tobacco" in client.prompts[1]
    assert "Food Items" in client.prompts[1]
    assert "Bakery" in client.prompts[2]
    assert "Crackers" in client.prompts[3]
    assert "Receipt item description: Nabisco Premium Saltine Crackers" in client.prompts[0]
    assert "Receipt item description: Nabisco Premium Saltine Crackers" in client.prompts[3]
    assert "Premium Saltines" not in "\n".join(client.prompts)
    assert "Crisp crackers sold in grocery stores." not in "\n".join(client.prompts)
    assert "Result 5" not in "\n".join(client.prompts)
    assert "Result 6" not in "\n".join(client.prompts)
    assert "RAW CONFUSING CODE" not in "\n".join(client.prompts)


def test_classify_receipt_items_by_product_taxonomy_stops_when_model_repeats_parent():
    item = ReceiptItem(
        description="TurboTax software",
        amount="25.00",
        brave_search_result=json.dumps(
            [{"title": "TurboTax Software", "description": "Tax software sold at Costco."}]
        ),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Costco",
        amount="-25.00",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    taxonomy: dict[str, object] = {
        "Electronics": {
            "Audio": {},
            "Computers": {},
            "Video": {},
        },
    }
    client = FakeCategoryClient(["Electronics", "Electronics"])

    classify_receipt_items_by_product_taxonomy(transaction, client=client, taxonomy=taxonomy)

    assert item.taxonomy1 == "Electronics"
    assert item.taxonomy2 is None
    assert len(client.prompts) == 2
    assert "Audio" in client.prompts[1]


def test_classify_receipt_items_by_product_taxonomy_stops_at_parent_on_low_probability():
    item = ReceiptItem(
        description="Chocolate pastry",
        amount="5.00",
        brave_search_result=json.dumps(
            [{"title": "Chocolate pastry", "description": "Bakery item with chocolate."}]
        ),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Bakery",
        amount="-5.00",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    taxonomy: dict[str, object] = {
        "Food, Beverages & Tobacco": {
            "Food Items": {
                "Bakery": {
                    "Cakes": {},
                    "Crackers": {},
                },
            },
        },
    }
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="Food, Beverages & Tobacco",
                probabilities=(CategoryChoiceProbability("Food, Beverages & Tobacco", 0.90),),
            ),
            CategoryCompletion(
                response="Food Items",
                probabilities=(CategoryChoiceProbability("Food Items", 0.80),),
            ),
            CategoryCompletion(
                response="Bakery",
                probabilities=(CategoryChoiceProbability("Bakery", 0.70),),
            ),
            CategoryCompletion(
                response="Cakes",
                probabilities=(
                    CategoryChoiceProbability("Cakes", 0.25),
                    CategoryChoiceProbability("Crackers", 0.20),
                ),
            ),
        ]
    )

    classify_receipt_items_by_product_taxonomy(transaction, client=client, taxonomy=taxonomy)

    assert item.taxonomy1 == "Food, Beverages & Tobacco"
    assert item.taxonomy2 == "Food Items"
    assert item.taxonomy3 == "Bakery"
    assert item.taxonomy4 is None
    assert len(client.prompts) == 4


def test_classify_receipt_items_by_product_taxonomy_maps_single_token_aliases():
    item = ReceiptItem(
        description="Organic Gala Apples 3 lbs",
        amount="5.99",
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Grocery",
        amount="-5.99",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    taxonomy: dict[str, object] = {
        "Bakery": {},
        "Candied & Chocolate Covered Fruit": {},
        "Candy & Chocolate": {},
        "Condiments & Sauces": {},
        "Cooking & Baking Ingredients": {},
        "Dairy Products": {},
        "Dips & Spreads": {},
        "Food Gift Baskets": {},
        "Frozen Desserts & Novelties": {},
        "Fruits & Vegetables": {},
    }
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="J",
                probabilities=(
                    CategoryChoiceProbability("J", 1.0),
                    CategoryChoiceProbability("I", 0.0),
                ),
            ),
        ]
    )

    classify_receipt_items_by_product_taxonomy(transaction, client=client, taxonomy=taxonomy)

    assert item.taxonomy1 == "Fruits & Vegetables"
    assert "I: Frozen Desserts & Novelties" in client.prompts[0]
    assert "J: Fruits & Vegetables" in client.prompts[0]
    assert client.prompts[0].endswith("Label: ")


def test_classify_receipt_items_by_product_taxonomy_rejects_too_many_single_token_alias_choices():
    item = ReceiptItem(
        description="Organic Gala Apples 3 lbs",
        amount="5.99",
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Grocery",
        amount="-5.99",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    taxonomy: dict[str, object] = {
        f"Category {index}": {} for index in range(1, MAX_TAXONOMY_ALIAS_CHOICES + 2)
    }
    client = FakeProbabilityCategoryClient([])

    with pytest.raises(
        RuntimeError,
        match=(
            "too many taxonomy choices for single-token alias prompt: "
            f"got {MAX_TAXONOMY_ALIAS_CHOICES + 1}, maximum is {MAX_TAXONOMY_ALIAS_CHOICES}"
        ),
    ):
        classify_receipt_items_by_product_taxonomy(transaction, client=client, taxonomy=taxonomy)

    assert client.prompts == []


def test_classify_receipt_items_by_product_taxonomy_searches_multiple_probable_paths():
    item = ReceiptItem(
        description="Sparkling cider",
        amount="6.00",
        brave_search_result=json.dumps(
            [{"title": "Sparkling cider", "description": "Non-alcoholic bottled drink."}]
        ),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Grocery",
        amount="-6.00",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    taxonomy: dict[str, object] = {
        "Food, Beverages & Tobacco": {
            "Food Items": {"Fruit": {}},
            "Beverages": {"Juice": {}},
        },
    }
    client = FakeProbabilityCategoryClient(
        [
            CategoryCompletion(
                response="Food, Beverages & Tobacco",
                probabilities=(CategoryChoiceProbability("Food, Beverages & Tobacco", 0.90),),
            ),
            CategoryCompletion(
                response="Food Items",
                probabilities=(
                    CategoryChoiceProbability("Food Items", 0.52),
                    CategoryChoiceProbability("Beverages", 0.41),
                ),
            ),
            CategoryCompletion(
                response="Fruit",
                probabilities=(CategoryChoiceProbability("Fruit", 0.30),),
            ),
            CategoryCompletion(
                response="Juice",
                probabilities=(CategoryChoiceProbability("Juice", 0.90),),
            ),
        ]
    )

    classify_receipt_items_by_product_taxonomy(transaction, client=client, taxonomy=taxonomy)

    assert item.taxonomy1 == "Food, Beverages & Tobacco"
    assert item.taxonomy2 == "Beverages"
    assert item.taxonomy3 == "Juice"
    assert len(client.prompts) == 4
    assert (
        "Selected product taxonomy path: Food, Beverages & Tobacco > Beverages" in client.prompts[3]
    )


def test_classify_receipt_items_by_product_taxonomy_vector_search_ranks_nearest_paths():
    item = ReceiptItem(
        description="Apple AirPods Pro 3",
        amount="249.00",
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Apple",
        amount="-249.00",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    taxonomy_embeddings = TaxonomyEmbeddingIndex(
        embedding_model="test-model",
        embedding_dimension=2,
        entries=(
            TaxonomyEmbeddingEntry(
                path=("Food, Beverages & Tobacco", "Food Items", "Bakery", "Crackers"),
                embedding=(0.0, 1.0),
            ),
            TaxonomyEmbeddingEntry(
                path=("Electronics", "Audio", "Speakers"),
                embedding=(0.8, 0.2),
            ),
            TaxonomyEmbeddingEntry(
                path=("Electronics", "Audio", "Headphones"),
                embedding=(1.0, 0.0),
            ),
        ),
    )
    embedding_client = FakeTaxonomyEmbeddingClient((1.0, 0.0))
    client = FakeCategoryClient(["Electronics > Audio > Headphones"])

    result = classify_receipt_items_by_product_taxonomy_vector_search(
        transaction,
        client=client,
        embedding_client=embedding_client,
        taxonomy_embeddings=taxonomy_embeddings,
        candidate_count=2,
    )

    assert result is transaction
    assert item.taxonomy1 == "Electronics"
    assert item.taxonomy2 == "Audio"
    assert item.taxonomy3 == "Headphones"
    assert item.taxonomy4 is None
    assert embedding_client.texts == ["Apple AirPods Pro 3"]
    assert len(client.prompts) == 1
    assert "Choose the correct Google Product Category path" in client.prompts[0]
    assert "Electronics > Audio > Headphones" in client.prompts[0]
    assert "Electronics > Audio > Speakers" in client.prompts[0]
    assert "Food, Beverages & Tobacco > Food Items > Bakery > Crackers" not in client.prompts[0]
    assert "Receipt item description: Apple AirPods Pro 3" in client.prompts[0]


def test_product_taxonomy_prompts_do_not_include_transaction_mcc_description():
    item = ReceiptItem(
        description="Apple AirPods Pro 3",
        amount="249.00",
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Apple",
        mcc="5732",
        mcc_description="Electronics Stores",
        amount="-249.00",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    taxonomy: dict[str, object] = {
        "Electronics": {"Audio": {"Headphones": {}}},
        "Food, Beverages & Tobacco": {},
    }
    client = FakeCategoryClient(["Electronics", "Audio", "Headphones"])

    classify_receipt_items_by_product_taxonomy(transaction, client=client, taxonomy=taxonomy)

    assert "Merchant category" not in "\n".join(client.prompts)
    assert "Electronics Stores" not in "\n".join(client.prompts)


def test_categorize_receipt_items_rejects_non_leaf_category_response():
    item = ReceiptItem(
        description="Coffee",
        amount="7.00",
        brave_search_result=json.dumps([{"title": "Coffee drink", "description": "Cafe item"}]),
    )
    transaction = Transaction(
        id="receipt_1",
        source=Source.receipt,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
        receipt=Receipt(items=[item]),
    )
    client = FakeCategoryClient(["Food & Dining", "Food & Dining"])

    with pytest.raises(RuntimeError, match="invalid category"):
        categorize_receipt_items(transaction, client=client)


def test_create_ollama_category_client_uses_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA_URL", "http://example.test:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "45.5")

    client = create_ollama_category_client()

    assert isinstance(client, UrlLibOllamaClient)
    assert client.url == "http://example.test:11434"
    assert client.generate_url == "http://example.test:11434/api/generate"
    assert client.model == "llama3.2"
    assert client.timeout_seconds == 45.5


def test_create_ollama_category_client_defaults_url_and_requires_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL_NAME", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)

    with pytest.raises(RuntimeError, match="OLLAMA_MODEL"):
        create_ollama_category_client()

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    client = create_ollama_category_client()
    assert isinstance(client, UrlLibOllamaClient)
    assert client.url == DEFAULT_OLLAMA_URL
    assert client.generate_url == f"{DEFAULT_OLLAMA_URL}/api/generate"
    assert client.timeout_seconds == 30.0


def test_create_ollama_category_client_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "not-a-number")

    with pytest.raises(RuntimeError, match="OLLAMA_TIMEOUT_SECONDS must be a number"):
        create_ollama_category_client()

    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "0")

    with pytest.raises(RuntimeError, match="OLLAMA_TIMEOUT_SECONDS must be greater than 0"):
        create_ollama_category_client()


def test_url_lib_ollama_client_posts_generate_request(monkeypatch: pytest.MonkeyPatch):
    requests: list[tuple[urllib.request.Request, float]] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse({"response": "Groceries\n"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(
        url="http://example.test:11434/", model="llama3.2", timeout_seconds=3.0
    )

    result = client.complete("Choose one")

    assert result == "Groceries\n"
    request, timeout = requests[0]
    assert timeout == 3.0
    assert request.full_url == "http://example.test:11434/api/generate"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    request_data = request.data
    assert isinstance(request_data, bytes)
    assert json.loads(request_data) == {
        "model": "llama3.2",
        "prompt": "Choose one",
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 64, "stop": ["\n"]},
    }


def test_url_lib_ollama_client_can_request_choice_schema(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == 30.0
        requests.append(request)
        return FakeResponse({"response": '{"category":"Groceries"}'})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="qwen3")

    result = client.complete_choice(
        "Choose one",
        choices=("Groceries", "Restaurants & Dining Out"),
    )

    assert result == "Groceries"
    request_data = requests[0].data
    assert isinstance(request_data, bytes)
    payload = json.loads(request_data)
    assert payload == {
        "model": "qwen3",
        "prompt": (
            "Choose one\n\n"
            "Return only JSON matching this schema. Do not include explanation text.\n"
            '{"type": "object", "properties": {"category": {"type": "string", '
            '"enum": ["Groceries", "Restaurants & Dining Out"]}}, '
            '"required": ["category"], "additionalProperties": false}'
        ),
        "stream": False,
        "think": False,
        "format": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Groceries", "Restaurants & Dining Out"],
                }
            },
            "required": ["category"],
            "additionalProperties": False,
        },
        "options": {"temperature": 0},
    }


def test_url_lib_ollama_client_can_request_choice_probabilities(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == 30.0
        requests.append(request)
        return FakeResponse(
            {
                "response": "Groceries",
                "logprobs": {
                    "top_logprobs": [
                        {"token": "Groceries", "logprob": -0.2},
                        {"token": "Dining", "logprob": -1.2},
                    ]
                },
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    completion = client.complete_with_probabilities("Choose one", choices=("Groceries", "Dining"))

    assert completion.response == "Groceries"
    assert len(completion.probabilities) == 2
    assert completion.probabilities[0].choice == "Groceries"
    assert math.isclose(completion.probabilities[0].probability, 0.8187307530779818)
    assert completion.probabilities[1].choice == "Dining"
    assert math.isclose(completion.probabilities[1].probability, 0.30119421191220214)
    request_data = requests[0].data
    assert isinstance(request_data, bytes)
    assert json.loads(request_data) == {
        "model": "llama3.2",
        "prompt": "Choose one",
        "stream": False,
        "think": False,
        "logprobs": True,
        "top_logprobs": 5,
        "options": {"temperature": 0, "num_predict": 1, "logprobs": True, "top_logprobs": 5},
    }


def test_url_lib_ollama_client_uses_first_token_logprobs_for_choice_probabilities(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        _ = request
        _ = timeout
        return FakeResponse(
            {
                "response": "Food Items",
                "logprobs": {
                    "content": [
                        {
                            "token": "Food",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "Food", "logprob": -0.2},
                                {"token": "Beverages", "logprob": -1.5},
                            ],
                        },
                        {
                            "token": " Items",
                            "logprob": -0.01,
                            "top_logprobs": [
                                {"token": "Beverages", "logprob": -0.01},
                                {"token": " Items", "logprob": -0.01},
                            ],
                        },
                    ]
                },
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    completion = client.complete_with_probabilities(
        "Choose one", choices=("Food Items", "Beverages")
    )

    assert [result.choice for result in completion.probabilities] == ["Food Items", "Beverages"]
    assert math.isclose(completion.probabilities[0].probability, math.exp(-0.2))
    assert math.isclose(completion.probabilities[1].probability, math.exp(-1.5))


def test_url_lib_ollama_client_keeps_alias_choice_case_distinct(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        _ = request
        _ = timeout
        return FakeResponse(
            {
                "response": "k",
                "logprobs": {
                    "top_logprobs": [
                        {"token": "k", "logprob": -0.1},
                        {"token": "K", "logprob": -2.0},
                    ]
                },
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    completion = client.complete_with_probabilities("Choose one", choices=("K", "k"))

    assert [result.choice for result in completion.probabilities] == ["k", "K"]
    assert math.isclose(completion.probabilities[0].probability, math.exp(-0.1))
    assert math.isclose(completion.probabilities[1].probability, math.exp(-2.0))


def test_cached_category_model_client_reuses_sqlite_cache(tmp_path: Path):
    cache_path = tmp_path / "api-cache.sqlite"
    first_client = FakeCategoryClient(["Groceries"])
    cached_client = CachedCategoryModelClient(client=first_client, cache=SqliteCallCache(cache_path))

    first_result = cached_client.complete("Choose one")

    second_client = FakeCategoryClient(["Should not be used"])
    second_cached_client = CachedCategoryModelClient(
        client=second_client, cache=SqliteCallCache(cache_path)
    )
    second_result = second_cached_client.complete("Choose one")

    assert first_result == second_result == "Groceries"
    assert first_client.prompts == ["Choose one"]
    assert second_client.prompts == []
    assert SqliteCallCache(cache_path).get("ollama", {"prompt": "Choose one"}) == "Groceries"


def test_cached_category_model_client_reuses_choice_response(tmp_path: Path):
    cache_path = tmp_path / "api-cache.sqlite"
    first_client = FakeCategoryClient(["Groceries"])
    cached_client = CachedCategoryModelClient(client=first_client, cache=SqliteCallCache(cache_path))

    first_result = cached_client.complete_choice(
        "Choose one", choices=("Groceries", "Restaurants & Dining Out")
    )

    second_client = FakeCategoryClient(["Should not be used"])
    second_cached_client = CachedCategoryModelClient(
        client=second_client, cache=SqliteCallCache(cache_path)
    )
    second_result = second_cached_client.complete_choice(
        "Choose one", choices=("Groceries", "Restaurants & Dining Out")
    )

    assert first_result == second_result == "Groceries"
    assert first_client.prompts == ["Choose one"]
    assert second_client.prompts == []
    assert (
        SqliteCallCache(cache_path).get(
            "ollama",
            {
                "prompt": "Choose one",
                "choices": ["Groceries", "Restaurants & Dining Out"],
                "format": "category_choice_schema_v1",
            },
        )
        == "Groceries"
    )


def test_cached_category_model_client_does_not_create_client_on_cache_hit(tmp_path: Path):
    cache_path = tmp_path / "api-cache.sqlite"
    cache = SqliteCallCache(cache_path)
    cache.set("ollama", {"prompt": "Choose one"}, "Groceries")
    factory_calls = 0

    def client_factory() -> FakeCategoryClient:
        nonlocal factory_calls
        factory_calls += 1
        return FakeCategoryClient(["Should not be used"])

    client = CachedCategoryModelClient(
        cache=SqliteCallCache(cache_path), client_factory=client_factory
    )

    result = client.complete("Choose one")

    assert result == "Groceries"
    assert factory_calls == 0


def test_url_lib_ollama_client_accepts_generate_endpoint_url(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == 30.0
        requests.append(request)
        return FakeResponse({"response": "Groceries"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434/api/generate", model="llama3.2")

    client.complete("Choose one")

    assert client.generate_url == "http://example.test:11434/api/generate"
    assert requests[0].full_url == "http://example.test:11434/api/generate"


def test_url_lib_ollama_client_logs_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert request.full_url == "http://example.test:11434/api/generate"
        assert timeout == 30.0
        return FakeResponse({"response": "Groceries"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    with caplog.at_level("INFO", logger="receipts_ai.categorization"):
        client.complete("Choose one")

    assert (
        "Sending Ollama generate request: url=http://example.test:11434/api/generate model=llama3.2"
    ) in caplog.text
    assert (
        "Received Ollama generate response: url=http://example.test:11434/api/generate "
        "model=llama3.2"
    ) in caplog.text
    assert "timeout_seconds=30.0" in caplog.text
    assert "stream=False" in caplog.text
    assert "think=False" in caplog.text
    assert "format=unset" in caplog.text
    assert "logprobs=unset" in caplog.text
    assert "top_logprobs=unset" in caplog.text
    assert 'options={"num_predict":64,"stop":["\\n"],"temperature":0}' in caplog.text
    assert "payload_bytes=" in caplog.text


def test_url_lib_ollama_client_debug_logs_generate_stats(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert request.full_url == "http://example.test:11434/api/generate"
        assert timeout == 30.0
        return FakeResponse(
            {
                "response": "Groceries",
                "total_duration": 11_500_000_000,
                "load_duration": 500_000_000,
                "prompt_eval_count": 42,
                "prompt_eval_duration": 2_000_000_000,
                "eval_count": 9,
                "eval_duration": 3_000_000_000,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    with caplog.at_level("DEBUG", logger="receipts_ai.categorization"):
        client.complete("Choose one")

    assert (
        "Ollama generate stats: url=http://example.test:11434/api/generate "
        "model=llama3.2 total=11.500s load=0.500s prompt_eval=2.000s "
        "eval=3.000s prompt_tokens=42 eval_tokens=9 eval_tokens_per_second=3.00"
    ) in caplog.text


def test_url_lib_ollama_client_debug_logs_curl_reproduction_command(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert request.full_url == "http://example.test:11434/api/generate"
        assert timeout == 30.0
        return FakeResponse({"response": "Groceries"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    with caplog.at_level("DEBUG", logger="receipts_ai.categorization"):
        client.complete("Choose one\nWith a second line")

    prefix = "Ollama generate curl reproduction command: "
    message = next(record.message for record in caplog.records if record.message.startswith(prefix))
    command_parts = shlex.split(message.removeprefix(prefix))

    assert command_parts == [
        "curl",
        "-sS",
        "http://example.test:11434/api/generate",
        "-H",
        "Accept: application/json",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(
            {
                "model": "llama3.2",
                "options": {"temperature": 0, "num_predict": 64, "stop": ["\n"]},
                "prompt": "Choose one\nWith a second line",
                "stream": False,
                "think": False,
            },
            sort_keys=True,
        ),
    ]


def test_url_lib_ollama_client_error_includes_endpoint(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == 30.0
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            Message(),
            BytesIO(b"404 page not found"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(url="http://example.test:11434", model="llama3.2")

    with pytest.raises(
        RuntimeError,
        match="Ollama request to http://example.test:11434/api/generate failed with HTTP 404",
    ):
        client.complete("Choose one")


def test_url_lib_ollama_client_timeout_error_includes_context(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_urlopen(_request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == 12.5
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = UrlLibOllamaClient(
        url="http://example.test:11434", model="llama3.2", timeout_seconds=12.5
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.complete("Choose one")

    message = str(exc_info.value)
    assert "Ollama request to http://example.test:11434/api/generate timed out" in message
    assert "model=llama3.2" in message
    assert "prompt_chars=10" in message
    assert "timeout_seconds=12.5" in message
