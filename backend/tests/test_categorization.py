from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from datetime import date
from email.message import Message
from io import BytesIO
from pathlib import Path

import pytest

from receipts_ai.cache import JsonCallCache
from receipts_ai.categorization import (
    DEFAULT_OLLAMA_URL,
    CachedCategoryModelClient,
    CategoryChoiceProbability,
    CategoryCompletion,
    UrlLibOllamaClient,
    categorize_receipt_items,
    classify_receipt_items_by_product_taxonomy,
    clean_receipt_item_descriptions,
    create_ollama_category_client,
    load_budget_categories,
    load_product_taxonomy,
)
from receipts_ai.models.transaction import Receipt, ReceiptItem, Source, Transaction


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


def test_categorize_receipt_items_uses_clean_description_and_leaf_category_only():
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
    client = FakeCategoryClient(["Food & Dining", "Groceries"])

    result = categorize_receipt_items(transaction, client=client)

    assert result is transaction
    assert item.category_id == "Groceries"
    assert len(client.prompts) == 2
    assert "Food & Dining" in client.prompts[0]
    assert "Groceries" not in client.prompts[0]
    assert "Groceries" in client.prompts[1]
    assert "Food & Dining" not in client.prompts[1].split("Options:", maxsplit=1)[1]
    assert "Receipt item description: Nabisco Premium Saltine Crackers" in client.prompts[0]
    assert "Receipt item description: Nabisco Premium Saltine Crackers" in client.prompts[1]
    assert "Premium Saltines" not in "\n".join(client.prompts)
    assert "Crisp crackers sold in grocery stores." not in "\n".join(client.prompts)
    assert "RAW CONFUSING CODE" not in "\n".join(client.prompts)


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
):
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
        "logprobs": True,
        "top_logprobs": 5,
        "options": {"temperature": 0, "logprobs": True, "top_logprobs": 5},
    }


def test_cached_category_model_client_reuses_json_file(tmp_path: Path):
    cache_path = tmp_path / "api-cache.json"
    first_client = FakeCategoryClient(["Groceries"])
    cached_client = CachedCategoryModelClient(client=first_client, cache=JsonCallCache(cache_path))

    first_result = cached_client.complete("Choose one")

    second_client = FakeCategoryClient(["Should not be used"])
    second_cached_client = CachedCategoryModelClient(
        client=second_client, cache=JsonCallCache(cache_path)
    )
    second_result = second_cached_client.complete("Choose one")

    assert first_result == second_result == "Groceries"
    assert first_client.prompts == ["Choose one"]
    assert second_client.prompts == []
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["ollama"][0]["request"] == {"prompt": "Choose one"}
    assert payload["ollama"][0]["response"] == "Groceries"


def test_cached_category_model_client_does_not_create_client_on_cache_hit(tmp_path: Path):
    cache_path = tmp_path / "api-cache.json"
    cache = JsonCallCache(cache_path)
    cache.set("ollama", {"prompt": "Choose one"}, "Groceries")
    factory_calls = 0

    def client_factory() -> FakeCategoryClient:
        nonlocal factory_calls
        factory_calls += 1
        return FakeCategoryClient(["Should not be used"])

    client = CachedCategoryModelClient(
        cache=JsonCallCache(cache_path), client_factory=client_factory
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
