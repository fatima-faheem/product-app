import asyncio

import pytest

from gemini_helper import (
    AIError,
    AITimeoutError,
    AIBlockedError,
    AIUpstreamError,
    AIQuotaError
)
from helpers import (
    count_products,
    fake_gemini,
    fake_gemini_error,
    get_product_by_name,
    seed_products,
    product_data,
)


# ---------------------------------------------------------------- helpers
def make_payload(remove=(), **overrides):
    """A valid manual-description product. `remove` drops keys entirely."""
    data = {
        "name": "Cotton T-Shirt",
        "description": "A soft cotton T-shirt.",
        "category": "Clothing",
        "price": 1200,
        "stock": 10,
        "rating": 4.5,
    }
    data.update(overrides)
    for key in remove:
        data.pop(key, None)
    return data


def make_ai_payload(**overrides):
    """A valid product that asks for an AI description (no manual one)."""
    return make_payload(
        remove=("description",),
        ai_description=True,
        **overrides,
    )


# ============================================================ MANUAL: success
async def test_create_product_manual_description(client):
    response = await client.post("/products", json=[make_payload()])

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert isinstance(data[0]["id"], int)
    assert data[0]["name"] == "Cotton T-Shirt"
    assert data[0]["description"] == "A soft cotton T-shirt."
    assert data[0]["category"] == "Clothing"
    assert data[0]["price"] == 1200
    assert data[0]["stock"] == 10
    assert data[0]["rating"] == 4.5

    row = await get_product_by_name("Cotton T-Shirt")
    assert row is not None
    assert row["description"] == "A soft cotton T-shirt."
    assert float(row["price"]) == 1200


async def test_create_multiple_products(client):
    payload = [
        make_payload(name="Product A", price=100),
        make_payload(name="Product B", price=200),
        make_payload(name="Product C", price=300),
    ]

    response = await client.post("/products", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert {p["name"] for p in data} == {"Product A", "Product B", "Product C"}
    assert len({p["id"] for p in data}) == 3
    assert await count_products() == 3


async def test_defaults_when_category_and_rating_omitted(client):
    payload = make_payload(remove=("category", "rating"))

    response = await client.post("/products", json=[payload])

    assert response.status_code == 200
    item = response.json()[0]
    assert item["category"] == "Other"
    assert item["rating"] == 0


@pytest.mark.parametrize(
    "category,expected",
    [
        (None, "Other"),
        ("", "Other"),
        ("   ", "Other"),
        (" Clothing ", "Clothing"),
    ],
    ids=["null", "empty", "blank", "trimmed"],
)
async def test_category_is_normalized(client, category, expected):
    response = await client.post("/products", json=[make_payload(category=category)])

    assert response.status_code == 200
    assert response.json()[0]["category"] == expected
    assert (await get_product_by_name("Cotton T-Shirt"))["category"] == expected


async def test_price_zero_is_accepted(client):
    response = await client.post("/products", json=[make_payload(price=0)])

    assert response.status_code == 200
    assert response.json()[0]["price"] == 0
    assert float((await get_product_by_name("Cotton T-Shirt"))["price"]) == 0


async def test_rating_null_is_accepted(client):
    response = await client.post("/products", json=[make_payload(rating=None)])

    assert response.status_code == 200
    assert response.json()[0]["rating"] is None
    assert (await get_product_by_name("Cotton T-Shirt"))["rating"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "ab"},
        {"name": "x" * 100},
        {"stock": 0},
        {"rating": 0},
        {"rating": 5},
        {"price": 0},
    ],
    ids=["name-min", "name-max", "stock-0", "rating-0", "rating-5", "price-0"],
)
async def test_boundary_values_are_accepted(client, overrides):
    response = await client.post("/products", json=[make_payload(**overrides)])

    assert response.status_code == 200
    assert await count_products() == 1


# ======================================================= MANUAL: validation
INVALID = [
    pytest.param({"name": "A"}, (), id="name-too-short"),
    pytest.param({"name": "X" * 101}, (), id="name-too-long"),
    pytest.param({"name": None}, (), id="name-null"),
    pytest.param({"price": -1}, (), id="price-negative"),
    pytest.param({"price": "free"}, (), id="price-text"),
    pytest.param({"stock": -1}, (), id="stock-negative"),
    pytest.param({"stock": 1.5}, (), id="stock-fraction"),
    pytest.param({"rating": 5.1}, (), id="rating-too-high"),
    pytest.param({"rating": -0.1}, (), id="rating-too-low"),
    pytest.param({}, ("name",), id="missing-name"),
    pytest.param({}, ("price",), id="missing-price"),
    pytest.param({}, ("stock",), id="missing-stock"),
]


@pytest.mark.parametrize("overrides,remove", INVALID)
async def test_invalid_product_returns_422(client, overrides, remove):
    response = await client.post(
        "/products", json=[make_payload(remove=remove, **overrides)]
    )

    assert response.status_code == 422
    assert await count_products() == 0


async def test_one_invalid_product_rejects_whole_batch(client):
    payload = [make_payload(name="Valid One"), make_payload(name="Bad One", price=-1)]

    response = await client.post("/products", json=payload)

    assert response.status_code == 422
    assert await count_products() == 0


# ============================================================ description rules
async def test_manual_description_and_ai_flag_returns_422(client, monkeypatch):
    prompts = fake_gemini(monkeypatch)

    response = await client.post(
        "/products", json=[make_payload(ai_description=True)]
    )

    assert response.status_code == 422
    assert prompts == []
    assert await count_products() == 0


async def test_no_description_and_no_ai_flag_returns_422(client):
    response = await client.post(
        "/products", json=[make_payload(remove=("description",))]
    )

    assert response.status_code == 422
    assert await count_products() == 0


async def test_empty_description_with_ai_flag_returns_422(client, monkeypatch):
    prompts = fake_gemini(monkeypatch)

    response = await client.post(
        "/products", json=[make_payload(description="", ai_description=True)]
    )

    assert response.status_code == 422
    assert prompts == []


# ================================================================ bad bodies
async def test_empty_product_list_returns_400(client):
    response = await client.post("/products", json=[])

    assert response.status_code == 400
    assert response.json()["detail"] == "No products provided for insertion."


async def test_single_object_instead_of_list_returns_422(client):
    response = await client.post("/products", json=make_payload())

    assert response.status_code == 422


async def test_malformed_json_returns_422(client):
    response = await client.post(
        "/products",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


# ================================================================== database
async def test_duplicate_name_in_database_returns_409(client):
    await seed_products([product_data(name="Existing Product")])

    response = await client.post(
        "/products", json=[make_payload(name="Existing Product")]
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "One or more products already exist."
    assert await count_products() == 1  # only the seeded one


async def test_duplicate_name_inside_one_batch_returns_409(client):
    payload = [make_payload(name="Twin"), make_payload(name="Twin")]

    response = await client.post("/products", json=payload)

    assert response.status_code == 409
    assert await count_products() == 0  # nothing inserted


async def test_one_duplicate_in_batch_inserts_nothing(client):
    await seed_products([product_data(name="Existing Product")])
    payload = [
        make_payload(name="Brand New"),
        make_payload(name="Existing Product"),
    ]

    response = await client.post("/products", json=payload)

    assert response.status_code == 409
    assert await get_product_by_name("Brand New") is None


async def test_stock_overflow_returns_400(client):
    response = await client.post(
        "/products", json=[make_payload(stock=3_000_000_000)]
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid product data."
    assert await count_products() == 0


# =================================================================== AI: success
async def test_create_product_ai_description(client, monkeypatch):
    prompts = fake_gemini(monkeypatch, text="A comfortable cotton T-shirt.")

    response = await client.post("/products", json=[make_ai_payload()])

    assert response.status_code == 200
    assert response.json()[0]["description"] == "A comfortable cotton T-shirt."
    assert len(prompts) == 1
    assert (await get_product_by_name("Cotton T-Shirt"))["description"] == (
        "A comfortable cotton T-shirt."
    )


async def test_ai_prompt_contains_name_and_category(client, monkeypatch):
    prompts = fake_gemini(monkeypatch)

    await client.post(
        "/products",
        json=[make_ai_payload(name="Wireless Mouse", category="Electronics")],
    )

    assert "Wireless Mouse" in prompts[0]
    assert "Electronics" in prompts[0]


async def test_gemini_not_called_for_manual_description(client, monkeypatch):
    prompts = fake_gemini(monkeypatch)

    response = await client.post("/products", json=[make_payload()])

    assert response.status_code == 200
    assert prompts == []


async def test_mixed_batch_only_calls_gemini_for_ai_items(client, monkeypatch):
    prompts = fake_gemini(monkeypatch, text="AI text.")
    payload = [
        make_payload(name="Manual Item", description="Hand written"),
        make_ai_payload(name="AI Item"),
    ]

    response = await client.post("/products", json=payload)

    assert response.status_code == 200
    by_name = {p["name"]: p["description"] for p in response.json()}
    assert by_name == {"Manual Item": "Hand written", "AI Item": "AI text."}
    assert len(prompts) == 1
    assert "AI Item" in prompts[0]


async def test_bulk_ai_success(client, monkeypatch):
    async def fake_call_gemini(prompt):
        if "Product A" in prompt:
            return "Description for Product A."
        if "Product B" in prompt:
            return "Description for Product B."
        raise AssertionError("Unexpected product in prompt")

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)

    response = await client.post(
        "/products",
        json=[make_ai_payload(name="Product A"), make_ai_payload(name="Product B")],
    )

    assert response.status_code == 200
    returned = {p["name"]: p["description"] for p in response.json()}
    assert returned == {
        "Product A": "Description for Product A.",
        "Product B": "Description for Product B.",
    }
    assert (await get_product_by_name("Product A"))["description"] == (
        "Description for Product A."
    )
    assert (await get_product_by_name("Product B"))["description"] == (
        "Description for Product B."
    )


# ============================================================= AI: failures
@pytest.mark.parametrize(
    "exc,expected_status,expected_detail",
    [
        (AITimeoutError("Gemini timed out"), 504,
         "AI service timed out. Please try again."),
        (AIBlockedError("Request blocked"), 400,
         "AI request was blocked."),
        (AIUpstreamError("Gemini unavailable", code=503), 503,
         "AI service is temporarily unavailable."),
        (AIError("Unexpected AI error"), 500,
         "An unexpected AI error occurred."),
            (
        AIQuotaError(kind="daily"),
        503,
        "AI daily request quota exhausted. Please try again later.",
        ),
        (
        AIQuotaError(kind="rate"),
        503,
        "AI request rate limit reached. Please try again shortly.",
        ),
        (
        AIQuotaError(kind="unknown"),
        503,
        "AI service quota limit reached. Please try again later.",
        ),
    ],
    ids=[
    "timeout",
    "blocked",
    "upstream",
    "unexpected",
    "daily-quota",
    "rate-quota",
    "unknown-quota",
],
)
async def test_ai_failure_returns_right_error_and_inserts_nothing(
    client, monkeypatch, exc, expected_status, expected_detail
):
    fake_gemini_error(monkeypatch, exc)

    response = await client.post("/products", json=[make_ai_payload()])

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
    assert await count_products() == 0


async def test_bulk_ai_failure_inserts_nothing(client, monkeypatch):
    async def fake_call_gemini(prompt):
        if "Product B" in prompt:
            raise AIUpstreamError("Gemini service unavailable", code=503)
        return "A good product description."

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)

    response = await client.post(
        "/products",
        json=[make_ai_payload(name="Product A"), make_ai_payload(name="Product B")],
    )

    assert response.status_code == 503
    assert await count_products() == 0


async def test_ai_failure_does_not_insert_manual_items_either(client, monkeypatch):
    fake_gemini_error(monkeypatch, AITimeoutError("Gemini timed out"))
    payload = [make_payload(name="Manual Item"), make_ai_payload(name="AI Item")]

    response = await client.post("/products", json=payload)

    assert response.status_code == 504
    assert await count_products() == 0


async def test_ai_product_with_duplicate_name_returns_409(client, monkeypatch):
    await seed_products([product_data(name="Existing Product")])
    prompts = fake_gemini(monkeypatch)

    response = await client.post(
        "/products", json=[make_ai_payload(name="Existing Product")]
    )

    assert response.status_code == 409
    assert len(prompts) == 1  # the AI ran, then the insert was rejected
    assert await count_products() == 1


# ======================================================== task cancellation
async def test_failed_product_cancels_remaining_tasks(client, monkeypatch):
    a_started = asyncio.Event()
    a_cancelled = asyncio.Event()

    async def fake_process_product(product):
        if product.name == "Product A":
            a_started.set()
            try:
                await asyncio.Future()  # runs until cancelled
            except asyncio.CancelledError:
                a_cancelled.set()
                raise
        elif product.name == "Product B":
            await a_started.wait()
            raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr("main.process_product", fake_process_product)

    # a timeout so a bug here fails fast instead of hanging the whole suite
    async with asyncio.timeout(5):
        response = await client.post(
            "/products",
            json=[make_payload(name="Product A"), make_payload(name="Product B")],
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "An unexpected server error occurred."
    assert a_cancelled.is_set()
    assert await count_products() == 0
