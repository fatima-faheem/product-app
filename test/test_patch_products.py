import pytest

from gemini_helper import (
    AIError,
    AITimeoutError,
    AIBlockedError,
    AIUpstreamError,
    AIQuotaError,
)
from helpers import (
    fake_gemini,
    fake_gemini_error,
    get_product_from_db,
    product_data,
    seed_products,
)


# ================================================================ manual PATCH
async def test_patch_product_single_field(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}", json={"price": 2000}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["price"] == 2000
    assert data["name"] == original["name"]
    assert data["description"] == original["description"]
    assert data["category"] == original["category"]
    assert data["stock"] == original["stock"]
    assert data["rating"] == original["rating"]

    assert (await get_product_from_db(original["id"]))["price"] == 2000


async def test_patch_product_multiple_fields(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={"name": "Updated Product", "price": 2500, "stock": 20},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Product"
    assert data["price"] == 2500
    assert data["stock"] == 20
    assert data["description"] == original["description"]
    assert data["category"] == original["category"]
    assert data["rating"] == original["rating"]

    db_product = await get_product_from_db(original["id"])
    assert db_product["name"] == "Updated Product"
    assert db_product["price"] == 2500
    assert db_product["stock"] == 20


async def test_patch_product_does_not_modify_other_products(client):
    product_a, product_b = await seed_products([
        product_data(name="Product A", price=1000),
        product_data(name="Product B", price=2000),
    ])

    response = await client.patch(
        f"/products/{product_a['id']}", json={"price": 5000}
    )

    assert response.status_code == 200
    assert (await get_product_from_db(product_a["id"]))["price"] == 5000
    assert await get_product_from_db(product_b["id"]) == product_b


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "New Product Name"),
        ("description", "New description"),
        ("category", "Clothing"),
        ("price", 2500),
        ("stock", 50),
        ("rating", 4.9),
    ],
)
async def test_patch_product_each_field(client, field, value):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}", json={field: value}
    )

    assert response.status_code == 200
    updated = response.json()
    for key, original_value in original.items():
        if key == field:
            assert updated[key] == value
        else:
            assert updated[key] == original_value


@pytest.mark.parametrize(
    "field,value",
    [("stock", 0), ("rating", 0), ("rating", 5), ("price", 0)],
    ids=["stock-0", "rating-0", "rating-5", "price-0"],
)
async def test_patch_product_boundary_values(client, field, value):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}", json={field: value}
    )

    assert response.status_code == 200
    assert response.json()[field] == value


async def test_patch_product_same_value(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={"price": float(original["price"])},
    )

    assert response.status_code == 200
    assert response.json()["price"] == original["price"]


async def test_patch_product_rating_null_is_allowed(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}", json={"rating": None}
    )

    assert response.status_code == 200
    assert response.json()["rating"] is None
    assert (await get_product_from_db(original["id"]))["rating"] is None


@pytest.mark.parametrize("category", [None, "", "   "], ids=["null", "empty", "blank"])
async def test_patch_product_null_or_blank_category_becomes_other(client, category):
    original = (await seed_products([product_data(category="Electronics")]))[0]

    response = await client.patch(
        f"/products/{original['id']}", json={"category": category}
    )

    assert response.status_code == 200
    assert response.json()["category"] == "Other"
    assert (await get_product_from_db(original["id"]))["category"] == "Other"


@pytest.mark.parametrize(
    "field", ["name", "description", "price", "stock"]
)
async def test_patch_product_null_in_required_field_returns_422(client, field):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}", json={field: None}
    )

    assert response.status_code == 422
    assert await get_product_from_db(original["id"]) == original


# ------------------------------------------------------------- bad requests
async def test_patch_product_not_found(client):
    response = await client.patch("/products/999999", json={"price": 2000})

    assert response.status_code == 404
    assert response.json()["detail"] == "Product Not Found"


async def test_patch_product_invalid_id(client):
    response = await client.patch("/products/abc", json={"price": 2000})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload", [{}, {"unknown_field": "something"}], ids=["empty", "unknown-field"]
)
async def test_patch_product_empty_or_unknown_payload(client, payload):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(f"/products/{original['id']}", json=payload)

    assert response.status_code == 422
    assert await get_product_from_db(original["id"]) == original


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "A"},
        {"name": "X" * 101},
        {"price": -0.01},
        {"price": -100},
        {"price": "free"},
        {"stock": -1},
        {"stock": 1.5},
        {"rating": 5.1},
        {"rating": -0.1},
    ],
    ids=[
        "name-short", "name-long", "price-tiny-negative", "price-negative",
        "price-text", "stock-negative", "stock-fraction", "rating-high",
        "rating-low",
    ],
)
async def test_patch_product_invalid_values(client, payload):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(f"/products/{original['id']}", json=payload)

    assert response.status_code == 422
    assert await get_product_from_db(original["id"]) == original


async def test_patch_product_stock_overflow(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}", json={"stock": 3_000_000_000}
    )

    assert response.status_code == 400
    assert await get_product_from_db(original["id"]) == original


async def test_patch_product_duplicate_name(client):
    product_a, product_b = await seed_products([
        product_data(name="Product A"),
        product_data(name="Product B"),
    ])

    response = await client.patch(
        f"/products/{product_b['id']}", json={"name": product_a["name"]}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "A product with this name already exists"
    assert await get_product_from_db(product_b["id"]) == product_b


# ====================================================== regenerate_description
async def test_patch_regenerate_description_success(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}", json={"regenerate_description": True}
    )

    assert response.status_code == 200
    assert response.json()["description"] == "Fresh AI description."
    assert len(prompts) == 1
    assert (await get_product_from_db(original["id"]))["description"] == (
        "Fresh AI description."
    )


async def test_patch_regenerate_keeps_other_fields(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    fake_gemini(monkeypatch)

    await client.patch(
        f"/products/{original['id']}", json={"regenerate_description": True}
    )

    db_product = await get_product_from_db(original["id"])
    for key, value in original.items():
        if key != "description":
            assert db_product[key] == value


async def test_patch_regenerate_prompt_uses_existing_values(client, monkeypatch):
    original = (await seed_products([
        product_data(name="Wireless Mouse", category="Electronics")
    ]))[0]
    prompts = fake_gemini(monkeypatch)

    await client.patch(
        f"/products/{original['id']}", json={"regenerate_description": True}
    )

    assert "Wireless Mouse" in prompts[0]
    assert "Electronics" in prompts[0]


async def test_patch_regenerate_prompt_uses_new_name(client, monkeypatch):
    original = (await seed_products([product_data(name="Wireless Mouse")]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"name": "Gaming Mouse", "regenerate_description": True},
    )

    assert response.status_code == 200
    assert "Gaming Mouse" in prompts[0]
    assert "Wireless Mouse" not in prompts[0]
    db_product = await get_product_from_db(original["id"])
    assert db_product["name"] == "Gaming Mouse"
    assert db_product["description"] == "Fresh AI description."


async def test_patch_regenerate_prompt_uses_new_category(client, monkeypatch):
    original = (await seed_products([product_data(category="Electronics")]))[0]
    prompts = fake_gemini(monkeypatch)

    await client.patch(
        f"/products/{original['id']}",
        json={"category": "Office", "regenerate_description": True},
    )

    assert "Office" in prompts[0]
    assert "Electronics" not in prompts[0]


async def test_patch_regenerate_null_category_uses_other(client, monkeypatch):
    original = (await seed_products([product_data(category="Electronics")]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"category": None, "regenerate_description": True},
    )

    assert response.status_code == 200
    assert "Other" in prompts[0]
    assert (await get_product_from_db(original["id"]))["category"] == "Other"


async def test_patch_regenerate_does_not_modify_other_products(client, monkeypatch):
    product_a, product_b = await seed_products([
        product_data(name="Product A"),
        product_data(name="Product B"),
    ])
    fake_gemini(monkeypatch)

    await client.patch(
        f"/products/{product_a['id']}", json={"regenerate_description": True}
    )

    assert await get_product_from_db(product_b["id"]) == product_b


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
        (AIQuotaError(kind="daily"), 503,
         "AI daily request quota exhausted. Please try again later."),
        (AIQuotaError(kind="rate"), 503,
         "AI request rate limit reached. Please try again shortly."),
        (AIQuotaError(kind="unknown"), 503,
         "AI service quota limit reached. Please try again later."),
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

async def test_patch_regenerate_ai_failure_leaves_row_unchanged(
    client, monkeypatch, exc, expected_status, expected_detail
):
    original = (await seed_products([product_data()]))[0]
    fake_gemini_error(monkeypatch, exc)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"name": "Renamed Product", "regenerate_description": True},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
    # not even the name from the same request may be saved
    assert await get_product_from_db(original["id"]) == original


@pytest.mark.parametrize("text", ["", "   "], ids=["empty", "blank"])
async def test_patch_regenerate_empty_ai_text_returns_502(client, monkeypatch, text):
    original = (await seed_products([product_data()]))[0]
    fake_gemini(monkeypatch, text=text)

    response = await client.patch(
        f"/products/{original['id']}", json={"regenerate_description": True}
    )

    assert response.status_code == 502
    assert await get_product_from_db(original["id"]) == original


async def test_patch_regenerate_duplicate_name_returns_409(client, monkeypatch):
    product_a, product_b = await seed_products([
        product_data(name="Product A"),
        product_data(name="Product B"),
    ])
    fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{product_b['id']}",
        json={"name": product_a["name"], "regenerate_description": True},
    )

    assert response.status_code == 409
    assert await get_product_from_db(product_b["id"]) == product_b


async def test_patch_regenerate_product_not_found(client, monkeypatch):
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        "/products/999999", json={"regenerate_description": True}
    )

    assert response.status_code == 404
    assert prompts == []


@pytest.mark.parametrize(
    "description",
    ["My own text", None],
    ids=["manual-description", "null-description"],
)
async def test_patch_regenerate_with_manual_description_returns_422(
    client, monkeypatch, description
):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={
            "description": description,
            "regenerate_description": True,
        },
    )

    assert response.status_code == 422
    assert prompts == []
    assert await get_product_from_db(original["id"]) == original

async def test_patch_regenerate_false_alone_returns_422(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}", json={"regenerate_description": False}
    )

    assert response.status_code == 422
    assert prompts == []
    assert await get_product_from_db(original["id"]) == original


async def test_patch_regenerate_with_null_name_returns_422(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"name": None, "regenerate_description": True},
    )

    assert response.status_code == 422
    assert prompts == []
    assert await get_product_from_db(original["id"]) == original


async def test_patch_without_flag_does_not_call_gemini(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}", json={"price": 2500}
    )

    assert response.status_code == 200
    assert response.json()["price"] == 2500
    assert prompts == []
    assert (await get_product_from_db(original["id"]))["description"] == (
        original["description"]
    )


async def test_patch_regenerate_false_with_other_field(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"stock": 99, "regenerate_description": False},
    )

    assert response.status_code == 200
    assert response.json()["stock"] == 99
    assert prompts == []
