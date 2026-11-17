from main import app
import pytest
import asyncio
from gemini_helper import (
    AIError,
    AITimeoutError,
    AIBlockedError,
    AIUpstreamError,
)
from helpers import product_data, seed_products, get_product_from_db

# Tests --> @app.patch("/products/{product_id}",response_model=Response)
# 1. Update one field and preserve the remaining fields
async def test_patch_product_single_field(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={"price": 2000}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["price"] == 2000
    assert data["name"] == original["name"]
    assert data["description"] == original["description"]
    assert data["category"] == original["category"]
    assert data["stock"] == original["stock"]
    assert data["rating"] == original["rating"]

    db_product = await get_product_from_db(original["id"])
    assert db_product["price"] == 2000


# 2. Update multiple fields together
async def test_patch_product_multiple_fields(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={
            "name": "Updated Product",
            "price": 2500,
            "stock": 20
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Updated Product"
    assert data["price"] == 2500
    assert data["stock"] == 20

    assert data["description"] == original["description"]
    assert data["category"] == original["category"]
    assert data["rating"] == original["rating"]
    
    # checking through database directly
    db_product = await get_product_from_db(original["id"])
    assert db_product["name"] == "Updated Product"
    assert db_product["price"] == 2500
    assert db_product["stock"] == 20


# 3. Ensure other products are not modified
async def test_patch_product_does_not_modify_other_products(client):
    products = await seed_products([
        product_data(name="Product A", price=1000),
        product_data(name="Product B", price=2000)
    ])

    product_a, product_b = products

    response = await client.patch(
        f"/products/{product_a['id']}",
        json={"price": 5000}
    )

    assert response.status_code == 200
    assert response.json()["price"] == 5000
    
    db_product_a = await get_product_from_db(product_a["id"])
    db_product_b = await get_product_from_db(product_b["id"])

    assert db_product_a["price"] == 5000
    assert db_product_b == product_b


# 4. Test every individual updateable field
@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "New Product Name"),
        ("description", "New description"),
        ("category", "Clothing"),
        ("price", 2500),
        ("stock", 50),
        ("rating", 4.9)
    ]
)
async def test_patch_product_each_field(client, field, value):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={field: value}
    )

    assert response.status_code == 200
    updated = response.json()
    for key, original_value in original.items():
        if key == field:
            assert updated[key] == value
        else:
            assert updated[key] == original_value

# 5. Valid boundary values
@pytest.mark.parametrize(
    "field,value",
    [
        ("stock", 0),
        ("rating", 0),
        ("rating", 5)
    ]
)
async def test_patch_product_boundary_values(client, field, value):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={field: value}
    )

    assert response.status_code == 200
    assert response.json()[field] == value


# 6. Update a field with its existing value
async def test_patch_product_same_value(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={"price": float(original["price"])}
    )

    assert response.status_code == 200
    assert response.json()["price"] == original["price"]


# 7. Explicitly set nullable description to NULL
async def test_patch_product_description_null(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={"description": None}
    )
    assert response.status_code == 400

    db_product = await get_product_from_db(original["id"])
    assert db_product["description"] is not None


# 8. Product does not exist
async def test_patch_product_not_found(client):
    response = await client.patch(
        "/products/999999",
        json={"price": 2000}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Product Not Found"


# 9. Invalid product ID
async def test_patch_product_invalid_id(client):
    response = await client.patch(
        "/products/abc",
        json={"price": 2000}
    )

    assert response.status_code == 422


# 10. Empty body or unknown fields
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"unknown_field": "something"}
    ]
)
async def test_patch_product_empty_or_unknown_payload(client, payload):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json=payload
    )

    assert response.status_code == 422

    db_product = await get_product_from_db(original["id"])
    assert db_product == original


# 11. Invalid values
@pytest.mark.parametrize(
    "payload",
    [
        {"name": "A"},
        {"name": "X" * 101},
        {"price": 0},
        {"price": -100},
        {"stock": -1},
        {"rating": 5.1},
        {"rating": -0.1},
        {"price": "free"}
    ]
)
async def test_patch_product_invalid_values(client, payload):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json=payload
    )

    assert response.status_code == 422

    db_product = await get_product_from_db(original["id"])
    assert db_product == original


# 12. Name cannot be NULL (assuming NOT NULL constraint)
async def test_patch_product_name_null(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={"name": None}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid data format or missing required value"

    db_product = await get_product_from_db(original["id"])
    assert db_product == original


# 13. Stock integer overflow (assuming PostgreSQL INTEGER)
async def test_patch_product_stock_overflow(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.patch(
        f"/products/{original['id']}",
        json={"stock": 3_000_000_000}
    )

    assert response.status_code == 400

    db_product = await get_product_from_db(original["id"])
    assert db_product == original


async def test_patch_product_duplicate_name(client):
    products = await seed_products([
        product_data(name="Product A"),
        product_data(name="Product B")
    ])

    product_a, product_b = products

    response = await client.patch(
        f"/products/{product_b['id']}",
        json={"name": product_a["name"]}
    )

    assert response.status_code == 409

    db_product_b = await get_product_from_db(product_b["id"])
    assert db_product_b == product_b


# Tests --> @app.patch("/products/{product_id}") with regenerate_description

def fake_gemini(monkeypatch, text="Fresh AI description."):
    """Mock main.call_gemini and return the list of prompts it receives."""
    prompts = []

    async def fake_call_gemini(prompt):
        prompts.append(prompt)
        return text

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)
    return prompts


# R1. New description is generated and saved
async def test_patch_regenerate_description_success(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"regenerate_description": True}
    )

    assert response.status_code == 200
    assert response.json()["description"] == "Fresh AI description."
    assert len(prompts) == 1

    db_product = await get_product_from_db(original["id"])
    assert db_product["description"] == "Fresh AI description."


# R2. Every other field stays the same
async def test_patch_regenerate_keeps_other_fields(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    fake_gemini(monkeypatch)

    await client.patch(
        f"/products/{original['id']}",
        json={"regenerate_description": True}
    )

    db_product = await get_product_from_db(original["id"])
    for key, value in original.items():
        if key != "description":
            assert db_product[key] == value


# R3. Prompt uses the existing name and category
async def test_patch_regenerate_prompt_uses_existing_values(client, monkeypatch):
    original = (await seed_products([
        product_data(name="Wireless Mouse", category="Electronics")
    ]))[0]
    prompts = fake_gemini(monkeypatch)

    await client.patch(
        f"/products/{original['id']}",
        json={"regenerate_description": True}
    )

    assert "Wireless Mouse" in prompts[0]
    assert "Electronics" in prompts[0]


# R4. Prompt uses the NEW name when it changes in the same request
async def test_patch_regenerate_prompt_uses_new_name(client, monkeypatch):
    original = (await seed_products([
        product_data(name="Wireless Mouse")
    ]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"name": "Gaming Mouse", "regenerate_description": True}
    )

    assert response.status_code == 200
    assert "Gaming Mouse" in prompts[0]
    assert "Wireless Mouse" not in prompts[0]

    db_product = await get_product_from_db(original["id"])
    assert db_product["name"] == "Gaming Mouse"
    assert db_product["description"] == "Fresh AI description."


# R5. Other products are not modified
async def test_patch_regenerate_does_not_modify_other_products(client, monkeypatch):
    product_a, product_b = await seed_products([
        product_data(name="Product A"),
        product_data(name="Product B")
    ])
    fake_gemini(monkeypatch)

    await client.patch(
        f"/products/{product_a['id']}",
        json={"regenerate_description": True}
    )

    assert await get_product_from_db(product_b["id"]) == product_b


# R6. AI failures: right status code, and the row does not change at all
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
    ],
    ids=["timeout", "blocked", "upstream", "unexpected"]
)
async def test_patch_regenerate_ai_failure_leaves_row_unchanged(
    client, monkeypatch, exc, expected_status, expected_detail
):
    original = (await seed_products([product_data()]))[0]

    async def fake_call_gemini(prompt):
        raise exc

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"name": "Renamed Product", "regenerate_description": True}
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail

    # not even the name from the same request may be saved
    assert await get_product_from_db(original["id"]) == original


# R7. Product does not exist: 404 and Gemini is never called
async def test_patch_regenerate_product_not_found(client, monkeypatch):
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        "/products/999999",
        json={"regenerate_description": True}
    )

    assert response.status_code == 404
    assert prompts == []


# R8. Flag together with a manual description is rejected
async def test_patch_regenerate_with_manual_description_returns_422(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"description": "My own text", "regenerate_description": True}
    )

    assert response.status_code == 422
    assert prompts == []
    assert await get_product_from_db(original["id"]) == original


# R9. Flag set to false and nothing else: nothing to update
async def test_patch_regenerate_false_alone_returns_422(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"regenerate_description": False}
    )

    assert response.status_code == 422
    assert prompts == []
    assert await get_product_from_db(original["id"]) == original


# R10. Flag with name = null: 400 and Gemini is never called
async def test_patch_regenerate_with_null_name_returns_400(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"name": None, "regenerate_description": True}
    )

    assert response.status_code == 400
    assert prompts == []
    assert await get_product_from_db(original["id"]) == original


# R11. A normal PATCH (no flag) must never call Gemini
async def test_patch_without_flag_does_not_call_gemini(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]

    async def fake_call_gemini(prompt):
        pytest.fail("Gemini should not have been called")

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"price": 2500}
    )

    assert response.status_code == 200
    assert response.json()["price"] == 2500

    db_product = await get_product_from_db(original["id"])
    assert db_product["description"] == original["description"]


# R12. Flag set to false plus a normal field works like a normal PATCH
async def test_patch_regenerate_false_with_other_field(client, monkeypatch):
    original = (await seed_products([product_data()]))[0]
    prompts = fake_gemini(monkeypatch)

    response = await client.patch(
        f"/products/{original['id']}",
        json={"stock": 99, "regenerate_description": False}
    )

    assert response.status_code == 200
    assert response.json()["stock"] == 99
    assert prompts == []