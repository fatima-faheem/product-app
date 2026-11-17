from main import app
import pytest
import asyncio
from gemini_helper import (
    AIError,
    AITimeoutError,
    AIBlockedError,
    AIUpstreamError,
)
from helpers import product_data, seed_products

# Tests --> @app.post("/products",response_model=list[Response])
# 1. Manual description
async def test_create_product_manual_description(client):

    response = await client.post( 
        "/products",
        json=[
            {
                "name": "Cotton T-Shirt",
                "description": "A soft cotton T-shirt.",
                "category": "Clothing",
                "price": 1200,
                "stock": 10,
                "rating": 4.5,
                "ai_description": False
            }
        ]
    )

    assert response.status_code == 200
    async with app.state.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM products WHERE name = $1",
                "Cotton T-Shirt"
            )
    
    assert row is not None
    assert row["name"]=="Cotton T-Shirt"
    assert row["description"] == "A soft cotton T-shirt."
    data = response.json()
    assert data[0]["description"] == "A soft cotton T-shirt."


# 2. AI-generated description
async def test_create_product_ai_description(client, monkeypatch):

    async def fake_call_gemini(prompt):
        return "A comfortable cotton T-shirt."

    monkeypatch.setattr(
        "main.call_gemini",
        fake_call_gemini
    )

    response = await client.post(
        "/products",
        json=[
            {
                "name": "Cotton T-Shirt",
                "category": "Clothing",
                "price": 1200,
                "stock": 10,
                "rating": 4.5,
                "ai_description": True
            }
        ]
    )

    assert response.status_code == 200
    assert response.json()[0]["description"] == (
        "A comfortable cotton T-shirt."
    )


# 3. Gemini should not be called when AI is disabled
async def test_gemini_not_called_for_manual_description(
    client, monkeypatch
):

    async def fake_call_gemini(prompt):
        pytest.fail("Gemini should not have been called")

    monkeypatch.setattr(
        "main.call_gemini",
        fake_call_gemini
    )

    response = await client.post(
        "/products",
        json=[
            {
                "name": "Laptop",
                "description": "A powerful laptop.",
                "category": "Electronics",
                "price": 50000,
                "stock": 5,
                "rating": 4.5,
                "ai_description": False
            }
        ]
    )

    assert response.status_code == 200


# 4. Invalid product data
@pytest.mark.parametrize(
    "payload",
    [
        {
            "description": "Missing name",
            "price": 1000,
            "stock": 5
        },
        {
            "name": "A",
            "description": "Invalid short name",
            "price": 1000,
            "stock": 5
        },
        {
            "name": "Laptop",
            "description": "Laptop description",
            "price": -100,
            "stock": 5
        },
        {
            "name": "Laptop",
            "description": "Laptop description",
            "price": 1000,
            "stock": -1
        }
    ]
)
async def test_invalid_product_returns_422(client, payload):

    response = await client.post(
        "/products",
        json=[payload]
    )

    assert response.status_code == 422


# 5. Empty product list
async def test_empty_product_list(client):

    response = await client.post(
        "/products",
        json=[]
    )

    assert response.status_code == 400


# 6. AI failure - no database insertion
async def test_ai_failure_no_database_insert(client, monkeypatch):

    async def fake_call_gemini(prompt):
        raise AIUpstreamError(
            "Gemini service unavailable",
            code=503
        )

    monkeypatch.setattr(
        "main.call_gemini",
        fake_call_gemini
    )

    response = await client.post(
        "/products",
        json=[
            {
                "name": "AI Failure Product",
                "category": "Electronics",
                "price": 1000,
                "stock": 5,
                "rating": 4.5,
                "ai_description": True
            }
        ]
    )

    assert response.status_code == 503

    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM products WHERE name = $1",
            "AI Failure Product"
        )

    assert row is None


# 7. Bulk AI failure - no products should be inserted
async def test_bulk_ai_failure_no_insert(client, monkeypatch):

    async def fake_call_gemini(prompt):

        if "Product B" in prompt:
            raise AIUpstreamError(
                "Gemini service unavailable",
                code=503
            )

        return "A good product description."

    monkeypatch.setattr(
        "main.call_gemini",
        fake_call_gemini
    )

    response = await client.post(
        "/products",
        json=[
            {
                "name": "Product A",
                "category": "Clothing",
                "price": 1000,
                "stock": 5,
                "rating": 4.0,
                "ai_description": True
            },
            {
                "name": "Product B",
                "category": "Clothing",
                "price": 1500,
                "stock": 8,
                "rating": 4.5,
                "ai_description": True
            }
        ]
    )

    assert response.status_code == 503

    async with app.state.pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM products
            WHERE name = ANY($1::text[])
            """,
            ["Product A", "Product B"]
        )

    assert count == 0


# Test-8 
async def test_failed_product_cancels_remaining_tasks(client, monkeypatch):

    a_started = asyncio.Event()
    a_cancelled = asyncio.Event()

    async def fake_process_product(product):

        if product.name == "Product A":

            a_started.set()

            try:
                # Keep Product A running until it gets cancelled.
                await asyncio.Future()

            except asyncio.CancelledError:
                a_cancelled.set()
                raise

        elif product.name == "Product B":

            # Wait until Product A has actually started.
            await a_started.wait()

            # Simulate a processing failure.
            raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr(
        "main.process_product",
        fake_process_product
    )

    response = await client.post(
        "/products",
        json=[
            {
                "name": "Product A",
                "description": "Description for product A",
                "category": "Electronics",
                "price": 1000,
                "stock": 5,
                "rating": 4.0,
                "ai_description": False
            },
            {
                "name": "Product B",
                "description": "Description for product B",
                "category": "Electronics",
                "price": 1500,
                "stock": 8,
                "rating": 4.5,
                "ai_description": False
            }
        ]
    )

    assert response.status_code == 500
    assert a_cancelled.is_set()

# Test-9
async def test_ai_timeout_returns_504(client, monkeypatch):

    async def fake_call_gemini(prompt):
        raise AITimeoutError("Gemini timed out")

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)

    response = await client.post(
        "/products",
        json=[{
            "name": "Product A",
            "price": 100,
            "stock": 5,
            "ai_description": True
        }]
    )

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "AI service timed out. Please try again."
    )

# Test-10
async def test_ai_blocked_returns_400(client, monkeypatch):

    async def fake_call_gemini(prompt):
        raise AIBlockedError("Request blocked")

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)

    response = await client.post(
        "/products",
        json=[{
            "name": "Product A",
            "price": 100,
            "stock": 5,
            "ai_description": True
        }]
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "AI request was blocked."

#Test-11
async def tei_upstream_error_returns_503(client, monkeypatch):

    async def fake_call_gemini(prompt):
        raise AIUpstreamError("Gemini unavailable", code=503)

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)

    response = await client.post(
        "/products",
        json=[{
            "name": "Product A",
            "price": 100,
            "stock": 5,
            "ai_description": True
        }]
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "AI service is temporarily unavailable."
    )

# Test-12
async def test_unexpected_ai_error_returns_500(client, monkeypatch):

    async def fake_call_gemini(prompt):
        raise AIError("Unexpected AI error")

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)

    response = await client.post(
        "/products",
        json=[{
            "name": "Product A",
            "price": 100,
            "stock": 5,
            "ai_description": True
        }]
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An unexpected AI error occurred."
    )

# Test-13

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
        json=[
            {
                "name": "Product A",
                "price": 100,
                "stock": 5,
                "ai_description": True
            },
            {
                "name": "Product B",
                "price": 200,
                "stock": 10,
                "ai_description": True
            }
        ]
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    assert data[0]["description"] == "Description for Product A."
    assert data[1]["description"] == "Description for Product B."
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT name, description
            FROM products
            WHERE name = ANY($1::text[])
            """,
            ["Product A", "Product B"]
        )

    saved_products = {
        row["name"]: row["description"]
        for row in rows
    }

    assert len(saved_products) == 2

    assert saved_products == {
        "Product A": "Description for Product A.",
        "Product B": "Description for Product B."
    }


