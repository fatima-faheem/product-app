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

# TESTS --> @app.get("/products",response_model=list[Response])

# 1. Get all products and check their order
async def test_get_all_products(client):
    await seed_products([
        {
            "name": "Cotton T-Shirt",
            "description": "Comfortable shirt",
            "category": "Clothing",
            "price": 1000,
            "stock": 10,
            "rating": 4.5
        },
        {
            "name": "Gaming Laptop",
            "description": "Powerful laptop",
            "category": "Electronics",
            "price": 150000,
            "stock": 5,
            "rating": 4.8
        }
    ])
    response = await client.get("/products")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "Cotton T-Shirt"
    assert data[1]["name"] == "Gaming Laptop"
    assert data[0]["price"] == 1000
    assert data[1]["price"] == 150000


# 2. Get products when database is empty
async def test_get_products_empty_database(client):
    response = await client.get("/products")

    assert response.status_code == 200
    assert response.json() == []


# 3. Filter products by category (case-insensitive)
async def test_get_products_by_category(client):
    await seed_products([
        {
            "name": "Cotton T-Shirt",
            "description": "Comfortable shirt",
            "category": "Clothing",
            "price": 1000,
            "stock": 10,
            "rating": 4.5
        },
        {
            "name": "Gaming Laptop",
            "description": "Powerful laptop",
            "category": "Electronics",
            "price": 150000,
            "stock": 5,
            "rating": 4.8
        }
    ])

    response = await client.get(
        "/products",
        params={"category": "electronics"}
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 1
    assert data[0]["name"] == "Gaming Laptop"
    assert data[0]["category"] == "Electronics"


# 4. Category with no matching products
async def test_get_products_category_no_match(client):
    await seed_products([
        {
            "name": "Gaming Laptop",
            "description": "Powerful laptop",
            "category": "Electronics",
            "price": 150000,
            "stock": 5,
            "rating": 4.8
        }
    ])

    response = await client.get(
        "/products",
        params={"category": "Furniture"}
    )

    assert response.status_code == 200
    assert response.json() == []


# 5. Minimum price filter
async def test_get_products_min_price_filter(client):
    await seed_products([
        {
            "name": "Product A",
            "description": "First product",
            "category": "Electronics",
            "price": 100,
            "stock": 10,
            "rating": 4.0
        },
        {
            "name": "Product B",
            "description": "Second product",
            "category": "Electronics",
            "price": 500,
            "stock": 20,
            "rating": 4.5
        }
    ])

    response = await client.get(
        "/products",
        params={"min_price": 500}
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 1
    assert data[0]["name"] == "Product B"
    assert data[0]["price"] == 500


# 6. Maximum price filter
async def test_get_products_max_price_filter(client):
    await seed_products([
        {
            "name": "Product A",
            "description": "First product",
            "category": "Electronics",
            "price": 100,
            "stock": 10,
            "rating": 4.0
        },
        {
            "name": "Product B",
            "description": "Second product",
            "category": "Electronics",
            "price": 500,
            "stock": 20,
            "rating": 4.5
        },
        {
            "name": "Product C",
            "description": "Third product",
            "category": "Electronics",
            "price": 800,
            "stock": 30,
            "rating": 4.8
        }
    ])

    response = await client.get(
        "/products",
        params={"max_price": 500}
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 2
    assert [p["name"] for p in data] == ["Product A", "Product B"]


# 7. Minimum and maximum price together
async def test_get_products_price_range(client):
    await seed_products([
        {
            "name": "Product A",
            "description": "First product",
            "category": "Electronics",
            "price": 100,
            "stock": 10,
            "rating": 4.0
        },
        {
            "name": "Product B",
            "description": "Second product",
            "category": "Electronics",
            "price": 300,
            "stock": 20,
            "rating": 4.5
        },
        {
            "name": "Product C",
            "description": "Third product",
            "category": "Electronics",
            "price": 500,
            "stock": 30,
            "rating": 4.8
        },
        {
            "name": "Product D",
            "description": "Fourth product",
            "category": "Electronics",
            "price": 700,
            "stock": 40,
            "rating": 4.2
        }
    ])

    response = await client.get(
        "/products",
        params={"min_price": 300, "max_price": 500}
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 2
    assert [p["name"] for p in data] == ["Product B", "Product C"]


# 8. Category and both price filters together
async def test_get_products_all_filters(client):
    await seed_products([
        {
            "name": "Product A",
            "description": "First product",
            "category": "Electronics",
            "price": 200,
            "stock": 10,
            "rating": 4.0
        },
        {
            "name": "Product B",
            "description": "Second product",
            "category": "Electronics",
            "price": 400,
            "stock": 20,
            "rating": 4.5
        },
        {
            "name": "Product C",
            "description": "Third product",
            "category": "Electronics",
            "price": 800,
            "stock": 30,
            "rating": 4.8
        },
        {
            "name": "Product D",
            "description": "Fourth product",
            "category": "Clothing",
            "price": 400,
            "stock": 40,
            "rating": 4.2
        }
    ])

    response = await client.get(
        "/products",
        params={
            "category": "electronics",
            "min_price": 300,
            "max_price": 500
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 1
    assert data[0]["name"] == "Product B"


# 9. Price filters without category
async def test_get_products_price_filters_without_category(client):
    await seed_products([
        {
            "name": "Product A",
            "description": "First product",
            "category": "Electronics",
            "price": 100,
            "stock": 10,
            "rating": 4.0
        },
        {
            "name": "Product B",
            "description": "Second product",
            "category": "Clothing",
            "price": 300,
            "stock": 20,
            "rating": 4.5
        },
        {
            "name": "Product C",
            "description": "Third product",
            "category": "Electronics",
            "price": 500,
            "stock": 30,
            "rating": 4.8
        }
    ])

    response = await client.get(
        "/products",
        params={"min_price": 200, "max_price": 400}
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data) == 1
    assert data[0]["name"] == "Product B"


# 10. Invalid price query parameters
async def test_get_products_invalid_price_parameters(client):
    response1 = await client.get(
        "/products",
        params={"min_price": -100}
    )

    response2 = await client.get(
        "/products",
        params={"max_price": "abc"}
    )

    assert response1.status_code == 422
    assert response2.status_code == 422


# 11. Minimum price greater than maximum price
async def test_get_products_min_price_greater_than_max_price(client):
    await seed_products([
        {
            "name": "Product A",
            "description": "First product",
            "category": "Electronics",
            "price": 300,
            "stock": 10,
            "rating": 4.0
        }
    ])

    response = await client.get(
        "/products",
        params={"min_price": 500, "max_price": 200}
    )

    assert response.status_code == 200
    assert response.json() == []
