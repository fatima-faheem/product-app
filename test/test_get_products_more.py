"""
Extra GET /products tests. Your existing test_get_products.py stays as it is,
this file only adds the cases it was missing.
"""
import pytest

from helpers import product_data, seed_products


async def test_category_match_is_exact_not_partial(client):
    await seed_products([product_data(name="Laptop", category="Electronics")])

    response = await client.get("/products", params={"category": "Elec"})

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("category", ["%", "_", "Elec%", "%ics"])
async def test_category_wildcards_are_not_wildcards(client, category):
    await seed_products([product_data(name="Laptop", category="Electronics")])

    response = await client.get("/products", params={"category": category})

    assert response.status_code == 200
    assert response.json() == []


async def test_blank_category_means_no_filter(client):
    await seed_products([
        product_data(name="Shirt", category="Clothing"),
        product_data(name="Laptop", category="Electronics"),
    ])

    response = await client.get("/products", params={"category": ""})

    assert response.status_code == 200
    assert len(response.json()) == 2


async def test_min_price_boundary_is_inclusive_and_zero_works(client):
    await seed_products([
        product_data(name="Free Sample", price=0),
        product_data(name="Paid Item", price=10),
    ])

    response = await client.get("/products", params={"min_price": 0})

    assert response.status_code == 200
    assert {p["name"] for p in response.json()} == {"Free Sample", "Paid Item"}


async def test_max_price_zero_returns_only_free_products(client):
    await seed_products([
        product_data(name="Free Sample", price=0),
        product_data(name="Paid Item", price=10),
    ])

    response = await client.get("/products", params={"max_price": 0})

    assert response.status_code == 200
    assert [p["name"] for p in response.json()] == ["Free Sample"]


async def test_product_with_null_rating_is_returned_with_null(client):
    await seed_products([product_data(name="Unrated", rating=None)])

    response = await client.get("/products")

    assert response.status_code == 200
    assert response.json()[0]["rating"] is None


async def test_response_has_all_expected_fields(client):
    await seed_products([product_data(name="Laptop")])

    response = await client.get("/products")

    assert set(response.json()[0]) == {
        "id", "name", "description", "category", "price", "stock", "rating",
    }


async def test_results_are_ordered_by_id(client):
    await seed_products([
        product_data(name="Zebra"),
        product_data(name="Apple"),
        product_data(name="Mango"),
    ])

    response = await client.get("/products")

    ids = [p["id"] for p in response.json()]
    assert ids == sorted(ids)
    assert [p["name"] for p in response.json()] == ["Zebra", "Apple", "Mango"]
