import pytest

from helpers import (
    count_products,
    get_product_from_db,
    product_data,
    seed_products,
)


# ============================================================ DELETE /products/{id}
async def test_delete_product_success(client):
    original = (await seed_products([product_data()]))[0]

    response = await client.delete(f"/products/{original['id']}")

    assert response.status_code == 204
    assert response.content == b""
    assert await get_product_from_db(original["id"]) is None


async def test_delete_product_does_not_touch_other_products(client):
    product_a, product_b = await seed_products([
        product_data(name="Product A"),
        product_data(name="Product B"),
    ])

    response = await client.delete(f"/products/{product_a['id']}")

    assert response.status_code == 204
    assert await get_product_from_db(product_a["id"]) is None
    assert await get_product_from_db(product_b["id"]) == product_b
    assert await count_products() == 1


async def test_delete_product_not_found(client):
    response = await client.delete("/products/999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Product Not Found"


async def test_delete_product_twice_returns_404_the_second_time(client):
    original = (await seed_products([product_data()]))[0]

    first = await client.delete(f"/products/{original['id']}")
    second = await client.delete(f"/products/{original['id']}")

    assert first.status_code == 204
    assert second.status_code == 404


async def test_delete_product_invalid_id(client):
    response = await client.delete("/products/abc")

    assert response.status_code == 422


# ============================================================ GET /products/stats
async def test_stats_empty_database_returns_empty_list(client):
    response = await client.get("/products/stats")

    assert response.status_code == 200
    assert response.json() == []


async def test_stats_groups_by_category(client):
    await seed_products([
        product_data(name="Shirt", category="Clothing", price=1000, stock=10),
        product_data(name="Jeans", category="Clothing", price=2000, stock=20),
        product_data(name="Laptop", category="Electronics", price=5000, stock=5),
    ])

    response = await client.get("/products/stats")

    assert response.status_code == 200
    # the endpoint has no ORDER BY, so never rely on row order
    stats = {row["category"]: row for row in response.json()}
    assert set(stats) == {"Clothing", "Electronics"}

    assert stats["Clothing"]["average_price"] == pytest.approx(1500)
    assert stats["Clothing"]["total_stock"] == 30
    assert stats["Clothing"]["total_products"] == 2

    assert stats["Electronics"]["average_price"] == pytest.approx(5000)
    assert stats["Electronics"]["total_stock"] == 5
    assert stats["Electronics"]["total_products"] == 1


async def test_stats_average_is_rounded_to_two_decimals(client):
    await seed_products([
        product_data(name="A", category="Books", price=10, stock=1),
        product_data(name="B", category="Books", price=10, stock=1),
        product_data(name="C", category="Books", price=11, stock=1),
    ])

    response = await client.get("/products/stats")

    # (10 + 10 + 11) / 3 = 10.3333...
    assert response.json()[0]["average_price"] == pytest.approx(10.33)


async def test_stats_includes_free_products(client):
    await seed_products([
        product_data(name="Free Sample", category="Promo", price=0, stock=3),
    ])

    response = await client.get("/products/stats")

    row = response.json()[0]
    assert row["category"] == "Promo"
    assert row["average_price"] == 0
    assert row["total_products"] == 1


async def test_stats_reflects_deletions(client):
    products = await seed_products([
        product_data(name="A", category="Books"),
        product_data(name="B", category="Books"),
    ])
    await client.delete(f"/products/{products[0]['id']}")

    response = await client.get("/products/stats")

    assert response.json()[0]["total_products"] == 1
