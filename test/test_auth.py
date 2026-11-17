
import os

import pytest

import Api_Key_Check
from main import app

from fastapi import HTTPException

# Import your existing test helpers.
# Change "test_products" to the actual name of your existing test file.
from helpers import (
    seed_products,
    product_data,
    get_product_from_db,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_KEY = os.environ["PRODUCT_API_KEY"]

VALID_POST_BODY = [{
    "name": "Auth Test Product",
    "description": "Some description",
    "category": "Other",
    "price": 100,
    "stock": 5,
    "rating": 4.0,
}]


# Protected endpoints and their expected responses
PROTECTED = [
    pytest.param("POST", "/products", VALID_POST_BODY, 200, id="post"),
    pytest.param("PATCH", "/products/999999", {"price": 100}, 404, id="patch"),
    pytest.param("DELETE", "/products/999999", None, 404, id="delete"),
]


# ==================================================
# 1. MISSING API KEY
# ==================================================

@pytest.mark.parametrize(
    "method,url,body,ok_status",
    PROTECTED
)
async def test_missing_key_returns_401(
    anon_client, method, url, body, ok_status
):
    response = await anon_client.request(
        method,
        url,
        json=body
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


# ==================================================
# 2. WRONG API KEY
# ==================================================

@pytest.mark.parametrize(
    "method,url,body,ok_status",
    PROTECTED
)
@pytest.mark.parametrize(
    "bad_key",
    [
        "wrong-key",
        "",
        TEST_KEY + "x",
        TEST_KEY[:-1],
        TEST_KEY.upper(),
    ],
    ids=[
        "wrong",
        "empty",
        "extra-char",
        "missing-char",
        "wrong-case",
    ],
)
async def test_wrong_key_returns_401(
    client, method, url, body, ok_status, bad_key
):
    response = await client.request(
        method,
        url,
        json=body,
        headers={"X-API-Key": bad_key}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


# ==================================================
# 3. CORRECT API KEY
# ==================================================

@pytest.mark.parametrize(
    "method,url,body,ok_status",
    PROTECTED
)
async def test_correct_key_passes_auth(
    client, method, url, body, ok_status
):
    response = await client.request(
        method,
        url,
        json=body
    )

    assert response.status_code == ok_status


# ==================================================
# 4. HEADER NAME IS CASE-INSENSITIVE
# ==================================================

async def test_header_name_is_case_insensitive(anon_client):
    response = await anon_client.delete(
        "/products/999999",
        headers={"x-api-key": TEST_KEY}
    )

    assert response.status_code == 404


# ==================================================
# 5. PUBLIC GET ENDPOINTS
# ==================================================

async def test_read_endpoints_need_no_key(anon_client):

    response = await anon_client.get("/Hello")
    assert response.status_code == 200

    response = await anon_client.get("/products")
    assert response.status_code == 200

    response = await anon_client.get("/products/stats")
    assert response.status_code != 401


# ==================================================
# 6. UNAUTHORIZED REQUESTS MUST NOT CHANGE DATABASE
# ==================================================

async def test_post_without_key_inserts_nothing(anon_client):

    async with app.state.pool.acquire() as conn:
        before = await conn.fetchval(
            "SELECT COUNT(*) FROM products"
        )

    response = await anon_client.post(
        "/products",
        json=VALID_POST_BODY
    )

    assert response.status_code == 401

    async with app.state.pool.acquire() as conn:
        after = await conn.fetchval(
            "SELECT COUNT(*) FROM products"
        )

    assert after == before


async def test_patch_without_key_leaves_row_unchanged(anon_client):

    original = (await seed_products([product_data()]))[0]

    response = await anon_client.patch(
        f"/products/{original['id']}",
        json={"price": 5000}
    )

    assert response.status_code == 401

    updated = await get_product_from_db(original["id"])

    assert updated == original


async def test_delete_without_key_keeps_the_row(anon_client):

    original = (await seed_products([product_data()]))[0]

    response = await anon_client.delete(
        f"/products/{original['id']}"
    )

    assert response.status_code == 401

    existing = await get_product_from_db(original["id"])

    assert existing == original


# ==================================================
# 7. AUTHENTICATION BEFORE VALIDATION
# ==================================================

async def test_invalid_body_without_key_returns_401(anon_client):

    response = await anon_client.post(
        "/products",
        json=[{"name": "A"}]
    )

    assert response.status_code == 401


# ==================================================
# 8. SERVER API KEY NOT CONFIGURED
# ==================================================

@pytest.mark.parametrize(
    "server_key",
    [None, ""],
    ids=["none", "empty"]
)
async def test_missing_server_key_fails_closed_with_500(
    client, monkeypatch, server_key
):

    monkeypatch.setattr(
        Api_Key_Check,
        "PRODUCT_API_KEY",
        server_key
    )

    response = await client.delete(
        "/products/999999",
        headers={"X-API-Key": ""}
    )

    assert response.status_code == 500

    assert response.json()["detail"] == "API key is not configured"


# ==================================================
# 9. NON-ASCII API KEY
# ==================================================

async def test_non_ascii_key_returns_401(monkeypatch):

    monkeypatch.setattr(
        Api_Key_Check,
        "PRODUCT_API_KEY",
        "correct-secret"
    )

    with pytest.raises(HTTPException) as exc:
        await Api_Key_Check.verify_api_key("clé-secrète")

    assert exc.value.status_code == 401
