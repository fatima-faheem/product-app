from main import app


# ---------------------------------------------------------------- database
async def seed_products(products):
    """Insert products directly into the test database (never through POST)."""
    names = [p["name"] for p in products]
    descriptions = [p["description"] for p in products]
    categories = [p["category"] for p in products]
    prices = [p["price"] for p in products]
    stocks = [p["stock"] for p in products]
    ratings = [p["rating"] for p in products]

    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            INSERT INTO products
                (name, description, category, price, stock, rating)
            SELECT *
            FROM UNNEST(
                $1::text[],
                $2::text[],
                $3::text[],
                $4::numeric[],
                $5::integer[],
                $6::numeric[]
            )
            RETURNING id, name, description, category, price, stock, rating
            """,
            names,
            descriptions,
            categories,
            prices,
            stocks,
            ratings,
        )

    return [dict(row) for row in rows]


def product_data(**overrides):
    data = {
        "name": "Test Product",
        "description": "Test description",
        "category": "Electronics",
        "price": 1000,
        "stock": 10,
        "rating": 4.5,
    }
    data.update(overrides)
    return data


async def get_product_from_db(product_id):
    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, description, category, price, stock, rating
            FROM products
            WHERE id = $1
            """,
            product_id,
        )

    return dict(row) if row else None


async def get_product_by_name(name):
    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, description, category, price, stock, rating
            FROM products
            WHERE name = $1
            """,
            name,
        )

    return dict(row) if row else None


async def count_products():
    async with app.state.pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM products")


# ------------------------------------------------------------------ gemini
def fake_gemini(monkeypatch, text="Fresh AI description."):
    """Replace main.call_gemini. Returns the list of prompts it receives."""
    prompts = []

    async def fake_call_gemini(prompt):
        prompts.append(prompt)
        return text

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)
    return prompts


def fake_gemini_error(monkeypatch, exc):
    """Replace main.call_gemini with a version that raises `exc`.
    Returns the list of prompts it was called with."""
    prompts = []

    async def fake_call_gemini(prompt):
        prompts.append(prompt)
        raise exc

    monkeypatch.setattr("main.call_gemini", fake_call_gemini)
    return prompts
