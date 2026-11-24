"""
Locks in the database contract (the demodb schema) for my_app_test.

These run ONLY against the local test database (conftest refuses anything
else). If the test database ever drifts from the contract, these fail first.
"""
import asyncpg
import pytest

from main import app

EXPECTED_COLUMNS = {
    # name: (data_type, max_length, is_nullable, default)
    "id": ("integer", None, "NO", "nextval('products_id_seq'::regclass)"),
    "name": ("character varying", 100, "NO", None),
    "description": ("text", None, "NO", None),
    "category": ("text", None, "NO", "'Other'::text"),
    "price": ("numeric", None, "NO", None),
    "stock": ("integer", None, "NO", "0"),
    "rating": ("numeric", None, "YES", "0.00"),
    "created_at": ("timestamp without time zone", None, "YES", "CURRENT_TIMESTAMP"),
}

EXPECTED_CONSTRAINTS = {
    "check_price": "CHECK ((price >= (0)::numeric))",
    "check_rating": "CHECK (((rating >= (0)::numeric) AND (rating <= (5)::numeric)))",
    "check_stock": "CHECK ((stock >= 0))",
    "products_name_key": "UNIQUE (name)",
    "products_pkey": "PRIMARY KEY (id)",
}

EXPECTED_INDEXES = {"products_name_key", "products_pkey"}


# ------------------------------------------------------------ structure
async def test_columns_match_the_contract(client):
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, character_maximum_length,
                   is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'products'
            """
        )

    actual = {
        r["column_name"]: (
            r["data_type"],
            r["character_maximum_length"],
            r["is_nullable"],
            r["column_default"],
        )
        for r in rows
    }
    assert actual == EXPECTED_COLUMNS


async def test_constraints_match_the_contract(client):
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT conname, pg_get_constraintdef(oid) AS definition
            FROM pg_constraint
            WHERE conrelid = 'public.products'::regclass
              AND contype IN ('c', 'u', 'p')
            """
        )

    assert {r["conname"]: r["definition"] for r in rows} == EXPECTED_CONSTRAINTS


async def test_indexes_match_the_contract(client):
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public' AND tablename = 'products'
            """
        )

    assert {r["indexname"] for r in rows} == EXPECTED_INDEXES


# ------------------------------------------- the rules actually enforced
INSERT = """
    INSERT INTO products (name, description, category, price, stock, rating)
    VALUES ($1, $2, $3, $4, $5, $6)
"""


def row(**overrides):
    data = {
        "name": "Direct Insert",
        "description": "A description",
        "category": "Other",
        "price": 10,
        "stock": 1,
        "rating": 3,
    }
    data.update(overrides)
    return tuple(data.values())


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"price": -1}, asyncpg.CheckViolationError),
        ({"stock": -1}, asyncpg.CheckViolationError),
        ({"rating": 6}, asyncpg.CheckViolationError),
        ({"rating": -1}, asyncpg.CheckViolationError),
        ({"description": None}, asyncpg.NotNullViolationError),
        ({"category": None}, asyncpg.NotNullViolationError),
        ({"price": None}, asyncpg.NotNullViolationError),
        ({"stock": None}, asyncpg.NotNullViolationError),
        ({"name": "x" * 101}, asyncpg.StringDataRightTruncationError),
    ],
    ids=[
        "price-negative", "stock-negative", "rating-high", "rating-low",
        "description-null", "category-null", "price-null", "stock-null",
        "name-too-long",
    ],
)
async def test_database_rejects_invalid_rows(client, overrides, error):
    async with app.state.pool.acquire() as conn:
        with pytest.raises(error):
            await conn.execute(INSERT, *row(**overrides))


async def test_database_rejects_duplicate_names(client):
    async with app.state.pool.acquire() as conn:
        await conn.execute(INSERT, *row(name="Same Name"))
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(INSERT, *row(name="Same Name"))


@pytest.mark.parametrize(
    "overrides",
    [{"price": 0}, {"stock": 0}, {"rating": 0}, {"rating": 5}, {"rating": None}],
    ids=["price-0", "stock-0", "rating-0", "rating-5", "rating-null"],
)
async def test_database_accepts_boundary_rows(client, overrides):
    async with app.state.pool.acquire() as conn:
        await conn.execute(INSERT, *row(**overrides))
