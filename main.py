import time
import os
import asyncio
import logging
from contextlib import asynccontextmanager

import asyncpg
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field, model_validator, field_validator

from gemini_helper import (
    call_gemini,
    AIError,
    AITimeoutError,
    AIBlockedError,
    AIUpstreamError,
    AIQuotaError
)
from Api_Key_Check import verify_api_key

# ---------- logger ----------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DEFAULT_CATEGORY = "Other"


# ---------- lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to PostgreSQL")

    app.state.pool = await asyncpg.create_pool(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        ssl=os.getenv("DB_SSL") or None,
        min_size=1,
        max_size=5,
    )

    yield

    await app.state.pool.close()


app = FastAPI(lifespan=lifespan)

# Limits how many Gemini calls run at the same time
gemini_semaphore = asyncio.Semaphore(5)


# ---------- dependencies ----------
async def get_db():
    """Connection for endpoints that only touch the DB."""
    async with app.state.pool.acquire() as conn:
        yield conn


def get_pool():
    """Pool for endpoints that call Gemini first, so they don't hold a
    connection while waiting for the AI."""
    return app.state.pool


# ---------- helpers ----------
def normalize_category(value: str | None) -> str:
    """One rule everywhere: null / blank category -> 'Other'."""
    if value is None:
        return DEFAULT_CATEGORY
    value = value.strip()
    return value or DEFAULT_CATEGORY



# ---------- models ----------

class ProductResponse(BaseModel):
    id: int
    name: str
    description: str
    category: str = DEFAULT_CATEGORY
    price: float
    stock: int
    rating: float | None = None

    @field_validator("category", mode="before")
    @classmethod
    def category_default(cls, v):
        return normalize_category(v)


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: str | None = Field(None, min_length=1)
    category: str = DEFAULT_CATEGORY
    price: float = Field(..., ge=0)
    stock: int = Field(..., ge=0)
    rating: float | None = Field(0.0, ge=0.0, le=5.0)
    ai_description: bool = False

    @field_validator("category", mode="before")
    @classmethod
    def category_default(cls, v):
        return normalize_category(v)

    @model_validator(mode="after")
    def validate_description(self):
        if self.ai_description and self.description is not None:
            raise ValueError(
                "Don't provide a description when AI description is enabled"
            )

        if self.description is None and not self.ai_description:
            raise ValueError(
                "Provide a description or enable AI description"
            )

        return self


class ProductPatch(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=100)
    description: str | None = None
    category: str | None = None
    price: float | None = Field(None, ge=0)
    stock: int | None = Field(None, ge=0)
    rating: float | None = Field(None, ge=0.0, le=5.0)
    regenerate_description: bool = False

    @field_validator("category", mode="before")
    @classmethod
    def category_default(cls, v):
        return normalize_category(v)

    @model_validator(mode="after")
    def check_request(self):
        real_fields = self.model_fields_set - {"regenerate_description"}

        # These columns are NOT NULL in demodb
        for field in ("name", "description", "price", "stock"):
            if field in real_fields and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")

        if self.regenerate_description and "description" in real_fields:
            raise ValueError(
                "Don't send a description when regenerate_description is true"
            )

        if not real_fields and not self.regenerate_description:
            raise ValueError("At least one field must be provided")

        return self



# ---------- AI helpers ----------
def ai_error_to_http(exc: AIError) -> HTTPException:
    if isinstance(exc, AITimeoutError):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="AI service timed out. Please try again.",
        )
    if isinstance(exc, AIBlockedError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="AI request was blocked.",
        )
    if isinstance(exc, AIQuotaError):
        if exc.kind == "daily":
            detail = "AI daily request quota exhausted. Please try again later."

        elif exc.kind == "rate":
            detail = "AI request rate limit reached. Please try again shortly."

        else:
            detail = "AI service quota limit reached. Please try again later."

        return HTTPException(
            status_code=503,
            detail=detail
        )
    if isinstance(exc, AIUpstreamError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is temporarily unavailable.",
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected AI error occurred.",
    )


def build_description_prompt(name: str, category: str | None) -> str:
    return f"""Write a short product description for this product.

Product name: {name}
Category: {category or DEFAULT_CATEGORY}

Keep it concise and suitable for an online store.
Do not invent specifications that were not provided."""


async def describe_with_ai(name: str, category: str | None) -> str:
    """The one place that calls Gemini and converts AI errors to HTTP errors."""
    try:
        async with gemini_semaphore:
            return await call_gemini(build_description_prompt(name, category))
    except AIError as exc:
        raise ai_error_to_http(exc)


async def process_product(product: ProductCreate) -> None:
    if not product.ai_description:
        logger.info("Product '%s' | manual description", product.name)
        return
    logger.info("Product '%s' | generating AI description", product.name)
    product.description = await describe_with_ai(product.name, product.category)


# ---------- routes ----------
@app.get("/Hello")
def read_root():
    return "Hello World"


# 1 - GET & FILTER
@app.get("/products", response_model=list[ProductResponse])
async def get_products(
    category: str | None = None,
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    conn=Depends(get_db),
):
    try:
        query = """SELECT id, name, description, category, price, stock, rating
                   FROM products WHERE 1=1"""
        values = []

        if category:
            values.append(category)
            # exact, case-insensitive match; NULL rows count as 'Other'
            query += f" AND LOWER(category) = LOWER(${len(values)})"
        if min_price is not None:
            values.append(min_price)
            query += f" AND price >= ${len(values)}"
        if max_price is not None:
            values.append(max_price)
            query += f" AND price <= ${len(values)}"

        query += " ORDER BY id ASC"
        rows = await conn.fetch(query, *values)
        return [dict(r) for r in rows]

    except asyncpg.PostgresConnectionError:
        logger.exception("Database connection failed in get_products")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except asyncpg.PostgresError:
        logger.exception("Database error in get_products")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception:
        logger.exception("Unexpected error in get_products")
        raise HTTPException(status_code=500, detail="Internal server error")


# 2 - GET STATS
@app.get("/products/stats")
async def get_product_stats(conn=Depends(get_db)):
    try:
        query = f"""
            SELECT
            category,
            ROUND(AVG(price), 2) AS average_price,
            SUM(stock) AS total_stock,
            COUNT(*) AS total_products
        FROM products
        GROUP BY category
        ORDER BY total_products DESC, category
        """
        rows = await conn.fetch(query)
        return [dict(r) for r in rows]  # empty table -> []

    except asyncpg.PostgresConnectionError:
        logger.exception("Database connection failed in get_product_stats")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except asyncpg.PostgresError:
        logger.exception("Database error in get_product_stats")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception:
        logger.exception("Unexpected error in get_product_stats")
        raise HTTPException(status_code=500, detail="Internal server error")


# 3 - POST
@app.post(
    "/products",
    response_model=list[ProductResponse],
    dependencies=[Depends(verify_api_key)],
)
async def create_products(
    products: list[ProductCreate],
    pool=Depends(get_pool),
):
    if not products:
        raise HTTPException(
            status_code=400, detail="No products provided for insertion."
        )

    try:
        # 1) Do all AI work first, with no DB connection held
        tasks = [asyncio.create_task(process_product(p)) for p in products]
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # 2) Then take a connection just for the insert
        names = [p.name for p in products]
        descriptions = [p.description for p in products]
        categories = [p.category for p in products]
        prices = [p.price for p in products]
        stocks = [p.stock for p in products]
        ratings = [p.rating for p in products]

        query = """
            INSERT INTO products (name, description, category, price, stock, rating)
            SELECT * FROM UNNEST(
                $1::text[],
                $2::text[],
                $3::text[],
                $4::numeric[],
                $5::integer[],
                $6::numeric[]
            )
            RETURNING id, name, description, category, price, stock, rating;
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query, names, descriptions, categories, prices, stocks, ratings
            )
        return [dict(r) for r in rows]

    except HTTPException:
        raise
    except asyncpg.UniqueViolationError:
        logger.exception("Duplicate product encountered during insertion")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="One or more products already exist.",
        )
    except (asyncpg.DataError, asyncpg.NotNullViolationError):
        logger.exception("Invalid product data during database insertion")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid product data.",
        )
    except asyncpg.PostgresError:
        logger.exception("Database error while creating products")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A database error occurred.",
        )
    except Exception:
        logger.exception("Unexpected error while creating products")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected server error occurred.",
        )


# 4 - PATCH
@app.patch(
    "/products/{product_id}",
    response_model=ProductResponse,
    dependencies=[Depends(verify_api_key)],
)
async def update_products(
    product_id: int,
    product: ProductPatch,
    pool=Depends(get_pool),
):
    # the flag is not a DB column, so keep it out of the SET clause
    data = product.model_dump(exclude_unset=True, exclude={"regenerate_description"})

    try:
        # Generate the description BEFORE the update, so if Gemini fails
        # the row stays unchanged.
        if product.regenerate_description:
            async with pool.acquire() as conn:
                existing = await conn.fetchrow(
                    "SELECT name, category FROM products WHERE id = $1",
                    product_id,
                )
            if not existing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Product Not Found",
                )
            # use the new value if it's changed in this same request
            name = data.get("name", existing["name"])
            category = data.get("category", existing["category"])
            generated_description = await describe_with_ai(name, category)

            if not generated_description or not generated_description.strip():
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="AI generated an empty description",
                )

            data["description"] = generated_description

        # Column names come only from the model's fixed fields, never from
        # user input, so building the SET clause this way is safe.
        to_update = []
        values = []
        for idx, (key, value) in enumerate(data.items(), start=1):
            to_update.append(f"{key} = ${idx}")
            values.append(value)
        values.append(product_id)

        query = f"""
            UPDATE products
            SET {", ".join(to_update)}
            WHERE id = ${len(values)}
            RETURNING id, name, description, category, price, stock, rating;
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *values)

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product Not Found",
            )
        return dict(row)

    except HTTPException:
        raise
    except asyncpg.UniqueViolationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A product with this name already exists",
        )
    except (asyncpg.DataError, asyncpg.NotNullViolationError):
        logger.exception("Invalid data while updating product %s", product_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid data format or missing required value",
        )
    except asyncpg.PostgresError:
        logger.exception("Database error while updating product %s", product_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred",
        )
    except Exception:
        logger.exception("Unexpected error while updating product %s", product_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# 5 - DELETE
@app.delete(
    "/products/{product_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_api_key)],
)
async def delete_product(
    product_id: int,
    conn: asyncpg.Connection = Depends(get_db),
):
    try:
        # a single DELETE is already atomic, no transaction needed
        row = await conn.fetchrow(
            "DELETE FROM products WHERE id = $1 RETURNING id;", product_id
        )
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product Not Found",
            )
    except HTTPException:
        raise
    except asyncpg.PostgresError:
        logger.exception("Database error while deleting product %s", product_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred",
        )
    except Exception:
        logger.exception("Unexpected error while deleting product %s", product_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )