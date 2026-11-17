import os
from fastapi import FastAPI, HTTPException,status,Query,Depends
from pydantic import BaseModel, Field, model_validator, field_validator
import logging
import asyncpg
import asyncio
from contextlib import asynccontextmanager
from google import genai
from google.genai import types
from fastapi.responses import StreamingResponse
from google.genai import errors
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from gemini_helper import (
    call_gemini,
    AIError,
    AITimeoutError,
    AIBlockedError,
    AIUpstreamError,
)
from Api_Key_Check import verify_api_key
load_dotenv()

#------client for genai sdk--------------
client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=1
        )
    )
)

#----------logger-------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

#--------------Lifespan--------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    db_name = os.getenv("DB_NAME")
    print(f"--> Lifespan connecting to DB: {db_name}")
    app.state.pool = await asyncpg.create_pool(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        min_size=5,
        max_size=20
    )
    yield
    await app.state.pool.close()
app = FastAPI(lifespan=lifespan)

# -----------------------Semaphore-----------------
gemini_semaphore = asyncio.Semaphore(5)

#-----------------dependency_function-------------
async def get_db():
    async with app.state.pool.acquire() as conn:
        yield conn

#-----------(ai_error --> http_error) Converter----------------------
def ai_error_to_http(exc: AIError) -> HTTPException:

    if isinstance(exc, AITimeoutError):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="AI service timed out. Please try again."
        )

    if isinstance(exc, AIBlockedError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="AI request was blocked."
        )

    if isinstance(exc, AIUpstreamError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is temporarily unavailable."
        )

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected AI error occurred."
    )


#-----------------------------------------------
# One place that calls Gemini and converts AI errors to HTTP errors.
# Both POST and PATCH use it.
#-----------------------------------------------
async def describe_with_ai(name: str, category: str | None) -> str:
    try:
        return await call_gemini(build_description_prompt(name, category))
    except AIError as exc:
        raise ai_error_to_http(exc)


# ---------prompt_making_for_gemini------------
def build_description_prompt(name: str, category: str | None) -> str:
    return f"""Write a short product description for this product.

Product name: {name}
Category: {category or "Other"}

Keep it concise and suitable for an online store.
Do not invent specifications that were not provided."""


#----------process_prdouct (FOR POST & PATCH ENDPOINT)------------------
async def process_product(product: Create_Product):
    if not product.ai_description:
        logger.info("Product '%s' | manual description", product.name)
        return
    logger.info("Product '%s' | generating AI description", product.name)
    product.description = await describe_with_ai(product.name, product.category)


# Response_Model
class Response(BaseModel):
    id:int
    name:str=Field(...,min_length=2,max_length=100)
    description:str|None=None
    category:str="other"
    price:float=Field(...,gt=0)
    stock:int=Field(...,ge=0)
    rating:float|None=Field(0.00,le=5.0,ge=0.0)
# Create Model
class Create_Product(BaseModel):
    name:str=Field(...,min_length=2,max_length=100)
    description:str|None=None
    category:str|None="Other"
    price:float=Field(...,gt=0)
    stock:int=Field(...,ge=0)
    rating:float|None=Field(0.00,ge=0.00,le=5.0)
    ai_description:bool=False
    @model_validator(mode="after")
    def validate_description(self):
        if self.ai_description and self.description:
            raise ValueError("Donot provide a description when AI description is enabled")
        if not self.description and not self.ai_description:
            raise ValueError("Provide  a description or enable AI description")
        return self

# Patch Model
class patch_product(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=100)
    description: str | None = None
    category: str | None = None
    price: float | None = Field(None, gt=0)
    stock: int | None = Field(None, ge=0)
    rating: float | None = Field(None, ge=0.00, le=5.0)
    regenerate_description: bool = False  # NEW
 
    @model_validator(mode="after")
    def check_request(self):
        real_fields = self.model_fields_set - {"regenerate_description"}
 
        if self.regenerate_description and "description" in real_fields:
            raise ValueError(
                "Don't send a description when regenerate_description is true"
            )
        if not real_fields and not self.regenerate_description:
            raise ValueError("At least one field must be provided")
        return self

#0 Hello
@app.get("/Hello")
def read_root():
    return "Hello World"
#1- GET & FILTER
@app.get("/products",response_model=list[Response])
async def get_product(category:str|None=None,
                min_price:float|None=Query(None,ge=0),
                max_price:float|None=Query(None,ge=0),
                conn=Depends(get_db)):
        try:
                query='''SELECT id,name,description,category,
                price,stock,rating FROM products WHERE 1=1'''
                values=[]
                if category:
                    query+=" AND category ILIKE $1"
                    values.append(category)
                if min_price is not None:
                    query+=f" AND price>=${(len(values))+1}"
                    values.append(min_price)
                if max_price is not None:
                    query+=f" AND price<=${(len(values))+1}"
                    values.append(max_price)
                query+=" ORDER BY id ASC"
                row=await conn.fetch(query,*values)
                return [dict(r) for r in row]
        except asyncpg.PostgresSyntaxError:
            logger.exception("Bad SQL syntax in get_product query")
            raise HTTPException(status_code=500, detail="Database error")
        except asyncpg.PostgresConnectionError:
            logger.exception("Database connection failed in get_product")
            raise HTTPException(status_code=503, detail="Database unavailable")
        except HTTPException:
           raise
        except Exception:
            logger.exception("Unexpected error in get_product")
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
    )
            

#2- GET STATS
@app.get("/products/stats")
async def get_product_stats(conn=Depends(get_db)):
    try:
        query='''SELECT
            category,
            ROUND(AVG(price),2) AS average_price,
            SUM(stock) AS total_stock,
            COUNT(*) AS total_products
            FROM products GROUP BY category 
            ORDER BY total_products DESC
            '''
        row=await conn.fetch(query)
        if not row:
             raise HTTPException(
                status_code=404,
                detail="No Details found for Product_Stats"
            ) 
        return [dict(r) for r in row]
    except asyncpg.PostgresSyntaxError:
                logger.exception("Bad SQL syntax in get_product query")
                raise HTTPException(status_code=500, detail="Database error")
    except asyncpg.PostgresConnectionError:
        logger.exception("Database connection failed in get_product")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in get_product")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
)


#3- POST endpoint
@app.post("/products",response_model=list[Response], dependencies=[Depends(verify_api_key)])
async def create_products(products:list[Create_Product],conn=Depends(get_db)):
    try:
        if not products:
            raise HTTPException(status_code=400, detail="No products provided for insertion.")
        tasks = [
            asyncio.create_task(process_product(product)) # using create_task to use the methods --> .done/.cancel
            for product in products
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    
        for product,result in zip(products,results):
            if isinstance (result,Exception):
                logger.error("AI generation failed for product '%s':'%s'",product.name,result)
        for result in results:
            if isinstance(result,Exception):
                raise result
        
        names = [p.name for p in products]
        descriptions = [p.description for p in products]
        categories = [p.category for p in products]
        prices = [p.price for p in products]
        stocks = [p.stock for p in products]
        ratings = [p.rating for p in products]
        all_columns=[names,descriptions,categories,prices,stocks,ratings]
        query="""
                INSERT INTO products (
                    name, description, category, price, stock, rating
                )
                SELECT * FROM UNNEST(
                    $1::text[], 
                    $2::text[], 
                    $3::text[], 
                    $4::numeric[], 
                    $5::integer[], 
                    $6::numeric[]
                )
                RETURNING id,name,description,category,price,stock,rating;
            """
        rows=await conn.fetch(query,*all_columns)
        result = [dict(r) for r in rows]
        return result
    except HTTPException:
        raise
    except asyncpg.UniqueViolationError:
        logger.exception("Duplicate product encountered during insertion")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="One or more products already exist."
    )
    except (asyncpg.DataError, asyncpg.NotNullViolationError):
        logger.exception("Invalid product data during database insertion")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid product data."
        )
    except asyncpg.PostgresError:
        logger.exception("Database error while creating products")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A database error occurred."
        )
    except Exception:
        logger.exception("Unexpected error while creating products")
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred."
        )


# PATCH endpoint
@app.patch("/products/{product_id}", response_model=Response, dependencies=[Depends(verify_api_key)])
async def update_products(
    product_id: int,
    product: patch_product,
    conn=Depends(get_db),
):
    # the flag is not a DB column, so we keep it out of the SET clause
    data = product.model_dump(exclude_unset=True, exclude={"regenerate_description"})

    try:
        # --- NEW: generate the description BEFORE touching the DB,
        # so if Gemini fails the row stays unchanged.
        if product.regenerate_description:
            existing = await conn.fetchrow(
                "SELECT name, category FROM products WHERE id = $1", product_id
            )
            if not existing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Product Not Found",
                )
            # use the new value if it's being changed in this same request
            name = data.get("name", existing["name"])
            category = data.get("category", existing["category"])
            if name is None:
                raise HTTPException(status_code=400, detail="name cannot be null")
                
            data["description"] = await describe_with_ai(name, category)

        # --- same dynamic UPDATE as before
        to_update = []
        values = []
        for idx, (key, value) in enumerate(data.items(), start=1):
            to_update.append(f"{key} = ${idx}")
            values.append(value)
        values.append(product_id)
        where_idx = len(to_update) + 1

        query = f"""
            UPDATE products
            SET {", ".join(to_update)}
            WHERE id = ${where_idx}
            RETURNING id, name, description, category, price, stock, rating;
        """
        row = await conn.fetchrow(query, *values)  # one statement = already atomic

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
# DELETE 
@app.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT,  dependencies=[Depends(verify_api_key)])
async def delete_product(
    product_id: int,
    conn: asyncpg.Connection = Depends(get_db)
):
    query = """
        DELETE FROM products
        WHERE id = $1
        RETURNING id;
    """
    try:
        async with conn.transaction():
            row = await conn.fetchrow(query, product_id)

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Product Not Found"
                )

    except HTTPException:
        raise

    except asyncpg.PostgresError:
        logger.exception(
            f"Database error while deleting product {product_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

    except Exception:
        logger.exception(
            f"Unexpected error while deleting product {product_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error"
        )









