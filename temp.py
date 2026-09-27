import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException,status,Query,Depends
from pydantic import BaseModel, Field, model_validator
import logging
import asyncpg
import asyncio
from contextlib import asynccontextmanager
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
load_dotenv()
@asynccontextmanager
async def lifespan(app: FastAPI):
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

async def get_db():
    async with app.state.pool.acquire() as conn:
        yield conn
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
# Patch Model
class patch_product(BaseModel):
    name:str|None=Field(None,min_length=2,max_length=100)
    description:str|None=None
    category:str|None=None
    price:float|None=Field(None,gt=0)
    stock:int|None=Field(None,ge=0)
    rating:float|None=Field(None,ge=0.00,le=5.0)
    @model_validator(mode="after")
    def check_at_least_one_field(self):
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided")
        return self

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
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail="No products found"
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
    
#3- CREATE
@app.post("/products",response_model=list[Response])
async def create_products(products:list[Create_Product],conn=Depends(get_db)):
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
    try:
        if not products:
            raise HTTPException(status_code=400, detail="No products provided for insertion.")
        rows=await conn.fetch(
             query,
             *all_columns
        )
        result = [dict(r) for r in rows]
        return result
    except HTTPException:
        
        raise
    except (asyncpg.DataError, asyncpg.NotNullViolationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format or missing required fields: {str(e)}"
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error occurred: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create products: {str(e)}"
        )

        
# PATCH 
@app.patch("/products/{product_id}",response_model=Response)
async def update_products(product_id:int,product:patch_product,conn=Depends(get_db)):
    data = product.model_dump(exclude_unset=True)
    to_update = []
    values = []
    for idx, (key, value) in enumerate(data.items(), start=1):
        to_update.append(f"{key} = ${idx}")
        values.append(value)
    values.append(product_id)
    where_idx=len(to_update)+1
    query=f"""
    UPDATE products
    SET {", ".join(to_update)}
    WHERE id = ${where_idx}
    RETURNING id, name, description, category, price, stock, rating;
"""
    try:
        async with conn.transaction():
            row=await conn.fetchrow(query,*values)
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product Not Found"
            )
        return dict(row)
    except HTTPException:
        raise
    except asyncpg.UniqueViolationError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A product with this name already exists: {e.detail or str(e)}"
        )
    except (asyncpg.DataError, asyncpg.NotNullViolationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data format: {str(e)}"
        )
    except asyncpg.PostgresError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database execution failed: {str(e)}"
        )
        
# DELETE 
@app.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
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









