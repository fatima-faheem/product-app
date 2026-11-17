import hmac
from fastapi import Header, HTTPException
import os 
from dotenv import load_dotenv
load_dotenv()

#---------API KEY----------
PRODUCT_API_KEY = os.getenv("PRODUCT_API_KEY")


async def verify_api_key(
    x_api_key: str | None = Header(default=None)
):
    if not PRODUCT_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key is not configured"
        )

    if (
    x_api_key is None
    or not hmac.compare_digest(
        x_api_key.encode("utf-8"),
        PRODUCT_API_KEY.encode("utf-8")
    )
):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )