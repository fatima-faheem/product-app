import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from asgi_lifespan import LifespanManager
import os 

from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env.test"

if not ENV_FILE.exists():
  raise FileNotFoundError(
      f"CRITICAL: Test environment file not found at {ENV_FILE}. "
      "Create a .env.test file in the root folder!"
  )
print(f"--> Loading env from: {ENV_FILE}")

load_dotenv(dotenv_path=ENV_FILE, override=True)

TEST_API_KEY=os.getenv("PRODUCT_API_KEY")
assert TEST_API_KEY == "test-secret-key"
os.environ["DB_NAME"] = "my_app_test"

assert os.environ["DB_NAME"] == "my_app_test", (
    "Tests must use the test database!"
)
from main  import app

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def client():
    async with LifespanManager(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"X-API-Key": TEST_API_KEY},      # <- default key
        ) as c:
            yield c

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def anon_client(client):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

@pytest_asyncio.fixture(autouse=True,loop_scope="session")
async def clean_database():
  if not hasattr(app.state, "pool"):
    yield
    return
  async with app.state.pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE TABLE products RESTART IDENTITY CASCADE"
        )
  yield