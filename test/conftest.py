import os
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from dotenv import dotenv_values, load_dotenv
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# 1) Load ONLY the test environment, and make sure nothing can leak in from
#    your normal .env (main.py calls load_dotenv() without override, so any
#    variable missing from .env.test would be silently filled from .env).
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env.test"
TEST_DB_NAME = "my_app_test"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

if not ENV_FILE.exists():
    raise FileNotFoundError(
        f"CRITICAL: Test environment file not found at {ENV_FILE}. "
        "Create a .env.test file in the root folder!"
    )

print(f"--> Loading env from: {ENV_FILE}")

_required = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "PRODUCT_API_KEY"]
_in_file = dotenv_values(ENV_FILE)
_missing = [name for name in _required if name not in _in_file]
if _missing:
    raise RuntimeError(
        ".env.test must define: " + ", ".join(_missing) + ". "
        "Otherwise main.py could fill them in from your normal .env."
    )

load_dotenv(dotenv_path=ENV_FILE, override=True)
os.environ["DB_NAME"] = TEST_DB_NAME

if os.environ["DB_HOST"] not in LOCAL_HOSTS:
    raise RuntimeError(
        f"UNSAFE: DB_HOST is {os.environ['DB_HOST']!r}. Tests may only run "
        f"against a local database ({', '.join(sorted(LOCAL_HOSTS))})."
    )

TEST_API_KEY = os.environ["PRODUCT_API_KEY"]
assert TEST_API_KEY == "test-secret-key", "Incorrect test API key!"

# Import the app only AFTER the environment is fixed.
from main import app  # noqa: E402


# ---------------------------------------------------------------------------
# 2) Clients
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def client():
    async with LifespanManager(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"X-API-Key": TEST_API_KEY},
        ) as c:
            yield c


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def anon_client(client):
    """Same app, no API key header. Depends on `client` so the lifespan
    (and therefore the DB pool) is already running."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# 3) Database safety + cleanup
#    These use their OWN short connection (not app.state.pool), so tests that
#    never touch the app (like the Gemini helper tests) do not start it.
# ---------------------------------------------------------------------------
async def _connect():
    return await asyncpg.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        database=TEST_DB_NAME,
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def verified_test_database():
    """Runs once. Aborts the WHOLE test run if this is not the local test DB."""
    conn = await _connect()
    try:
        info = await conn.fetchrow(
            "SELECT current_database() AS database, "
            "inet_server_addr() AS server_address"
        )
    finally:
        await conn.close()

    database = info["database"]
    server_address = str(info["server_address"])

    if database != TEST_DB_NAME:
        pytest.exit(
            f"UNSAFE DATABASE: expected {TEST_DB_NAME}, connected to "
            f"{database}. Aborting before anything is truncated.",
            returncode=2,
        )
    if server_address not in ("127.0.0.1", "::1"):
        pytest.exit(
            f"UNSAFE SERVER: connected to {server_address}. Aborting before "
            "anything is truncated.",
            returncode=2,
        )
    return True


@pytest_asyncio.fixture(autouse=True, loop_scope="session")
async def clean_database(verified_test_database):
    conn = await _connect()
    try:
        await conn.execute("TRUNCATE TABLE products RESTART IDENTITY CASCADE")
    finally:
        await conn.close()
    yield
