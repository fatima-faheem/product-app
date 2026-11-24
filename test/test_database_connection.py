import pytest
from main import app 
@pytest.mark.asyncio
async def test_database_connection_details(client):
    async with app.state.pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT
                current_database() AS database,
                current_user AS username,
                inet_server_addr() AS server_address,
                inet_server_port() AS server_port,
                current_schema() AS schema
        """)

        print(dict(result))

        assert result["database"] == "my_app_test"
        
@pytest.mark.asyncio
async def test_print_test_database_schema(client):
    from main import app

    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = 'products'
              AND table_schema = 'public'
            ORDER BY ordinal_position
        """)

        for row in rows:
            print(dict(row))
