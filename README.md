# AI Product Management API

A backend application built with FastAPI and PostgreSQL for managing product inventories, with AI-powered product description generation using Google's Gemini API.

## Features

- Create and manage products.
- Bulk product creation.
- Update product information.
- Generate and regenerate product descriptions using Gemini AI.
- Retrieve products.
- Filter products by category and price.
- View product statistics.
- Validate incoming data using Pydantic.
- Handle AI API errors, timeouts and retries.
- Automated tests using Pytest.

## Technologies

- Python
- FastAPI
- PostgreSQL
- asyncpg
- Pydantic
- Google Gemini API
- Tenacity
- Pytest

## Project Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_DIRECTORY
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

For development:

```bash
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and provide your own database credentials and Gemini API key.

### 5. Run the application

```bash
uvicorn main:app --reload
```

### 6. Access API documentation

Open:

```text
http://127.0.0.1:8000/docs
```

## Running Tests

```bash
pytest -v
```

Make sure your test database is configured before running the test suite.

## Environment Variables

The application requires database configuration and a Gemini API key. See `.env.example` for the required variable names.

## License

Not specified yet.