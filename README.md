
# AI Product Management API

A backend application built with FastAPI and PostgreSQL for managing product inventories, featuring AI-powered product description generation using Google's Gemini API.

## Features

- Create products individually or in bulk.
- Update and retrieve product information.
- Generate and regenerate product descriptions using Gemini AI.
- Filter products by category and price range.
- View product statistics.
- Validate incoming data using Pydantic.
- Handle AI API errors, timeouts, and retries.
- Prevent partial database insertion when AI generation fails.
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
- Git & GitHub

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/fatima-faheem/product-app.git
cd product-app
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` as a reference.

Add your database credentials and Gemini API key.

### 5. Run the application

```bash
uvicorn main:app --reload
```

### 6. Access API documentation

Open:

http://127.0.0.1:8000/docs

## Running Tests

Make sure your test database is configured before running:

```bash
pytest -v
```

## Environment Variables

The application requires database configuration and a Gemini API key. Refer to `.env.example` for the required variable names.

