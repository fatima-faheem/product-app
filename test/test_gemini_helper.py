
from types import SimpleNamespace

import pytest
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import wait_none

import gemini_helper as gh

from gemini_helper import (
    AIBlockedError,
    AITimeoutError,
    AIUpstreamError,
    call_gemini,
    is_retryable,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def fake_response(
    text="Hello world",
    finish=types.FinishReason.STOP,
    candidates=True,
    block_reason=None,
):
    feedback = (
        SimpleNamespace(block_reason=block_reason)
        if block_reason
        else None
    )

    return SimpleNamespace(
        candidates=[
            SimpleNamespace(finish_reason=finish)
        ] if candidates else [],
        text=text,
        prompt_feedback=feedback,
    )


def api_error(code):
    return errors.APIError(
        code,
        {"error": {"message": "boom", "status": "X"}}
    )


def mock_call(monkeypatch, *, response=None, exc=None):
    """Replace _call_with_retry for ordinary helper tests."""

    seen = {}

    async def fake(prompt, config, timeout):
        seen["prompt"] = prompt
        seen["config"] = config
        seen["timeout"] = timeout

        if exc is not None:
            raise exc

        return response if response is not None else fake_response()

    monkeypatch.setattr(
        "gemini_helper._call_with_retry",
        fake
    )

    return seen


# --------------------------------------------------
# Success tests
# --------------------------------------------------

async def test_returns_text(monkeypatch):
    mock_call(
        monkeypatch,
        response=fake_response(text="A great product.")
    )

    assert await call_gemini("write something") == "A great product."


async def test_text_is_stripped(monkeypatch):
    mock_call(
        monkeypatch,
        response=fake_response(text="  padded text \n")
    )

    assert await call_gemini("x") == "padded text"


async def test_prompt_and_timeout_are_passed_through(monkeypatch):
    seen = mock_call(monkeypatch)

    await call_gemini("my prompt", timeout=3.5)

    assert seen["prompt"] == "my prompt"
    assert seen["timeout"] == 3.5


async def test_default_config_has_no_json_mode(monkeypatch):
    seen = mock_call(monkeypatch)

    await call_gemini("x", temperature=0.2)

    assert seen["config"].temperature == 0.2
    assert not seen["config"].response_mime_type


class Filters(BaseModel):
    category: str | None = None


async def test_response_schema_turns_on_json_mode(monkeypatch):
    seen = mock_call(
        monkeypatch,
        response=fake_response(text='{"category": "x"}')
    )

    await call_gemini("x", response_schema=Filters)

    assert seen["config"].response_mime_type == "application/json"


# --------------------------------------------------
# Failure tests
# --------------------------------------------------

# def mock_call(monkeypatch, *, response=None, exc=None):
#     """Replace _call_with_retry for ordinary helper tests."""

#     seen = {}

#     async def fake(prompt, config, timeout):
#         seen["prompt"] = prompt
#         seen["config"] = config
#         seen["timeout"] = timeout

#         if exc is not None:
#             raise exc

#         return response if response is not None else fake_response()

#     monkeypatch.setattr(
#         "gemini_helper._call_with_retry",
#         fake
#     )

#     return seen

@pytest.mark.parametrize(
    "mock_kwargs, expected_error",
    [
        ({"exc": TimeoutError()}, AITimeoutError),
        ({"exc": api_error(503)}, AIUpstreamError),

        (
            {
                "response": fake_response(
                    candidates=False,
                    block_reason="SAFETY"
                )
            },
            AIBlockedError
        ),

        (
            {"response": fake_response(candidates=False)},
            AIUpstreamError
        ),

        (
            {
                "response": fake_response(
                    finish=types.FinishReason.SAFETY
                )
            },
            AIBlockedError
        ),

        (
            {
                "response": fake_response(
                    finish=types.FinishReason.MAX_TOKENS
                )
            },
            AIUpstreamError
        ),

        (
            {
                "response": fake_response(
                    finish=types.FinishReason.RECITATION
                )
            },
            AIUpstreamError
        ),

        ({"response": fake_response(text=None)}, AIUpstreamError),
        ({"response": fake_response(text="")}, AIUpstreamError),
        ({"response": fake_response(text="   \n ")}, AIUpstreamError),
    ],
    ids=[
        "timeout",
        "api-error",
        "prompt-blocked",
        "no-candidates-no-reason",
        "safety-finish",
        "max-tokens",
        "unexpected-finish",
        "text-none",
        "text-empty",
        "text-whitespace",
    ],
)
async def test_failures_raise_the_right_ai_error(
    monkeypatch,
    mock_kwargs,
    expected_error
):
    mock_call(monkeypatch, **mock_kwargs)

    with pytest.raises(expected_error):
        await call_gemini("x")


async def test_api_error_keeps_the_status_code(monkeypatch):
    mock_call(monkeypatch, exc=api_error(429))

    with pytest.raises(AIUpstreamError) as info:
        await call_gemini("x")

    assert info.value.code == 429


async def test_timeout_is_not_reported_as_upstream_error(monkeypatch):
    mock_call(monkeypatch, exc=TimeoutError())

    with pytest.raises(AITimeoutError) as info:
        await call_gemini("x")

    assert not isinstance(info.value, AIUpstreamError)


# --------------------------------------------------
# Retry predicate tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "code",
    [408, 429, 500, 502, 503, 504]
)
def test_retryable_status_codes(code):
    assert is_retryable(api_error(code)) is True


@pytest.mark.parametrize(
    "code",
    [400, 401, 403, 404]
)
def test_non_retryable_status_codes(code):
    assert is_retryable(api_error(code)) is False


def test_timeout_is_retryable():
    assert is_retryable(TimeoutError()) is True


def test_other_exceptions_are_not_retryable():
    assert is_retryable(ValueError("boom")) is False


# --------------------------------------------------
# Helpers for actual retry behavior
# --------------------------------------------------

def patch_client(monkeypatch, outcomes):
    """
    Replace the actual Gemini API call.

    Keep the real Tenacity retry mechanism active.
    """

    calls = {"n": 0}

    async def fake_generate_content(**kwargs):
        outcome = outcomes[calls["n"]]
        calls["n"] += 1

        if isinstance(outcome, Exception):
            raise outcome

        return outcome

    monkeypatch.setattr(
        "gemini_helper.client.aio.models.generate_content",
        fake_generate_content
    )

    monkeypatch.setattr(
        "gemini_helper._call_with_retry",
        gh._call_with_retry.retry_with(wait=wait_none())
    )
    return calls


# --------------------------------------------------
# Actual retry tests
# --------------------------------------------------

async def test_retries_a_transient_error_then_succeeds(monkeypatch):

    calls = patch_client(
        monkeypatch,
        [
            api_error(503),
            api_error(503),
            fake_response(text="ok")
        ]
    )

    assert await call_gemini("x") == "ok"
    assert calls["n"] == 3


async def test_does_not_retry_a_non_retryable_error(monkeypatch):

    calls = patch_client(
        monkeypatch,
        [
            api_error(400),
            fake_response()
        ]
    )

    with pytest.raises(AIUpstreamError):
        await call_gemini("x")

    assert calls["n"] == 1


async def test_gives_up_after_three_attempts(monkeypatch):

    calls = patch_client(
        monkeypatch,
        [api_error(503)] * 5
    )

    with pytest.raises(AIUpstreamError):
        await call_gemini("x")

    assert calls["n"] == 3
