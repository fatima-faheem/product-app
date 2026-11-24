import time
import asyncio
import logging
import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")  

client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1) 
    )
)

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


# ---------- our own errors (no HTTP here) ----------
class AIError(Exception):
    def __init__(self, message: str = "AI error", code: int | None = None):
        super().__init__(message)
        self.message = message
        self.code = code


class AITimeoutError(AIError):
    """Gemini did not answer in time, after all retries."""


class AIBlockedError(AIError):
    """Prompt or output blocked by safety filters."""

    
class AIQuotaError(AIError):
    """Gemini quota or rate limit reached."""

    def __init__(self, message: str = "AI quota reached", kind: str = "unknown"):
        super().__init__(message, code=429)
        self.kind = kind


class AIUpstreamError(AIError):
    """Gemini failed, was truncated, or returned nothing usable."""



# ---------- 429 helper ----------
def quota_kind(exc: errors.APIError) -> str:
    details = exc.details

    if not isinstance(details, dict):
        return "unknown"

    error_data = details.get("error", details)

    if not isinstance(error_data, dict):
        return "unknown"

    error_details = error_data.get("details", [])

    for item in error_details:
        if not isinstance(item, dict):
            continue

        if not item.get("@type", "").endswith("QuotaFailure"):
            continue
        
        violations = item.get("violations", [])

        if not isinstance(violations, list):
            continue

        for violation in violations:
            if not isinstance(violation, dict):
                continue

            quota_id = violation.get("quotaId", "").lower()

            if "perday" in quota_id:
                return "daily"

            if "perminute" in quota_id:
                return "rate"

    return "unknown"



# ---------- retry ----------
def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True

    if isinstance(exc, errors.APIError):
        if exc.code == 429 and quota_kind(exc) == "daily":
            return False

        return exc.code in RETRYABLE_CODES

    return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=6, jitter=1),
    retry=retry_if_exception(is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

async def _call_with_retry(
    prompt: str,
    config: types.GenerateContentConfig,
    timeout: float
):
    attempt_started = time.perf_counter()

    logger.info(
        "Gemini attempt started | model=%s",
        MODEL
    )

    try:
        async with asyncio.timeout(timeout):
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )

    except TimeoutError:
        elapsed = time.perf_counter() - attempt_started

        logger.warning(
            "Gemini attempt timed out | elapsed=%.2fs",
            elapsed
        )
        raise

    except errors.APIError as exc:
        elapsed = time.perf_counter() - attempt_started

        logger.warning(
            "Gemini API error | code=%s | elapsed=%.2fs",
            exc.code,
            elapsed
        )
        raise

    else:
        elapsed = time.perf_counter() - attempt_started

        logger.info(
            "Gemini attempt completed | elapsed=%.2fs",
            elapsed
        )

        return response



# ---------- the function endpoints use ----------
async def call_gemini(
    prompt: str,
    *,
    response_schema=None,   # a Pydantic model forces JSON output
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float = 30.0,
) -> str:
    """Return the model's text, or raise an AIError subclass."""
    function_started = time.perf_counter()
    config_started = time.perf_counter()

    config_kwargs = {
        "temperature": temperature,
        "automatic_function_calling": (
            types.AutomaticFunctionCallingConfig(disable=True)
        ),
    }

    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens

    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_kwargs)

    logger.info(
        "Gemini config created | elapsed=%.3fs",
        time.perf_counter() - config_started,
    )

    logger.info(
        "Entering retry function | elapsed_since_call=%.3fs",
        time.perf_counter() - function_started,
    )
    # 1) call Gemini (with retries); translate library errors into ours
    try:
        response = await _call_with_retry(prompt, config, timeout)
    except TimeoutError as exc:
        logger.error("Gemini timed out after retries")
        raise AITimeoutError("AI service took too long to respond") from exc
    except errors.APIError as exc:
        if exc.code == 429:
            kind = quota_kind(exc)

            logger.error(
                "Gemini 429 (%s): %s",
                kind,
                exc.message,
            )

            raise AIQuotaError(
                "AI quota reached",
                kind=kind,
            ) from exc

        logger.error(
            "Gemini API error [%s]: %s",
            exc.code,
            exc.message,
        )

        raise AIUpstreamError(
            f"Gemini API error {exc.code}",
            code=exc.code,
        ) from exc
    # 2) no candidates: usually the prompt was blocked
    if not response.candidates:
        feedback = response.prompt_feedback
        reason = feedback.block_reason if feedback and feedback.block_reason else None
        if reason:
            logger.warning("Prompt blocked: %s", reason)
            raise AIBlockedError(f"Prompt blocked ({reason})")
        raise AIUpstreamError("Gemini returned no candidates")

    # 3) why did the model stop?
    finish = response.candidates[0].finish_reason
    if finish == types.FinishReason.SAFETY:
        logger.warning("Output blocked by safety filters")
        raise AIBlockedError("Output blocked by safety filters")
    if finish == types.FinishReason.MAX_TOKENS:
        logger.error("Output truncated (max tokens)")
        raise AIUpstreamError("AI response was cut off")
    if finish not in (None, types.FinishReason.STOP):
        logger.error("Unexpected finish reason: %s", finish)
        raise AIUpstreamError(f"Unexpected finish reason: {finish}")

    # 4) empty text
    text = response.text
    if not text or not text.strip():
        raise AIUpstreamError("Gemini returned empty text")

    return text.strip()