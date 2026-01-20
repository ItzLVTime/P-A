from openai import OpenAI
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Retry configuration - can be overridden via environment variables
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_MIN_WAIT = int(os.getenv("LLM_RETRY_MIN_WAIT", "1"))  # seconds
RETRY_MAX_WAIT = int(os.getenv("LLM_RETRY_MAX_WAIT", "10"))  # seconds

class LLMError(Exception):
    """Raised when LLM operations fail."""
    pass


class LLMRetryableError(Exception):
    """Raised for errors that should trigger a retry (network issues, timeouts, etc.)."""
    pass

def get_client():
    """Get configured OpenAI client for LiteLLM."""
    api_base = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")

    if not api_base:
        logger.error("LITELLM_API_BASE not set in environment")
        raise LLMError("LITELLM_API_BASE not set in environment")
    if not api_key:
        logger.error("LITELLM_API_KEY not set in environment")
        raise LLMError("LITELLM_API_KEY not set in environment")

    logger.debug(f"Creating LLM client with base URL: {api_base}")
    return OpenAI(api_key=api_key, base_url=api_base)

def _is_retryable_error(exception):
    """Check if an error is retryable (network issues, timeouts, server errors)."""
    error_str = str(exception).lower()
    retryable_keywords = [
        "timeout", "timed out",
        "connection", "network",
        "503", "502", "504",  # Server errors
        "rate limit", "too many requests",
        "temporary", "unavailable"
    ]
    return any(keyword in error_str for keyword in retryable_keywords)


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type(LLMRetryableError),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def _call_llm_with_retry(client, model, prompt, temperature):
    """Internal function that makes the actual LLM call with retry logic."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return resp.choices[0].message.content
    except Exception as e:
        if _is_retryable_error(e):
            logger.warning(f"Retryable error occurred: {e}")
            raise LLMRetryableError(str(e))
        else:
            # Non-retryable error, raise immediately
            raise


def ask_llm(prompt, model=None, temperature=0.2):
    """
    Send a prompt to the LLM and get a response.
    Automatically retries on network errors, timeouts, and server issues.

    Args:
        prompt: The prompt text
        model: Model name (default from LITELLM_MODEL env or fallback)
        temperature: Sampling temperature (default 0.2)

    Returns:
        Response text from the LLM

    Raises:
        LLMError: If the API call fails after all retries
    """
    if model is None:
        model = os.getenv("LITELLM_MODEL", "ollama/qwen2.5:7b-instruct-q4_k_m")

    logger.info(f"Sending request to LLM (model: {model})")
    logger.debug(f"Prompt length: {len(prompt)} characters")

    try:
        client = get_client()
        response = _call_llm_with_retry(client, model, prompt, temperature)
        logger.info(f"LLM response received ({len(response)} characters)")
        return response
    except LLMError:
        raise
    except RetryError as e:
        # This is raised when all retries are exhausted
        original_error = e.last_attempt.exception() if e.last_attempt else e
        logger.error(f"LLM request failed after {MAX_RETRIES} retries: {original_error}")
        raise LLMError(f"LLM request failed after {MAX_RETRIES} retries. Please check if Ollama/LiteLLM is running.")
    except LLMRetryableError as e:
        logger.error(f"LLM request failed: {e}")
        raise LLMError(f"LLM request failed: {e}")
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        raise LLMError(f"LLM request failed: {e}")
