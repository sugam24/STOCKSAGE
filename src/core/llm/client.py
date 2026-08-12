import os
import sys
import time

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)
from groq.types.chat import ChatCompletionMessageParam

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MAX_ATTEMPTS = 3
INITIAL_WAIT_SECONDS = 2

RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)


def _stream_completion(client: Groq, messages: list[ChatCompletionMessageParam]) -> str:
    """Execute a streaming chat completion and return the full response text."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000,
        stream=True,
    )

    parts: list[str] = []
    usage = None

    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            parts.append(delta)

    full_response = "".join(parts)

    if usage:
        print("\n\n--- Usage Metadata ---")
        print(f"Input Tokens:  {usage.prompt_tokens}")
        print(f"Output Tokens: {usage.completion_tokens}")
        print("-----------------------\n")

    return full_response


def _call_with_backoff(
    client: Groq,
    messages: list[ChatCompletionMessageParam],
    *,
    retry_on: tuple[type[Exception], ...] = RETRYABLE_ERRORS,
) -> str:
    """
    Call Groq with exponential backoff on transient failures.

    Waits 2s before the 2nd attempt, 4s before the 3rd (doubling each time).
    """
    wait_seconds = INITIAL_WAIT_SECONDS
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _stream_completion(client, messages)
        except retry_on as exc:
            last_error = exc

            if isinstance(exc, (APIConnectionError, APITimeoutError)):
                error_kind = "Connection error"
            elif isinstance(exc, RateLimitError):
                error_kind = "Rate limit error"
            elif isinstance(exc, AuthenticationError):
                error_kind = "Authentication error"
            else:
                error_kind = type(exc).__name__

            if attempt < MAX_ATTEMPTS:
                print(
                    f"\n[{error_kind}] Attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}"
                )
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                wait_seconds *= 2
            else:
                print(
                    f"\n[{error_kind}] Attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}"
                )

    assert last_error is not None
    raise RuntimeError(
        f"Groq API request failed after {MAX_ATTEMPTS} attempts. "
        f"Last error ({type(last_error).__name__}): {last_error}"
    ) from last_error


def get_llm_response(
    prompt: str,
    system_prompt: str = "You are a helpful financial assistant.",
    *,
    api_key: str | None = None,
) -> str:
    """
    Sends a single prompt to Groq using the official Groq SDK.

    Args:
        prompt:        The user's question or instruction.
        system_prompt: System-level instruction for the model persona.
        api_key:       Optional override for testing (defaults to GROQ_API_KEY).

    Returns:
        The model's text response.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
        RuntimeError: If all retry attempts are exhausted.
    """
    resolved_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    client = Groq(api_key=resolved_key)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    return _call_with_backoff(client, messages)


def demo_bad_api_key() -> None:
    """Step 69: retry with an invalid key, then raise a clear final error."""
    print("=" * 60)
    print("  Step 69: Bad API key retry demo")
    print("=" * 60)
    print("Using an intentionally invalid API key...\n")

    bad_client = Groq(api_key="invalid-key-for-retry-demo")
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    try:
        _call_with_backoff(
            bad_client,
            messages,
            retry_on=(*RETRYABLE_ERRORS, AuthenticationError),
        )
    except RuntimeError as exc:
        print(f"\nFinal error after {MAX_ATTEMPTS} attempts:")
        print(f"  {exc}")
    except AuthenticationError as exc:
        print(f"\nFinal error after {MAX_ATTEMPTS} attempts:")
        print(f"  Authentication failed: {exc}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-bad-key":
        demo_bad_api_key()
        sys.exit(0)

    test_prompt = "What is a stock split, and how does it affect a retail investor?"
    print(f"Sending prompt to Groq: '{test_prompt}'...")
    try:
        print("Groq's Response:\n")
        reply = get_llm_response(test_prompt)
        print(f"\n\n[Accumulated {len(reply)} characters]")
    except (RuntimeError, AuthenticationError, RateLimitError, APIConnectionError) as exc:
        print(f"Failed to get response: {exc}")
