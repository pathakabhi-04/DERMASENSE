from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# openai/gpt-oss-120b: the largest general-purpose instruction model this
# API key has access to (checked via GET /openai/v1/models -- llama-3.3
# and llama-3.1 chat models are not enabled on this account/region).
# Chosen over gpt-oss-20b because this adapter drives constrained medical
# paraphrase + grounding (spec section 4.2), where faithfulness to the
# supplied evidence matters more than the latency difference between the
# two sizes, and over groq/compound because that model can invoke tools
# (e.g. web search) on its own, which is undesirable for a task that must
# stay strictly grounded in the evidence passed in the prompt.
DEFAULT_MODEL = "openai/gpt-oss-120b"
# Low, not zero: near-deterministic phrasing (reduces run-to-run variance
# that otherwise makes the downstream banned-phrase safety check pass or
# fail on the same query), while avoiding the repetition/looping some
# hosted models exhibit at temperature=0.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4
MAX_BACKOFF_SECONDS = 16.0
RETRYABLE_HTTP_CODES = {429, 500, 503}
API_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMGenerationError(Exception):
    """
    Raised when the hosted LLM call fails: network error, timeout,
    a non-retryable HTTP error, or a response with no usable text
    (e.g. blocked by content filters).

    Per the primary specification (section 5, generation-failure
    fallback): callers must treat this as a signal to fall back to
    unnarrated evidence, never let it propagate as an unhandled
    crash to the user.
    """


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str | None


def _load_api_key_from_dotenv(env_path: Path) -> str | None:
    """
    Minimal .env parser for GROQ_API_KEY only. Avoids adding a
    python-dotenv dependency for a single variable.
    """

    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("GROQ_API_KEY="):
            value = line.split("=", 1)[1].strip()
            return value or None

    return None


def resolve_api_key(env_path: str | Path = ".env") -> str:
    """
    Resolve the Groq API key. A real environment variable takes
    precedence over the .env file.
    """

    key = os.environ.get("GROQ_API_KEY")

    if key:
        return key

    key = _load_api_key_from_dotenv(Path(env_path))

    if not key:
        raise LLMGenerationError(
            "GROQ_API_KEY is not set. Set it as an environment "
            "variable or in a .env file at the project root."
        )

    return key


class GroqAdapter:
    """
    Thin adapter over Groq's OpenAI-compatible chat completions API.

    Per the primary specification (section 4.2): the baseline uses a
    hosted API model, not local or self-hosted inference. This
    adapter is intentionally minimal -- one system+user prompt pair
    in, one text response out -- mirroring the previous Gemini
    adapter's interface so the rest of the pipeline needed no
    changes beyond the import.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.api_key = api_key or resolve_api_key()
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """
        Send one system+user prompt pair to Groq and return the
        model's text response.

        Retries on transient errors (429 rate limit, 500/503 server
        errors) with exponential backoff capped at
        MAX_BACKOFF_SECONDS, up to max_retries times. All other
        failures raise immediately.
        """

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        body = json.dumps(payload).encode("utf-8")

        last_error: LLMGenerationError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                data = self._send_request(body)
                return self._parse_response(data)

            except LLMGenerationError as error:
                last_error = error

                if not self._is_retryable(error) or (
                    attempt == self.max_retries
                ):
                    raise

                time.sleep(min(2 ** attempt, MAX_BACKOFF_SECONDS))

        # Unreachable, but keeps type-checkers satisfied.
        raise last_error

    def _send_request(self, body: bytes) -> dict:
        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "dermasense-rag/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                return json.loads(response.read())

        except urllib.error.HTTPError as error:
            raise LLMGenerationError(
                f"Groq API returned HTTP {error.code}: "
                f"{self._safe_error_body(error)}"
            ) from error

        except urllib.error.URLError as error:
            raise LLMGenerationError(
                f"Groq API request failed: {error.reason}"
            ) from error

        except TimeoutError as error:
            raise LLMGenerationError(
                "Groq API request timed out."
            ) from error

    def _is_retryable(self, error: LLMGenerationError) -> bool:
        """
        Retry transient failures, whether they came back as an HTTP
        status or never reached the server at all.

        Connection-level failures were originally not retried, only
        429/500/503 were. That is inconsistent: a DNS hiccup or a reset
        connection is as transient as a 503, and arguably more so. On a
        flaky link it cost a whole request each time -- measured during
        a Phase 1 gate run where 13 of 16 queries failed with
        "[Errno -3] Temporary failure in name resolution" while DNS
        resolved fine seconds later. In production that is a user losing
        their answer to a momentary blip.

        Deliberately still NOT retried: 4xx other than 429 (the request
        itself is wrong, so repeating it changes nothing), and an empty
        or candidate-less response (the model replied; retrying invites
        a different answer to the same prompt rather than fixing a
        fault).
        """

        message = str(error)

        if any(f"HTTP {code}" in message for code in RETRYABLE_HTTP_CODES):
            return True

        return any(
            marker in message
            for marker in ("Groq API request failed:", "Groq API request timed out")
        )

    def _safe_error_body(
        self,
        error: urllib.error.HTTPError,
    ) -> str:
        try:
            raw = error.read().decode(errors="replace")
            return raw.replace(self.api_key, "<redacted>")
        except Exception:
            return "(error body unavailable)"

    def _parse_response(self, data: dict) -> LLMResponse:
        choices = data.get("choices") or []

        if not choices:
            raise LLMGenerationError(
                "Groq API returned no choices "
                "(the response may have been blocked by content "
                "filters)."
            )

        finish_reason = choices[0].get("finish_reason")
        text = choices[0].get("message", {}).get("content", "") or ""

        if not text.strip():
            raise LLMGenerationError(
                "Groq API returned an empty response "
                f"(finish_reason={finish_reason})."
            )

        return LLMResponse(
            text=text,
            model=self.model,
            finish_reason=finish_reason,
        )
