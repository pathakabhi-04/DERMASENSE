from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
RETRYABLE_HTTP_CODES = {429, 500, 503}
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class LLMGenerationError(Exception):
    """
    Raised when the hosted LLM call fails: network error, timeout,
    a non-retryable HTTP error, or a response with no usable text
    (e.g. blocked by safety filters).

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
    Minimal .env parser for GEMINI_API_KEY only. Avoids adding a
    python-dotenv dependency for a single variable.
    """

    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("GEMINI_API_KEY="):
            value = line.split("=", 1)[1].strip()
            return value or None

    return None


def resolve_api_key(env_path: str | Path = ".env") -> str:
    """
    Resolve the Gemini API key. A real environment variable takes
    precedence over the .env file.
    """

    key = os.environ.get("GEMINI_API_KEY")

    if key:
        return key

    key = _load_api_key_from_dotenv(Path(env_path))

    if not key:
        raise LLMGenerationError(
            "GEMINI_API_KEY is not set. Set it as an environment "
            "variable or in a .env file at the project root."
        )

    return key


class GeminiAdapter:
    """
    Thin adapter over the Gemini generateContent REST API.

    Per the primary specification (section 4.2): the baseline uses a
    hosted API model, not local or self-hosted inference. This
    adapter is intentionally minimal -- one system+user prompt pair
    in, one text response out -- so it can be replaced with a
    different provider later without touching the rest of the
    pipeline.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.api_key = api_key or resolve_api_key()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """
        Send one system+user prompt pair to Gemini and return the
        model's text response.

        Retries a small, fixed number of times on transient errors
        (429 rate limit, 500/503 server errors) with a short backoff.
        All other failures raise immediately.
        """

        url = (
            f"{API_BASE_URL}/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
        }

        body = json.dumps(payload).encode("utf-8")

        last_error: LLMGenerationError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                data = self._send_request(url, body)
                return self._parse_response(data)

            except LLMGenerationError as error:
                last_error = error

                if not self._is_retryable(error) or (
                    attempt == self.max_retries
                ):
                    raise

                time.sleep(2 ** attempt)

        # Unreachable, but keeps type-checkers satisfied.
        raise last_error

    def _send_request(self, url: str, body: bytes) -> dict:
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
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
                f"Gemini API returned HTTP {error.code}: "
                f"{self._safe_error_body(error)}"
            ) from error

        except urllib.error.URLError as error:
            raise LLMGenerationError(
                f"Gemini API request failed: {error.reason}"
            ) from error

        except TimeoutError as error:
            raise LLMGenerationError(
                "Gemini API request timed out."
            ) from error

    def _is_retryable(self, error: LLMGenerationError) -> bool:
        message = str(error)
        return any(
            f"HTTP {code}" in message
            for code in RETRYABLE_HTTP_CODES
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
        candidates = data.get("candidates") or []

        if not candidates:
            raise LLMGenerationError(
                "Gemini API returned no candidates "
                "(the response may have been blocked by safety "
                "filters)."
            )

        finish_reason = candidates[0].get("finishReason")

        parts = candidates[0].get("content", {}).get("parts", [])

        text = "".join(part.get("text", "") for part in parts)

        if not text.strip():
            raise LLMGenerationError(
                "Gemini API returned an empty response "
                f"(finishReason={finish_reason})."
            )

        return LLMResponse(
            text=text,
            model=self.model,
            finish_reason=finish_reason,
        )
