import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from src.rag.llm.gemini_adapter import (
    GeminiAdapter,
    LLMGenerationError,
    resolve_api_key,
)


class FakeHTTPResponse:
    """Minimal stand-in for the object urlopen() returns."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_http_error(code: int, message: str = "error") -> urllib.error.HTTPError:
    body = json.dumps({"error": {"message": message}}).encode("utf-8")
    return urllib.error.HTTPError(
        url="https://example.com",
        code=code,
        msg=message,
        hdrs=None,
        fp=io.BytesIO(body),
    )


def make_success_payload(
    text: str = "Answer text.",
    finish_reason: str = "STOP",
) -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": finish_reason,
            }
        ]
    }


class ResolveApiKeyTests(unittest.TestCase):
    def test_prefers_real_environment_variable(self):
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "from-env"},
            clear=False,
        ):
            self.assertEqual(resolve_api_key(), "from-env")

    def test_falls_back_to_dotenv_file(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        with patch.dict("os.environ", {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                env_path = Path(tmp_dir) / ".env"
                env_path.write_text(
                    "GEMINI_API_KEY=from-dotenv\n",
                    encoding="utf-8",
                )

                self.assertEqual(
                    resolve_api_key(env_path),
                    "from-dotenv",
                )

    def test_raises_when_key_is_missing_everywhere(self):
        import tempfile
        from pathlib import Path

        with patch.dict("os.environ", {}, clear=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                missing_path = Path(tmp_dir) / ".env"

                with self.assertRaises(LLMGenerationError):
                    resolve_api_key(missing_path)


class GeminiAdapterGenerateTests(unittest.TestCase):
    def _adapter(self) -> GeminiAdapter:
        return GeminiAdapter(
            api_key="test-key",
            max_retries=2,
        )

    @patch("src.rag.llm.gemini_adapter.time.sleep", return_value=None)
    @patch("src.rag.llm.gemini_adapter.urllib.request.urlopen")
    def test_successful_generation_returns_text(
        self,
        mock_urlopen,
        mock_sleep,
    ):
        mock_urlopen.return_value = FakeHTTPResponse(
            make_success_payload("Hello there.")
        )

        adapter = self._adapter()
        result = adapter.generate("system", "user query")

        self.assertEqual(result.text, "Hello there.")
        self.assertEqual(result.finish_reason, "STOP")

    @patch("src.rag.llm.gemini_adapter.time.sleep", return_value=None)
    @patch("src.rag.llm.gemini_adapter.urllib.request.urlopen")
    def test_retries_on_503_then_succeeds(
        self,
        mock_urlopen,
        mock_sleep,
    ):
        mock_urlopen.side_effect = [
            make_http_error(503, "overloaded"),
            FakeHTTPResponse(make_success_payload("Recovered.")),
        ]

        adapter = self._adapter()
        result = adapter.generate("system", "user query")

        self.assertEqual(result.text, "Recovered.")
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("src.rag.llm.gemini_adapter.time.sleep", return_value=None)
    @patch("src.rag.llm.gemini_adapter.urllib.request.urlopen")
    def test_does_not_retry_on_non_retryable_error(
        self,
        mock_urlopen,
        mock_sleep,
    ):
        mock_urlopen.side_effect = make_http_error(400, "bad request")

        adapter = self._adapter()

        with self.assertRaises(LLMGenerationError):
            adapter.generate("system", "user query")

        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("src.rag.llm.gemini_adapter.time.sleep", return_value=None)
    @patch("src.rag.llm.gemini_adapter.urllib.request.urlopen")
    def test_gives_up_after_max_retries(
        self,
        mock_urlopen,
        mock_sleep,
    ):
        mock_urlopen.side_effect = make_http_error(503, "overloaded")

        adapter = self._adapter()

        with self.assertRaises(LLMGenerationError):
            adapter.generate("system", "user query")

        self.assertEqual(mock_urlopen.call_count, 3)

    @patch("src.rag.llm.gemini_adapter.urllib.request.urlopen")
    def test_no_candidates_raises(self, mock_urlopen):
        mock_urlopen.return_value = FakeHTTPResponse({"candidates": []})

        adapter = self._adapter()

        with self.assertRaises(LLMGenerationError):
            adapter.generate("system", "user query")

    @patch("src.rag.llm.gemini_adapter.urllib.request.urlopen")
    def test_empty_text_raises(self, mock_urlopen):
        mock_urlopen.return_value = FakeHTTPResponse(
            make_success_payload(text="", finish_reason="SAFETY")
        )

        adapter = self._adapter()

        with self.assertRaises(LLMGenerationError):
            adapter.generate("system", "user query")

    @patch("src.rag.llm.gemini_adapter.urllib.request.urlopen")
    def test_error_body_redacts_api_key(self, mock_urlopen):
        mock_urlopen.side_effect = make_http_error(
            400,
            "invalid key test-key",
        )

        adapter = self._adapter()

        with self.assertRaises(LLMGenerationError) as ctx:
            adapter.generate("system", "user query")

        self.assertNotIn("test-key", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
