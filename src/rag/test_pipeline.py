import unittest

from src.rag.llm.gemini_adapter import LLMGenerationError, LLMResponse
from src.rag.pipeline import RagAnswerPipeline
from src.rag.prompts.prompt_builder import PromptBuilder
from src.rag.retrieval.evidence import EvidenceBundle, EvidenceChunk

GROUNDED_EVIDENCE = EvidenceBundle(
    chunks=[
        EvidenceChunk(
            score=0.9,
            document_id="AAD_ACTINIC_KERATOSIS_SYMPTOMS_001",
            title="Actinic keratosis: Signs and symptoms",
            text=(
                "An actinic keratosis develops when skin has been "
                "badly damaged by ultraviolet light. It commonly "
                "feels rough, like sandpaper, and may look like a "
                "brown age spot."
            ),
        )
    ]
)


class FakeEvidenceFormatter:
    def __init__(self, bundle: EvidenceBundle):
        self.bundle = bundle
        self.last_query = None

    def get_evidence(self, query: str) -> EvidenceBundle:
        self.last_query = query
        return self.bundle


class FakeLLMAdapter:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))

        if self.error is not None:
            raise self.error

        return self.response


class RagAnswerPipelineTests(unittest.TestCase):
    def _pipeline(self, llm_adapter, bundle=GROUNDED_EVIDENCE):
        return RagAnswerPipeline(
            evidence_formatter=FakeEvidenceFormatter(bundle),
            prompt_builder=PromptBuilder(),
            llm_adapter=llm_adapter,
        )

    def test_safe_grounded_answer_passes_through(self):
        llm_adapter = FakeLLMAdapter(
            response=LLMResponse(
                text=(
                    "Actinic keratosis commonly feels rough, like "
                    "sandpaper, and may resemble a brown age spot."
                ),
                model="fake-model",
                finish_reason="STOP",
            )
        )

        pipeline = self._pipeline(llm_adapter)
        result = pipeline.answer(
            "What are common signs of actinic keratosis?"
        )

        self.assertFalse(result.used_fallback)
        self.assertIsNone(result.fallback_reason)
        self.assertIn("sandpaper", result.text)
        self.assertIn(
            "Actinic keratosis: Signs and symptoms",
            result.sources_line,
        )

    def test_llm_error_falls_back_to_evidence(self):
        llm_adapter = FakeLLMAdapter(
            error=LLMGenerationError("simulated timeout")
        )

        pipeline = self._pipeline(llm_adapter)
        result = pipeline.answer(
            "What are common signs of actinic keratosis?"
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("LLM generation failed", result.fallback_reason)
        self.assertIn(
            "full explanation isn't available right now",
            result.text,
        )
        self.assertIn(
            "Actinic keratosis: Signs and symptoms",
            result.text,
        )

    def test_unsafe_answer_falls_back_to_evidence(self):
        llm_adapter = FakeLLMAdapter(
            response=LLMResponse(
                text="You have actinic keratosis, confirmed.",
                model="fake-model",
                finish_reason="STOP",
            )
        )

        pipeline = self._pipeline(llm_adapter)
        result = pipeline.answer(
            "What are common signs of actinic keratosis?"
        )

        self.assertTrue(result.used_fallback)
        self.assertIsNotNone(result.fallback_reason)
        self.assertIn(
            "full explanation isn't available right now",
            result.text,
        )

    def test_ungrounded_answer_falls_back_to_evidence(self):
        llm_adapter = FakeLLMAdapter(
            response=LLMResponse(
                text=(
                    "Burn first aid depends on severity: cool the "
                    "area and cover it with a clean dressing."
                ),
                model="fake-model",
                finish_reason="STOP",
            )
        )

        pipeline = self._pipeline(llm_adapter)
        result = pipeline.answer(
            "What are common signs of actinic keratosis?"
        )

        self.assertTrue(result.used_fallback)

    def test_query_reaches_evidence_formatter(self):
        llm_adapter = FakeLLMAdapter(
            response=LLMResponse(
                text="rough sandpaper brown spot",
                model="fake-model",
                finish_reason="STOP",
            )
        )

        formatter = FakeEvidenceFormatter(GROUNDED_EVIDENCE)
        pipeline = RagAnswerPipeline(
            evidence_formatter=formatter,
            prompt_builder=PromptBuilder(),
            llm_adapter=llm_adapter,
        )

        pipeline.answer("What are common signs of actinic keratosis?")

        self.assertEqual(
            formatter.last_query,
            "What are common signs of actinic keratosis?",
        )


if __name__ == "__main__":
    unittest.main()
