from __future__ import annotations

from dataclasses import dataclass

from src.rag.llm.gemini_adapter import LLMGenerationError
from src.rag.prompts.prompt_builder import PromptBuilder
from src.rag.retrieval.evidence import EvidenceFormatter
from src.rag.safety.grounding_check import (
    build_fallback_answer,
    run_safety_check,
)


@dataclass
class RagAnswer:
    """
    The final result of one baseline RAG answer: either the LLM's
    narration (safety-checked) or the deterministic fallback.
    """

    text: str
    sources_line: str
    used_fallback: bool
    fallback_reason: str | None


class RagAnswerPipeline:
    """
    The complete baseline RAG pipeline (primary spec section 2):

        Question -> Retriever -> Evidence -> Prompt -> LLM ->
        Safety/Grounding Check -> Answer + Sources

    A user must never see nothing or a bare error when evidence was
    successfully retrieved (spec section 5, point 4) -- both an LLM
    failure and a failed safety check fall back to the same
    evidence-only response, never an exception or an empty answer.
    """

    def __init__(
        self,
        evidence_formatter: EvidenceFormatter,
        prompt_builder: PromptBuilder,
        llm_adapter,
    ):
        self.evidence_formatter = evidence_formatter
        self.prompt_builder = prompt_builder
        self.llm_adapter = llm_adapter

    def answer(self, query: str) -> RagAnswer:
        evidence = self.evidence_formatter.get_evidence(query)
        prompt = self.prompt_builder.build(query, evidence)

        try:
            llm_result = self.llm_adapter.generate(
                prompt.system_prompt,
                prompt.user_prompt,
            )
        except LLMGenerationError as error:
            return RagAnswer(
                text=build_fallback_answer(evidence),
                sources_line=evidence.format_sources_line(),
                used_fallback=True,
                fallback_reason=f"LLM generation failed: {error}",
            )

        safety_result = run_safety_check(llm_result.text, evidence)

        if not safety_result.passed:
            return RagAnswer(
                text=build_fallback_answer(evidence),
                sources_line=evidence.format_sources_line(),
                used_fallback=True,
                fallback_reason=safety_result.reason,
            )

        return RagAnswer(
            text=llm_result.text,
            sources_line=evidence.format_sources_line(),
            used_fallback=False,
            fallback_reason=None,
        )
