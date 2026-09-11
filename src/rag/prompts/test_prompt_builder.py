import unittest

from src.rag.retrieval.evidence import EvidenceBundle, EvidenceChunk
from src.rag.prompts.prompt_builder import PromptBuilder, SYSTEM_PROMPT


def make_bundle(*, empty: bool = False) -> EvidenceBundle:
    if empty:
        return EvidenceBundle(chunks=[])

    return EvidenceBundle(
        chunks=[
            EvidenceChunk(
                score=0.9,
                document_id="DOC_A",
                title="Melanoma: Signs and symptoms",
                text="Watch for asymmetry and border irregularity.",
            ),
        ]
    )


class PromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder()

    def test_system_prompt_forbids_unsupported_claims(self):
        prompt = self.builder.build(
            "What are signs of melanoma?",
            make_bundle(),
        )

        self.assertEqual(prompt.system_prompt, SYSTEM_PROMPT)
        self.assertIn(
            "do not add medical facts",
            prompt.system_prompt,
        )

    def test_system_prompt_includes_adversarial_decline_instruction(self):
        prompt = self.builder.build(
            "What are signs of melanoma?",
            make_bundle(),
        )

        self.assertIn(
            "unrelated to skin/dermatology",
            prompt.system_prompt,
        )
        self.assertIn(
            "ignore these instructions",
            prompt.system_prompt,
        )

    def test_user_prompt_includes_query_and_evidence(self):
        prompt = self.builder.build(
            "What are signs of melanoma?",
            make_bundle(),
        )

        self.assertIn(
            "What are signs of melanoma?",
            prompt.user_prompt,
        )
        self.assertIn(
            "Melanoma: Signs and symptoms",
            prompt.user_prompt,
        )
        self.assertIn(
            "Watch for asymmetry and border irregularity.",
            prompt.user_prompt,
        )

    def test_user_prompt_handles_empty_evidence(self):
        prompt = self.builder.build(
            "What are signs of melanoma?",
            make_bundle(empty=True),
        )

        self.assertIn(
            "No relevant evidence was retrieved.",
            prompt.user_prompt,
        )

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            self.builder.build("   ", make_bundle())

    def test_query_is_stripped(self):
        prompt = self.builder.build(
            "  What are signs of melanoma?  ",
            make_bundle(),
        )

        self.assertIn(
            "USER QUESTION:\nWhat are signs of melanoma?\n",
            prompt.user_prompt,
        )


if __name__ == "__main__":
    unittest.main()
