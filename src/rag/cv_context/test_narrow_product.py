"""
Plan A (narrow product) mode: the RAG layer must not narrate a diagnosis.

`docs/plan_a_narrow_product_spec.md` §5 forbids `native_class`,
`risk_category` and the diagnosis-attached confidence from reaching a
user surface, because a user told "most likely class: NEV" has been
reassured whether or not a risk category accompanied it.

These tests pin the narrow rendering AND the invariant that makes
grounding meaningful: prompt, safety check and fallback must all score
against IDENTICAL CV text (`render_cv_context`'s own docstring). If
narrow mode applied to only some of them, an answer could be rejected
as ungrounded for citing evidence the prompt never supplied.

The default (include_diagnosis=True) is deliberately unchanged, so the
Phase 1/2 behaviour and the existing tests of it are untouched.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.rag.cv_context.parser import parse_cv_assessment
from src.rag.cv_context.schema import render_cv_context
from src.rag.retrieval.evidence import EvidenceBundle
from src.rag.safety.grounding_check import build_fallback_answer

FIXTURES = Path(__file__).resolve().parents[3] / "docs/cv8_sample_outputs/sample_outputs.json"

FORBIDDEN_LABELS = ("Most likely class", "Risk category", "Assessment confidence")


def _contexts():
    payloads = json.loads(FIXTURES.read_text())
    return [parse_cv_assessment(entry["payload"]) for entry in payloads]


class NarrowProductRendering(unittest.TestCase):
    def setUp(self):
        self.contexts = _contexts()

    def test_narrow_mode_omits_diagnosis_class_and_risk(self):
        for context in self.contexts:
            text = context.format_for_prompt(include_diagnosis=False)
            for label in FORBIDDEN_LABELS:
                self.assertNotIn(label, text, f"{label} leaked for {context.lesion_id}")
            self.assertNotIn(context.native_class, text)
            self.assertNotIn(context.risk_category, text)

    def test_narrow_mode_keeps_what_the_narrow_product_ships(self):
        """Change over time and capture quality ARE the product."""
        for context in self.contexts:
            text = context.format_for_prompt(include_diagnosis=False)
            self.assertIn(context.lesion_id, text)
            self.assertIn("Change since the previous photo", text)
            self.assertIn("Image quality notes", text)
            self.assertIn("Flagged for professional review", text)

    def test_default_is_unchanged(self):
        """Phase 1/2 behaviour must not move; Plan A is opt-in."""
        for context in self.contexts:
            default = context.format_for_prompt()
            explicit = context.format_for_prompt(include_diagnosis=True)
            self.assertEqual(default, explicit)
            self.assertIn("Risk category", default)

    def test_render_cv_context_propagates_the_flag(self):
        rendered = render_cv_context(self.contexts, include_diagnosis=False)
        for label in FORBIDDEN_LABELS:
            self.assertNotIn(label, rendered)
        self.assertIn("Change since the previous photo", rendered)

    def test_fallback_does_not_leak_a_diagnosis_in_narrow_mode(self):
        """The fallback is a USER surface -- spec 5.4 requires CV evidence
        to reach the user when narration fails, and Plan A requires it to
        carry no diagnosis. Both hold: the change verdict still lands."""
        context = self.contexts[0]
        text = build_fallback_answer(
            EvidenceBundle(chunks=[]), cv_context=context, include_diagnosis=False
        )
        for label in FORBIDDEN_LABELS:
            self.assertNotIn(label, text)
        self.assertIn("Change since the previous photo", text)

    def test_grounding_and_prompt_see_identical_text(self):
        """The invariant render_cv_context's docstring warns about: if the
        safety check scored against different text than the prompt
        supplied, 'grounded' would stop meaning anything."""
        from src.rag.prompts.prompt_builder import PromptBuilder

        evidence = EvidenceBundle(chunks=[])
        prompt = PromptBuilder().build(
            "what changed?", evidence, cv_context=self.contexts[0],
            include_diagnosis=False,
        )
        scored = render_cv_context(self.contexts[0], include_diagnosis=False)
        self.assertIn(scored, prompt.user_prompt)


if __name__ == "__main__":
    unittest.main()
