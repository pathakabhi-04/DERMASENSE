"""
Spec section 6's third pass criterion: state uncertainty explicitly
when retrieval returned low-similarity evidence.

`EvidenceBundle.top_score` existed and was unit-tested, but nothing
consumed it, so the criterion had no implementation and could only be
satisfied by the model happening to hedge. These tests pin the wiring.

The threshold is calibrated against the real score distribution, per
that section's own instruction not to invent a number -- see
LOW_SIMILARITY_THRESHOLD in evidence.py for the measurements.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.rag.cv_context.parser import parse_cv_assessment
from src.rag.prompts.prompt_builder import PromptBuilder
from src.rag.retrieval.evidence import (
    LOW_SIMILARITY_THRESHOLD,
    EvidenceBundle,
    EvidenceChunk,
)

NOTE = "NOTE ON EVIDENCE STRENGTH"
FIXTURES = Path("docs/cv8_sample_outputs/sample_outputs.json")

# Real top_scores measured against the live index.
WEAKEST_IN_SCOPE = 0.3415      # "How should I clean an abrasion?" -- the
                               # retrieval eval's own known Top-1 miss
NEXT_IN_SCOPE = 0.4847         # "How should a cut be cleaned?"
STRONG_IN_SCOPE = 0.8001       # "How should I provide first aid for a burn?"
OUT_OF_SCOPE = 0.1220          # "What's the best recipe for cookies?"


def _bundle(score: float) -> EvidenceBundle:
    return EvidenceBundle(
        chunks=[EvidenceChunk(score, "DOC", "Title", "evidence text")]
    )


class ThresholdTests(unittest.TestCase):
    def test_threshold_sits_inside_the_measured_in_scope_gap(self):
        """
        The in-scope set jumps from 0.3415 to 0.4847 with nothing
        between. The threshold must stay in that gap: above it, and a
        legitimate query starts hedging; below it, the eval's own known
        near-miss stops hedging.
        """
        self.assertGreater(LOW_SIMILARITY_THRESHOLD, WEAKEST_IN_SCOPE)
        self.assertLess(LOW_SIMILARITY_THRESHOLD, NEXT_IN_SCOPE)

    def test_weak_retrieval_is_flagged(self):
        self.assertTrue(_bundle(WEAKEST_IN_SCOPE).is_low_similarity)
        self.assertTrue(_bundle(OUT_OF_SCOPE).is_low_similarity)

    def test_ordinary_retrieval_is_not_flagged(self):
        self.assertFalse(_bundle(NEXT_IN_SCOPE).is_low_similarity)
        self.assertFalse(_bundle(STRONG_IN_SCOPE).is_low_similarity)

    def test_empty_retrieval_counts_as_low_similarity(self):
        """Nothing retrieved is the strongest reason to hedge, not the weakest."""
        self.assertTrue(EvidenceBundle(chunks=[]).is_low_similarity)


class PromptWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = PromptBuilder()

    def test_note_appears_when_evidence_is_weak(self):
        prompt = self.builder.build("q", _bundle(WEAKEST_IN_SCOPE))
        self.assertIn(NOTE, prompt.user_prompt)

    def test_note_absent_when_evidence_is_strong(self):
        prompt = self.builder.build("q", _bundle(STRONG_IN_SCOPE))
        self.assertNotIn(NOTE, prompt.user_prompt)

    def test_note_appears_on_the_cv_branch_too(self):
        """Both prompt branches must honour it, or Phase 2 silently loses it."""
        context = parse_cv_assessment(json.loads(FIXTURES.read_text())[0]["payload"])
        weak = self.builder.build("q", _bundle(WEAKEST_IN_SCOPE), cv_context=context)
        strong = self.builder.build("q", _bundle(STRONG_IN_SCOPE), cv_context=context)
        self.assertIn(NOTE, weak.user_prompt)
        self.assertNotIn(NOTE, strong.user_prompt)

    def test_empty_evidence_gets_the_note(self):
        self.assertIn(NOTE, self.builder.build("q", EvidenceBundle(chunks=[])).user_prompt)

    def test_both_branches_share_the_citation_and_insufficiency_rules(self):
        context = parse_cv_assessment(json.loads(FIXTURES.read_text())[0]["payload"])
        for kwargs in ({}, {"cv_context": context}):
            with self.subTest(cv=bool(kwargs)):
                text = self.builder.build("q", _bundle(STRONG_IN_SCOPE), **kwargs).user_prompt
                self.assertIn("[1], [2], [3]", text)
                self.assertIn("say so explicitly", text)


if __name__ == "__main__":
    unittest.main()
