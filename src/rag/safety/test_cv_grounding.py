"""
Regression tests for the two CV-integration blockers fixed in the
safety layer. Both were anticipated by the primary spec (section 5)
but not implemented for CV context, and both are invisible until a CV
assessment actually reaches the pipeline.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.rag.cv_context.parser import parse_cv_assessment
from src.rag.retrieval.evidence import EvidenceBundle, EvidenceChunk
from src.rag.safety.grounding_check import (
    CV_SOURCE_PRESENCE_THRESHOLD,
    build_fallback_answer,
    check_source_presence,
    run_safety_check,
)

FIXTURES = Path("docs/cv8_sample_outputs/sample_outputs.json")

# A realistic CV-grounded narration: it explains the patient's own
# assessment and deliberately does NOT recite corpus vocabulary.
CV_NARRATION = (
    "Your lesion was assessed as medium risk with 84% confidence. Comparing your "
    "current photo against your previous visit, no meaningful change was detected. "
    "Note that size could not be measured, which is not the same as being "
    "unchanged. A clinician should still review this."
)

OFF_TOPIC = (
    "Here is a recipe for chocolate chip cookies. Cream the butter and sugar, "
    "then fold in the flour and chocolate chips."
)


def _context(index: int = 1):
    return parse_cv_assessment(json.loads(FIXTURES.read_text())[index]["payload"])


def _corpus_evidence() -> EvidenceBundle:
    """Real-shaped corpus evidence with no CV vocabulary in it."""
    return EvidenceBundle(chunks=[
        EvidenceChunk(
            score=0.74,
            document_id="AAD_ACTINIC_KERATOSIS_SYMPTOMS_001",
            title="Actinic keratosis: Signs and symptoms",
            text=(
                "Actinic keratoses usually develop on skin that has had years of "
                "sun exposure. They often appear as rough, dry, scaly patches that "
                "may feel like sandpaper, and the surrounding skin frequently shows "
                "other signs of sun damage such as mottled pigmentation."
            ),
        )
    ])


class BlockerAGroundingTests(unittest.TestCase):
    """
    A CV-grounded answer is grounded. Before the fix it was rejected,
    because grounding was measured only against retrieved corpus text
    and a CV narration is not corpus content.
    """

    def test_cv_narration_is_rejected_without_cv_context(self):
        # Documents the old behaviour, so a regression is loud.
        self.assertFalse(check_source_presence(CV_NARRATION, _corpus_evidence()))

    def test_cv_narration_is_accepted_with_cv_context(self):
        self.assertTrue(
            check_source_presence(
                CV_NARRATION, _corpus_evidence(), cv_context=_context()
            )
        )

    def test_run_safety_check_passes_a_cv_grounded_answer(self):
        result = run_safety_check(
            CV_NARRATION, _corpus_evidence(), cv_context=_context()
        )
        self.assertTrue(result.passed)
        self.assertFalse(result.source_presence_violation)

    def test_off_topic_is_still_rejected_with_cv_context(self):
        """The fix must not weaken the check into a rubber stamp."""
        self.assertFalse(
            check_source_presence(OFF_TOPIC, _corpus_evidence(), cv_context=_context())
        )
        self.assertFalse(
            run_safety_check(OFF_TOPIC, _corpus_evidence(), cv_context=_context()).passed
        )

    def test_corpus_grounded_answer_still_passes(self):
        corpus_answer = (
            "Actinic keratoses develop on skin with years of sun exposure and often "
            "appear as rough, dry, scaly patches that feel like sandpaper."
        )
        self.assertTrue(check_source_presence(corpus_answer, _corpus_evidence()))
        self.assertTrue(
            check_source_presence(
                corpus_answer, _corpus_evidence(), cv_context=_context()
            )
        )

    def test_banned_phrase_still_wins_over_cv_grounding(self):
        """CV grounding must not launder a direct-diagnosis claim."""
        result = run_safety_check(
            "This is melanoma and you have skin cancer. It is confirmed.",
            _corpus_evidence(),
            cv_context=_context(),
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.banned_phrase_violation)

    def test_empty_answer_is_never_grounded(self):
        self.assertFalse(check_source_presence("   ", _corpus_evidence(),
                                               cv_context=_context()))

    def test_cv_context_alone_can_ground_when_retrieval_returned_nothing(self):
        self.assertTrue(
            check_source_presence(
                CV_NARRATION, EvidenceBundle(chunks=[]), cv_context=_context()
            )
        )

    def test_nothing_supplied_is_never_grounded(self):
        self.assertFalse(
            check_source_presence(CV_NARRATION, EvidenceBundle(chunks=[]))
        )


class BlockerBFallbackTests(unittest.TestCase):
    """
    Spec section 5.4: structured CV evidence is safety-relevant and
    must reach the user even when narration fails. Before the fix the
    fallback rendered corpus text only, silently dropping the risk
    assessment on exactly the path CV answers hit most.
    """

    def _high_risk_context(self):
        payload = json.loads(FIXTURES.read_text())[0]["payload"]
        return parse_cv_assessment(payload)

    def test_fallback_without_cv_context_omits_the_assessment(self):
        text = build_fallback_answer(_corpus_evidence())
        self.assertNotIn("HIGH", text)

    def test_fallback_surfaces_risk_category(self):
        text = build_fallback_answer(
            _corpus_evidence(), cv_context=self._high_risk_context()
        )
        self.assertIn("HIGH", text)

    def test_fallback_surfaces_review_flag_and_confidence(self):
        context = _context(2)  # requires_review = True
        text = build_fallback_answer(_corpus_evidence(), cv_context=context)
        self.assertIn("Flagged for professional review: yes", text)
        self.assertIn(f"{context.confidence:.0%}", text)

    def test_cv_assessment_precedes_general_evidence(self):
        text = build_fallback_answer(
            _corpus_evidence(), cv_context=self._high_risk_context()
        )
        self.assertLess(text.index("CV ASSESSMENT"), text.index("Actinic keratoses"))

    def test_fallback_works_with_cv_context_and_no_evidence(self):
        text = build_fallback_answer(
            EvidenceBundle(chunks=[]), cv_context=self._high_risk_context()
        )
        self.assertIn("HIGH", text)
        self.assertNotIn("no relevant medical evidence was found", text)

    def test_fallback_with_neither_still_tells_the_user_something(self):
        text = build_fallback_answer(EvidenceBundle(chunks=[]))
        self.assertIn("consult a healthcare professional", text)

    def test_multi_lesion_fallback_leads_with_highest_risk(self):
        high = self._high_risk_context()
        low_payload = copy.deepcopy(json.loads(FIXTURES.read_text())[1]["payload"])
        low_payload["risk_category"] = "LOW"
        low_payload["lesion_id"] = "low-risk-lesion"
        low = parse_cv_assessment(low_payload)

        text = build_fallback_answer(_corpus_evidence(), cv_context=[low, high])
        self.assertLess(text.index(high.lesion_id), text.index(low.lesion_id))
        self.assertIn(low.lesion_id, text)


if __name__ == "__main__":
    unittest.main()


class ContainmentMetricTests(unittest.TestCase):
    """
    The CV assessment is scored by containment, not Jaccard.

    Jaccard divides by the union, so it penalised a thorough answer for
    its own length: real CV answers scored 0.1049-0.1776 against a 0.12
    threshold, the worst of them BELOW the bar despite being correct.
    Grounding depended on verbosity rather than on whether the answer
    used what it was given.
    """

    def test_a_thorough_cv_answer_is_not_penalised_for_length(self):
        context = _context()
        short = CV_NARRATION
        padded = CV_NARRATION + (
            " Your dermatologist can discuss the available options with you, "
            "explain what to expect at the appointment, and answer any "
            "questions you may have about next steps and follow-up care."
        )
        # Both are grounded; under Jaccard the longer one scored worse.
        for answer in (short, padded):
            with self.subTest(length=len(answer)):
                self.assertTrue(
                    check_source_presence(
                        answer, EvidenceBundle(chunks=[]), cv_context=context
                    )
                )

    def test_corpus_text_does_not_ground_against_cv_context(self):
        """
        Negative control across the real index: corpus prose is not
        grounded in a CV assessment. Worst of 780 pairs was 0.1622,
        below the 0.25 threshold.
        """
        chunks_path = Path("data/rag/indexes/medical_v0.1/chunks.json")
        if not chunks_path.exists():
            self.skipTest("index not built")

        context = _context()
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        grounded = [
            c for c in chunks
            if check_source_presence(
                c["text"], EvidenceBundle(chunks=[]), cv_context=context
            )
        ]
        self.assertEqual(grounded, [], f"{len(grounded)} corpus chunks grounded")

    def test_cv_threshold_is_overridable(self):
        context = _context()
        self.assertFalse(
            check_source_presence(
                CV_NARRATION, EvidenceBundle(chunks=[]),
                cv_context=context, cv_threshold=0.99,
            )
        )
