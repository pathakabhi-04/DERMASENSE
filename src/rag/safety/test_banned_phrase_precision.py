"""
Precision/recall tests for the narrowed banned-phrase check.

This is a SAFETY rule, so the recall set is the important half: every
case in `TRUE_POSITIVES` must be caught, and the 8 adversarial ones
exist specifically to stop anyone re-widening the hedge from clause
scope to sentence scope. A sentence-scoped version missed all 8.

The false-positive set is real data, not invented: sentences harvested
from the indexed medical corpus that the original co-occurrence rule
flagged (39 of 156 chunks, 25%), plus CV narrations observed in live
Groq answers.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.rag.safety.grounding_check import (
    CERTAINTY_PHRASES,
    CONDITION_NAMES,
    _SENTENCE_SPLIT_RE,
    check_banned_phrases,
)

CHUNKS_PATH = Path("data/rag/indexes/medical_v0.1/chunks.json")

# Must ALWAYS be caught: the system asserting the patient has a condition.
TRUE_POSITIVES = [
    "You have melanoma.",
    "You have a basal cell carcinoma.",
    "This is melanoma.",
    "This is a squamous cell carcinoma and needs urgent treatment.",
    "Based on the photo, you have skin cancer.",
    "You have been diagnosed with melanoma.",
    "This is definitely a seborrheic keratosis.",
    "It is confirmed melanoma.",
    "I can confirm this is basal cell carcinoma.",
    "Your lesion is confirmed skin cancer.",
    "This is actinic keratosis and you have skin cancer. It is confirmed.",
    "You definitely have melanoma.",
    "You have nevus that has turned into melanoma.",
    "This is a mole that is definitely cancerous melanoma.",
    "The assessment shows you have squamous cell carcinoma.",
]

# Real claims that happen to contain a hedge EARLIER in the sentence.
# These exist to keep the hedge scoped to its clause. Do not delete.
ADVERSARIAL_TRUE_POSITIVES = [
    "If you were wondering, you have melanoma.",
    "You asked whether it is serious: this is melanoma.",
    "In case you are unsure, this is basal cell carcinoma.",
    "Your risk of dying is high because you have melanoma.",
    "Although a biopsy may help, you have skin cancer.",
    "It is possible to treat this, but you have melanoma.",
    "You are likely to recover, but this is squamous cell carcinoma.",
    "If treatment is needed, you have basal cell carcinoma and must act.",
]

# Must NOT be caught: not diagnostic claims. Verbatim source text.
CORPUS_FALSE_POSITIVES = [
    "A dermatologist can tell you if you have basal cell carcinoma and if you "
    "do, what treatment is recommended.",
    "For example, in 2017-2018, the lifetime risk of being diagnosed with "
    "melanoma was 2.9% (1 in 34) for White people.",
    "Should that change be an AK, you have a greater risk of developing skin "
    "cancer.",
    "If you have a raised mole on skin that you shave, you may nick the mole, "
    "causing it to bleed.",
]

# Observed in real CV-grounded Groq answers.
CV_NARRATION_FALSE_POSITIVES = [
    "- **Lesion identifier:** HighRisk37_Lesion6 - this is simply the label "
    "the system uses to keep track of the mole.",
    "Bring the current photo (and, if you have it, the earlier photo that "
    "showed the mole before the colour change).",
    "This is the risk category the system assigned to your mole.",
    "This is simply the identifier for your mole, not a diagnosis.",
]


class RecallTests(unittest.TestCase):
    """Every true positive must be caught. No exceptions."""

    def test_plain_diagnostic_claims_are_caught(self):
        for text in TRUE_POSITIVES:
            with self.subTest(text=text):
                self.assertTrue(check_banned_phrases(text))

    def test_hedged_sentence_does_not_hide_a_real_claim(self):
        """
        The regression that matters most. A hedge governs its own
        clause; scoping it to the sentence let all 8 of these through.
        """
        for text in ADVERSARIAL_TRUE_POSITIVES:
            with self.subTest(text=text):
                self.assertTrue(check_banned_phrases(text))


class PrecisionTests(unittest.TestCase):
    """Non-claims must not be caught."""

    def test_conditional_corpus_sentences_are_not_flagged(self):
        for text in CORPUS_FALSE_POSITIVES:
            with self.subTest(text=text):
                self.assertFalse(check_banned_phrases(text))

    def test_cv_narration_is_not_flagged(self):
        for text in CV_NARRATION_FALSE_POSITIVES:
            with self.subTest(text=text):
                self.assertFalse(check_banned_phrases(text))


class CorpusFalsePositiveRateTest(unittest.TestCase):
    """
    Measured against the real index. The original rule flagged 39/156
    chunks (25%); the narrowed rule must stay far below that, or a
    faithful paraphrase of the evidence gets rejected for being
    faithful.
    """

    def test_corpus_false_positive_rate_is_low(self):
        if not CHUNKS_PATH.exists():
            self.skipTest("index not built")

        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        flagged = [c for c in chunks if check_banned_phrases(c["text"])]
        rate = len(flagged) / len(chunks)

        self.assertLess(
            rate, 0.05,
            f"{len(flagged)}/{len(chunks)} ({rate:.0%}) corpus chunks flagged; "
            "the original co-occurrence rule flagged 25%.",
        )

    def test_original_rule_would_have_flagged_far_more(self):
        """Documents the baseline the narrowing is measured against."""
        if not CHUNKS_PATH.exists():
            self.skipTest("index not built")

        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

        def original(text: str) -> bool:
            for sentence in _SENTENCE_SPLIT_RE.split(text):
                low = sentence.lower()
                if any(p in low for p in CERTAINTY_PHRASES) and any(
                    c in low for c in CONDITION_NAMES
                ):
                    return True
            return False

        before = sum(original(c["text"]) for c in chunks)
        after = sum(check_banned_phrases(c["text"]) for c in chunks)
        self.assertGreater(before, after)
        self.assertGreaterEqual(before / len(chunks), 0.20)


if __name__ == "__main__":
    unittest.main()
