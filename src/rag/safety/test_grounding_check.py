import unittest

from src.rag.retrieval.evidence import EvidenceBundle, EvidenceChunk
from src.rag.safety.grounding_check import (
    build_fallback_answer,
    check_banned_phrases,
    check_source_presence,
    run_safety_check,
)

REAL_GROUNDED_ANSWER = (
    "Based on the provided evidence, common signs of an actinic "
    "keratosis (AK) include a rough-feeling patch of skin. It "
    "commonly occurs on skin that has received a lot of sun "
    "exposure. You can often feel an actinic keratosis before you "
    "can see it, and it can feel like sandpaper. These growths can "
    "appear as brown spots on the skin that may look like age "
    "spots. These precancerous growths can appear on the skin or "
    "lips. Because actinic keratoses can potentially turn into a "
    "type of skin cancer called squamous cell carcinoma if left "
    "untreated, it is recommended to have any rough patches, "
    "unusual spots, or concerning skin changes evaluated by a "
    "healthcare professional or dermatologist."
)

REAL_EVIDENCE_TEXT = (
    "What are the signs and symptoms of actinic keratosis? An "
    "actinic keratosis (AK) develops when skin has been badly "
    "damaged by ultraviolet (UV) light from the sun or indoor "
    "tanning. Signs of actinic keratosis: the brown spots on this "
    "man's face may look like age spots, but they're actually "
    "actinic keratoses. Left untreated, some actinic keratoses turn "
    "into a type of skin cancer called squamous cell carcinoma "
    "(SCC)."
)

UNRELATED_EVIDENCE_TEXT = (
    "Burn first aid depends on the severity and extent of the burn. "
    "Second-degree burns blister and swell."
)


def make_bundle(text: str, title: str = "Test Source") -> EvidenceBundle:
    return EvidenceBundle(
        chunks=[
            EvidenceChunk(
                score=0.9,
                document_id="DOC_TEST",
                title=title,
                text=text,
            )
        ]
    )


class CheckBannedPhrasesTests(unittest.TestCase):
    def test_real_grounded_answer_has_no_violation(self):
        self.assertFalse(
            check_banned_phrases(REAL_GROUNDED_ANSWER)
        )

    def test_you_have_plus_condition_is_a_violation(self):
        self.assertTrue(
            check_banned_phrases(
                "Based on your description, you have melanoma."
            )
        )

    def test_this_is_plus_condition_is_a_violation(self):
        self.assertTrue(
            check_banned_phrases("This is basal cell carcinoma.")
        )

    def test_confirmed_plus_condition_is_a_violation(self):
        self.assertTrue(
            check_banned_phrases(
                "It is confirmed that this mole is a nevus."
            )
        )

    def test_diagnosed_with_plus_condition_is_a_violation(self):
        self.assertTrue(
            check_banned_phrases(
                "You are diagnosed with squamous cell carcinoma."
            )
        )

    def test_definitely_plus_condition_is_a_violation(self):
        self.assertTrue(
            check_banned_phrases(
                "This lesion is definitely melanoma."
            )
        )

    def test_certainty_phrase_without_condition_is_not_a_violation(self):
        self.assertFalse(
            check_banned_phrases(
                "This is a general overview of skin health."
            )
        )

    def test_condition_without_certainty_phrase_is_not_a_violation(self):
        self.assertFalse(
            check_banned_phrases(
                "Melanoma can present with asymmetry and color "
                "variation."
            )
        )

    def test_violation_split_across_different_sentences_does_not_trigger(
        self,
    ):
        self.assertFalse(
            check_banned_phrases(
                "You have described a change in size and color. "
                "Melanoma is one possible cause of such changes "
                "among others."
            )
        )


class CheckSourcePresenceTests(unittest.TestCase):
    def test_grounded_answer_passes_against_real_evidence(self):
        bundle = make_bundle(REAL_EVIDENCE_TEXT)

        self.assertTrue(
            check_source_presence(REAL_GROUNDED_ANSWER, bundle)
        )

    def test_grounded_answer_fails_against_unrelated_evidence(self):
        bundle = make_bundle(UNRELATED_EVIDENCE_TEXT)

        self.assertFalse(
            check_source_presence(REAL_GROUNDED_ANSWER, bundle)
        )

    def test_empty_evidence_is_never_grounded(self):
        empty_bundle = EvidenceBundle(chunks=[])

        self.assertFalse(
            check_source_presence(
                REAL_GROUNDED_ANSWER,
                empty_bundle,
            )
        )

    def test_empty_answer_is_never_grounded(self):
        bundle = make_bundle(REAL_EVIDENCE_TEXT)

        self.assertFalse(
            check_source_presence("", bundle)
        )


class RunSafetyCheckTests(unittest.TestCase):
    def test_grounded_safe_answer_passes(self):
        bundle = make_bundle(REAL_EVIDENCE_TEXT)

        result = run_safety_check(REAL_GROUNDED_ANSWER, bundle)

        self.assertTrue(result.passed)
        self.assertFalse(result.banned_phrase_violation)
        self.assertFalse(result.source_presence_violation)
        self.assertIsNone(result.reason)

    def test_banned_phrase_fails_even_if_grounded(self):
        bundle = make_bundle(REAL_EVIDENCE_TEXT)
        answer = (
            "You have actinic keratosis, confirmed by the rough, "
            "sandpaper-like brown spots described in the evidence."
        )

        result = run_safety_check(answer, bundle)

        self.assertFalse(result.passed)
        self.assertTrue(result.banned_phrase_violation)
        self.assertIsNotNone(result.reason)

    def test_ungrounded_answer_fails_even_without_banned_phrases(self):
        bundle = make_bundle(UNRELATED_EVIDENCE_TEXT)

        result = run_safety_check(REAL_GROUNDED_ANSWER, bundle)

        self.assertFalse(result.passed)
        self.assertFalse(result.banned_phrase_violation)
        self.assertTrue(result.source_presence_violation)


class BuildFallbackAnswerTests(unittest.TestCase):
    def test_fallback_includes_evidence_and_sources(self):
        bundle = make_bundle(
            REAL_EVIDENCE_TEXT,
            title="Actinic keratosis: Signs and symptoms",
        )

        fallback = build_fallback_answer(bundle)

        self.assertIn(
            "full explanation isn't available right now",
            fallback,
        )
        self.assertIn(
            "Actinic keratosis: Signs and symptoms",
            fallback,
        )

    def test_fallback_handles_empty_evidence(self):
        fallback = build_fallback_answer(EvidenceBundle(chunks=[]))

        self.assertIn(
            "consult a healthcare professional",
            fallback,
        )


if __name__ == "__main__":
    unittest.main()
