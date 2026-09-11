"""
CV-8 parser tests, including the primary spec's section 26 fixture
tests 1-7. Tests 1-5 run against the REAL delivered fixtures rather
than hand-written dicts, so a CV-side contract change breaks them.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.rag.cv_context.parser import (
    CVContractError,
    order_by_severity,
    parse_cv_assessment,
)
from src.rag.cv_context.schema import render_cv_context

FIXTURES = Path("docs/cv8_sample_outputs/sample_outputs.json")


def _payloads() -> list[dict]:
    # Each entry wraps the contract object as {description, source, payload}.
    return [e["payload"] for e in json.loads(FIXTURES.read_text())]


class RealFixtureTests(unittest.TestCase):
    """Spec section 26, tests 1-5 -- the five real CV-8 examples."""

    def setUp(self) -> None:
        self.payloads = _payloads()
        self.contexts = [parse_cv_assessment(p) for p in self.payloads]

    def test_all_five_fixtures_parse(self):
        self.assertEqual(len(self.contexts), 5)
        for c in self.contexts:
            self.assertEqual(c.contract_version, "1.1")

    def test_1_first_visit_does_not_claim_a_comparison(self):
        c = self.contexts[0]
        self.assertEqual(c.temporal.verdict, "NO_PRIOR_DATA")
        self.assertIn("NO_TEMPORAL_COMPARISON", c.quality_flags)
        text = c.format_for_prompt()
        self.assertIn("no previous photo was available", text)
        # Must not imply stability from an absent comparison.
        self.assertNotIn("no meaningful change", text)

    def test_2_stable_visit_keeps_size_unmeasured_not_unchanged(self):
        c = self.contexts[1]
        self.assertEqual(c.temporal.verdict, "STABLE")
        self.assertIsNone(c.temporal.size_delta)          # null, not 0.0
        self.assertIsNotNone(c.temporal.border_delta)
        text = c.format_for_prompt()
        self.assertIn("size could not be measured", text)
        self.assertIn("not the same as being unchanged", text)

    def test_3_changed_color_preserves_review_signal(self):
        c = self.contexts[2]
        self.assertEqual(c.temporal.verdict, "CHANGED_COLOR")
        self.assertEqual(c.risk_category, "MEDIUM")
        self.assertTrue(c.requires_review)
        self.assertIn("Flagged for professional review: yes", c.format_for_prompt())

    def test_4_failed_comparison_is_distinct_from_first_visit(self):
        first, failed = self.contexts[0], self.contexts[3]
        # Identical verdict; only the flags separate them (spec section 17).
        self.assertEqual(first.temporal.verdict, failed.temporal.verdict)
        self.assertIn("TEMPORAL_NO_PRIOR_DATA", failed.quality_flags)
        self.assertIn("DEGENERATE_MASK", failed.quality_flags)
        text = failed.format_for_prompt()
        self.assertIn("previous photo was supplied", text)
        self.assertNotIn("no meaningful change", text)

    def test_5_quality_flag_does_not_change_risk(self):
        c = self.contexts[4]
        self.assertIn("LOW_CROP_BLUR", c.quality_flags)
        self.assertEqual(c.risk_category, "MEDIUM")
        self.assertFalse(c.requires_review)
        self.assertIn("blurred", c.format_for_prompt())

    def test_rendered_confidence_is_calibrated_never_raw_softmax(self):
        """The defect CV-8 v1.1 fixed, asserted from the RAG side too."""
        for payload, c in zip(self.payloads, self.contexts):
            raw = payload["diagnosis"]["probabilities"][c.native_class]
            calibrated = payload["uncertainty"]["confidence"]
            self.assertAlmostEqual(c.confidence, calibrated)
            self.assertIn(f"{calibrated:.0%}", c.format_for_prompt())
            if round(raw * 100) != round(calibrated * 100):
                self.assertNotIn(f"{raw:.0%}", c.format_for_prompt())

    def test_unsafe_numbers_never_reach_the_rendered_text(self):
        """magnitude, per-feature deltas and timestamps are not patient-facing."""
        for payload, c in zip(self.payloads, self.contexts):
            text = c.format_for_prompt()
            t = payload["temporal"]
            if t["magnitude"]:
                self.assertNotIn(f"{t['magnitude']:.2f}", text)
            for value in t["per_feature_deltas"].values():
                if value is not None:
                    self.assertNotIn(f"{abs(value):.1f}", text)
            for stamp in t["compared_timestamps"]:
                if stamp:
                    self.assertNotIn(str(stamp), text)


class MalformedInputTests(unittest.TestCase):
    """Spec section 26, test 6 -- fail loudly, never guess a default."""

    def setUp(self) -> None:
        self.payload = _payloads()[0]

    def test_missing_temporal_key_fails_loudly(self):
        bad = copy.deepcopy(self.payload)
        del bad["temporal"]
        with self.assertRaises(CVContractError) as ctx:
            parse_cv_assessment(bad)
        self.assertIn("temporal", str(ctx.exception))

    def test_unrecognised_risk_category_fails_loudly(self):
        bad = copy.deepcopy(self.payload)
        bad["risk_category"] = "CRITICAL"
        with self.assertRaises(CVContractError):
            parse_cv_assessment(bad)

    def test_unrecognised_verdict_fails_loudly(self):
        bad = copy.deepcopy(self.payload)
        bad["temporal"]["verdict"] = "EXPLODED"
        with self.assertRaises(CVContractError):
            parse_cv_assessment(bad)

    def test_missing_contract_version_fails_loudly(self):
        bad = copy.deepcopy(self.payload)
        del bad["contract_version"]
        with self.assertRaises(CVContractError):
            parse_cv_assessment(bad)

    def test_unsupported_major_version_fails_loudly(self):
        bad = copy.deepcopy(self.payload)
        bad["contract_version"] = "2.0"
        with self.assertRaises(CVContractError):
            parse_cv_assessment(bad)

    def test_higher_minor_version_is_accepted(self):
        ok = copy.deepcopy(self.payload)
        ok["contract_version"] = "1.7"
        self.assertEqual(parse_cv_assessment(ok).contract_version, "1.7")

    def test_unknown_quality_flag_is_preserved_not_rejected(self):
        ok = copy.deepcopy(self.payload)
        ok["quality_flags"] = ["NO_TEMPORAL_COMPARISON", "SOME_FUTURE_CV_FLAG"]
        c = parse_cv_assessment(ok)
        self.assertIn("SOME_FUTURE_CV_FLAG", c.quality_flags)
        self.assertIn("SOME_FUTURE_CV_FLAG", c.format_for_prompt())

    def test_wrapper_object_is_rejected_with_a_useful_message(self):
        entry = json.loads(FIXTURES.read_text())[0]
        with self.assertRaises(CVContractError) as ctx:
            parse_cv_assessment(entry)
        self.assertIn("payload", str(ctx.exception))

    def test_null_delta_is_never_coerced_to_zero(self):
        c = parse_cv_assessment(self.payload)
        self.assertIsNone(c.temporal.size_delta)
        self.assertIsNone(c.temporal.border_delta)
        self.assertIsNone(c.temporal.color_delta)


class MultiLesionTests(unittest.TestCase):
    """Spec section 26, test 7 -- highest risk first, none dropped."""

    def setUp(self) -> None:
        payloads = _payloads()
        self.high = parse_cv_assessment(payloads[0])   # HIGH
        self.low = copy.deepcopy(payloads[1])
        self.low["risk_category"] = "LOW"
        self.low["lesion_id"] = "low-risk-lesion"
        self.low = parse_cv_assessment(self.low)

    def test_high_risk_lesion_is_narrated_first(self):
        text = render_cv_context([self.low, self.high])
        self.assertLess(text.index(self.high.lesion_id), text.index(self.low.lesion_id))

    def test_no_lesion_is_silently_dropped(self):
        text = render_cv_context([self.low, self.high])
        self.assertIn(self.low.lesion_id, text)
        self.assertIn(self.high.lesion_id, text)

    def test_ordering_is_deterministic_regardless_of_input_order(self):
        a = order_by_severity([self.low, self.high])
        b = order_by_severity([self.high, self.low])
        self.assertEqual([c.lesion_id for c in a], [c.lesion_id for c in b])

    def test_single_context_and_none_both_render(self):
        self.assertEqual(render_cv_context(None), "")
        self.assertEqual(render_cv_context([]), "")
        self.assertIn(self.high.lesion_id, render_cv_context(self.high))


if __name__ == "__main__":
    unittest.main()
