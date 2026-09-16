"""
Plan A's shipping guarantee: no diagnosis or risk claim leaves the API.

`docs/plan_a_narrow_product_spec.md` §5. These tests exist because the
failure they prevent is a shipped product telling a worried person their
melanoma is a mole -- so they check the DEFAULT, not just the configured
behaviour, and they check by allow-list rather than by naming the fields
that happen to be unsafe today.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from src.serving.product_mode import (
    ProductModeError,
    apply_product_mode,
    current_mode,
    is_narrow,
    narrow_assessment,
)

FIXTURES = Path(__file__).resolve().parents[1] / "docs/cv8_sample_outputs/sample_outputs.json"
FORBIDDEN_KEYS = ("diagnosis", "risk_category", "risk_reason", "uncertainty")


def _assessments() -> list[dict]:
    return [entry["payload"] for entry in json.loads(FIXTURES.read_text())]


class ProductModeDefaults(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("DERMASENSE_PRODUCT_MODE", None)

    def tearDown(self):
        os.environ.pop("DERMASENSE_PRODUCT_MODE", None)
        if self._saved is not None:
            os.environ["DERMASENSE_PRODUCT_MODE"] = self._saved

    def test_default_is_narrow(self):
        """Unset must mean safe. A forgotten variable cannot emit a diagnosis."""
        self.assertEqual(current_mode(), "narrow")
        self.assertTrue(is_narrow())

    def test_unrecognised_mode_fails_loudly(self):
        os.environ["DERMASENSE_PRODUCT_MODE"] = "partial"
        with self.assertRaises(ProductModeError):
            current_mode()

    def test_full_mode_must_be_explicit(self):
        os.environ["DERMASENSE_PRODUCT_MODE"] = "full"
        self.assertFalse(is_narrow())


class NarrowAssessment(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("DERMASENSE_PRODUCT_MODE", None)
        self.payloads = _assessments()

    def tearDown(self):
        os.environ.pop("DERMASENSE_PRODUCT_MODE", None)
        if self._saved is not None:
            os.environ["DERMASENSE_PRODUCT_MODE"] = self._saved

    def test_no_diagnosis_or_risk_survives(self):
        for payload in self.payloads:
            narrowed = narrow_assessment(payload)
            for key in FORBIDDEN_KEYS:
                self.assertNotIn(key, narrowed)
            serialised = json.dumps(narrowed)
            self.assertNotIn(payload["diagnosis"]["native_class"], serialised)
            self.assertNotIn(payload["risk_category"], serialised)

    def test_what_the_product_ships_survives(self):
        for payload in self.payloads:
            narrowed = narrow_assessment(payload)
            self.assertEqual(narrowed["lesion_id"], payload["lesion_id"])
            self.assertIn("temporal", narrowed)
            self.assertIn("quality_flags", narrowed)

    def test_requires_review_is_escalation_only(self):
        """False is reassurance by implication, so it is omitted."""
        flagged = {"lesion_id": "a", "uncertainty": {"requires_review": True}}
        unflagged = {"lesion_id": "b", "uncertainty": {"requires_review": False}}
        self.assertTrue(narrow_assessment(flagged)["requires_review"])
        self.assertNotIn("requires_review", narrow_assessment(unflagged))

    def test_new_contract_fields_are_withheld_by_default(self):
        """Allow-list, not deny-list: a field added to CV-8 later must not
        leak just because nobody remembered to blacklist it."""
        payload = dict(self.payloads[0])
        payload["some_future_risk_signal"] = "HIGH"
        self.assertNotIn("some_future_risk_signal", narrow_assessment(payload))

    def test_apply_product_mode_narrows_every_assessment(self):
        narrowed = apply_product_mode(self.payloads)
        self.assertEqual(len(narrowed), len(self.payloads))
        for item in narrowed:
            for key in FORBIDDEN_KEYS:
                self.assertNotIn(key, item)

    def test_full_mode_passes_the_contract_through_untouched(self):
        os.environ["DERMASENSE_PRODUCT_MODE"] = "full"
        self.assertEqual(apply_product_mode(self.payloads), self.payloads)


if __name__ == "__main__":
    unittest.main()
