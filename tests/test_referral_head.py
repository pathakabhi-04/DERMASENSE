"""
CV-4b referral head, and its effect on CV-8 risk convergence.

The head answers "does this need a clinician?" directly, instead of
inferring it from which of six classes wins the argmax. On ISIC2019
test it routes 0.9024 of melanomas versus 0.7180 for the shipped
argmax. See analysis/quality/mel_sensitivity/referral_head_result.md.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.inference.orchestrator import CandidateResult
from src.risk.action_mapping import ProductAction
from src.risk.convergence import CONTRACT_VERSION, RiskCategory, assess_risk
from src.risk.referral_head import (
    ReferralDecision,
    ReferralHead,
    ReferralHeadError,
)
from src.risk.safety_gate import GateDecision

HEAD_PATH = Path("checkpoints/referral_head/referral_head.json")


def _candidate(action: ProductAction, *, requires_review: bool = False) -> CandidateResult:
    return CandidateResult(
        candidate_index=0, box_pixels=(0, 0, 10, 10), detection_confidence=None,
        predicted_class="NEV", confidence=0.9, probabilities={"NEV": 0.9},
        product_action=action,
        gate_decision=GateDecision.REVIEW if requires_review else GateDecision.AUTO_RELEASE,
        requires_review=requires_review, gate_reason="test",
        mask_area_fraction=0.2, mask_degenerate=False, mask_touches_border=False,
        crop_blur=0.5, crop_contrast=0.5, calibrated_confidence=0.74,
        ensemble_agree=None,
    )


def _refer(refer: bool) -> ReferralDecision:
    return ReferralDecision(refer=refer, probability=0.9 if refer else 0.001,
                            threshold=0.003)


class FloorBehaviourTests(unittest.TestCase):
    """A floor, not a ratchet: it prevents LOW and nothing else."""

    def test_referral_raises_low_to_medium(self):
        result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1",
                             referral=_refer(True))
        self.assertEqual(result.risk_category, RiskCategory.MEDIUM)
        self.assertTrue(result.requires_review)
        self.assertIn("REFERRAL_HEAD_RAISED_FROM_LOW", result.quality_flags)
        self.assertIn("referral head", result.risk_reason)

    def test_referral_does_not_promote_medium_to_high(self):
        """
        The head was validated on refer-vs-benign, not on degree of
        risk. Promoting MEDIUM would assert something unmeasured.
        """
        result = assess_risk(_candidate(ProductAction.EVALUATE_SOON), lesion_id="L1",
                             referral=_refer(True))
        self.assertEqual(result.risk_category, RiskCategory.MEDIUM)
        self.assertNotIn("REFERRAL_HEAD_RAISED_FROM_LOW", result.quality_flags)

    def test_referral_leaves_high_alone(self):
        result = assess_risk(_candidate(ProductAction.URGENT_EVALUATION), lesion_id="L1",
                             referral=_refer(True))
        self.assertEqual(result.risk_category, RiskCategory.HIGH)

    def test_no_referral_signal_changes_nothing(self):
        """Rollback path: absent the head, CV-8 behaves exactly as before."""
        without = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1")
        with_no = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1",
                              referral=None)
        self.assertEqual(without.to_dict(), with_no.to_dict())
        self.assertEqual(without.risk_category, RiskCategory.LOW)

    def test_not_referred_leaves_low_alone(self):
        result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1",
                             referral=_refer(False))
        self.assertEqual(result.risk_category, RiskCategory.LOW)
        self.assertNotIn("REFERRAL_HEAD_RAISED_FROM_LOW", result.quality_flags)

    def test_referral_forces_review_even_when_not_raised(self):
        result = assess_risk(_candidate(ProductAction.EVALUATE_SOON), lesion_id="L1",
                             referral=_refer(True))
        self.assertTrue(result.requires_review)

    def test_contract_version_is_1_2(self):
        result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1",
                             referral=_refer(True))
        self.assertEqual(result.to_dict()["contract_version"], CONTRACT_VERSION)
        self.assertEqual(CONTRACT_VERSION, "1.2")

    def test_native_class_is_never_touched_by_the_head(self):
        """It is a routing signal, not a diagnosis."""
        referred = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1",
                               referral=_refer(True))
        plain = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1")
        self.assertEqual(referred.native_class, plain.native_class)
        self.assertEqual(referred.probabilities, plain.probabilities)


@unittest.skipUnless(HEAD_PATH.exists(), "referral head not fitted")
class HeadLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.head = ReferralHead.load(HEAD_PATH)

    def test_loads_with_the_expected_shape(self):
        self.assertEqual(self.head.feature_dim, 2048)
        self.assertEqual(self.head.coef.shape, (2048,))

    def test_wrong_feature_dimension_is_rejected(self):
        with self.assertRaises(ReferralHeadError):
            self.head.decide(np.zeros(64))

    def test_decision_respects_its_own_threshold(self):
        big = self.head.decide(self.head.coef * 10.0)
        self.assertGreaterEqual(big.probability, big.threshold)
        self.assertTrue(big.refer)

    def test_missing_head_file_raises(self):
        with self.assertRaises(ReferralHeadError):
            ReferralHead.load(Path("checkpoints/referral_head/nope.json"))

    def test_shipped_head_records_the_operating_point_it_was_measured_at(self):
        data = json.loads(HEAD_PATH.read_text())
        self.assertAlmostEqual(data["target_benign_referral"], 0.40)
        self.assertGreaterEqual(data["test_melanoma_routed"], 0.90)


if __name__ == "__main__":
    unittest.main()
