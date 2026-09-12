"""
CV-4b referral head: "does this lesion need a clinician?"

A logistic head on the same frozen 2048-d backbone features CV-4 already
computes, trained refer-vs-benign. It exists because the product question
was never actually being asked: the shipped system infers referral from
which of six classes wins the argmax, so a melanoma that loses argmax to
NEV is reported as LOW risk.

Measured on the untouched ISIC2019 test split (666 melanomas),
threshold chosen on val, fit on train:

    approach                        melanoma routed   benign referred
    argmax (shipped 6-class)               0.7180            0.2320
    best probability threshold             0.8060            0.3800
    referral head                          0.9024            0.3940

+18.4 points of melanoma routing, 120 more melanomas of 666, for +16.2
points of benign referral. No backbone retraining -- the signal was in
the representation all along and the 6-way softmax was discarding it.
See `analysis/quality/mel_sensitivity/referral_head_result.md`.

## What it is NOT

Not a diagnosis, and not a replacement for `native_class`. It answers
one binary question and is used only as a routing input; narration still
comes from the 6-class prediction. Keeping the two separate is the point
of the change -- previously referral was derived from diagnosis, and
that derivation is what lost the melanomas.

## Standing caveats

- **Dermoscopy only.** Every number above is ISIC2019. Phone photos are
  unmeasured and will be worse. This is the largest open unknown.
- **39.4% benign referral is a real cost**, roughly double the 23.2% the
  shipped argmax produces. That was accepted deliberately as a staffing
  trade, not a free win.
- One split, one seed. Confirm across seeds before relying on it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_HEAD_PATH = Path("checkpoints/referral_head/referral_head.json")


class ReferralHeadError(RuntimeError):
    """The referral head could not be loaded or applied."""


@dataclass(frozen=True)
class ReferralDecision:
    """One lesion's referral signal. Advisory to CV-8, never a diagnosis."""

    refer: bool
    probability: float
    threshold: float

    def to_dict(self) -> dict:
        return {
            "refer": self.refer,
            "probability": self.probability,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class ReferralHead:
    """Logistic head: weights, intercept, and the operating point."""

    coef: np.ndarray
    intercept: float
    threshold: float
    feature_dim: int

    @classmethod
    def load(cls, path: Path | str = DEFAULT_HEAD_PATH) -> "ReferralHead":
        path = Path(path)
        if not path.exists():
            raise ReferralHeadError(
                f"No referral head at {path}. Fit one with "
                "`python -m scripts.train_referral_head`."
            )

        data = json.loads(path.read_text(encoding="utf-8"))
        missing = {"coef", "intercept", "threshold", "feature_dim"} - set(data)
        if missing:
            raise ReferralHeadError(
                f"referral head at {path} is missing {', '.join(sorted(missing))}"
            )

        coef = np.asarray(data["coef"], dtype=np.float64)
        if coef.ndim != 1 or coef.shape[0] != int(data["feature_dim"]):
            raise ReferralHeadError(
                f"referral head coef has shape {coef.shape}, expected "
                f"({data['feature_dim']},)"
            )

        return cls(
            coef=coef,
            intercept=float(data["intercept"]),
            threshold=float(data["threshold"]),
            feature_dim=int(data["feature_dim"]),
        )

    def decide(self, features: np.ndarray) -> ReferralDecision:
        """
        Score one lesion's backbone features.

        Raises on a dimension mismatch rather than scoring anyway: a
        silently wrong score here would move a risk category, and a
        wrong risk category is exactly the harm this head exists to
        prevent.
        """

        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if vector.shape[0] != self.feature_dim:
            raise ReferralHeadError(
                f"expected {self.feature_dim}-d features, got {vector.shape[0]}"
            )

        logit = float(np.dot(self.coef, vector) + self.intercept)
        probability = 1.0 / (1.0 + math.exp(-logit)) if logit > -700 else 0.0

        return ReferralDecision(
            refer=probability >= self.threshold,
            probability=probability,
            threshold=self.threshold,
        )
