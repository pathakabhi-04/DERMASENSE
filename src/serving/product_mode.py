"""
Product mode: what the served API is allowed to tell a user.

Plan A (`docs/plan_a_narrow_product_spec.md`) is what ships. Its §5 rule:

    The narrow product never emits reassurance. It never says low risk,
    benign, or probably fine, and it never names a diagnosis.

`RiskAssessment.to_dict()` carries `native_class`, `probabilities`,
`risk_category` and `risk_reason`. A client rendering that response would
break the rule no matter how carefully the RAG layer was narrowed --
so the narrowing has to happen before the JSON leaves the server.

## Narrow is the DEFAULT, deliberately

Everywhere else in this codebase the safe behaviour is opt-in, to avoid
changing what existing callers see. Here it is opt-out, because the
failure mode is a shipped product telling a worried person their
melanoma is a mole. A forgotten environment variable must fail toward
silence, not toward a diagnosis.

    DERMASENSE_PRODUCT_MODE=narrow   (default)  Plan A
    DERMASENSE_PRODUCT_MODE=full                Plan B / internal only

## What survives narrowing

The change verdict, the capture-quality flags, and the lesion id -- the
things the narrow product actually ships. The full assessment is still
computed and still logged server-side (§4 of the spec): those
predictions sitting beside a future biopsy result are the evaluation
Plan C exists to make possible.

## Why `requires_review` is escalation-only

`requires_review: false` is reassurance by implication, so it is emitted
only when True and omitted otherwise. Same one-directional discipline as
CV-4b's floor-not-ratchet and the abstention rule: flag the concerning
direction, never affirm the reassuring one.
"""

from __future__ import annotations

import os
from typing import Any

NARROW = "narrow"
FULL = "full"
_ENV_VAR = "DERMASENSE_PRODUCT_MODE"

# Fields that constitute a diagnostic or risk claim. Removed wholesale in
# narrow mode rather than rewritten, so a new key added to the contract
# is excluded by default rather than leaking until someone notices.
_ALLOWED_IN_NARROW = ("contract_version", "lesion_id", "temporal", "quality_flags")


class ProductModeError(RuntimeError):
    """The configured product mode is not a recognised value."""


def current_mode() -> str:
    mode = os.environ.get(_ENV_VAR, NARROW).strip().lower()
    if mode not in (NARROW, FULL):
        raise ProductModeError(
            f"{_ENV_VAR}={mode!r} is not recognised. Use {NARROW!r} (Plan A, the "
            f"shipped product) or {FULL!r} (Plan B / internal). Refusing to guess, "
            "because guessing wrong means emitting a diagnosis."
        )
    return mode


def is_narrow() -> bool:
    return current_mode() == NARROW


def narrow_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    """Strip every diagnostic and risk claim from one CV-8 assessment.

    Allow-list, not deny-list: anything the contract gains later is
    withheld until somebody decides it is safe to show.
    """
    narrowed = {
        key: assessment[key] for key in _ALLOWED_IN_NARROW if key in assessment
    }

    # Escalation-only: present when True, absent when False.
    uncertainty = assessment.get("uncertainty") or {}
    if uncertainty.get("requires_review"):
        narrowed["requires_review"] = True

    return narrowed


def apply_product_mode(assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not is_narrow():
        return assessments
    return [narrow_assessment(assessment) for assessment in assessments]
