"""
Parse a CV-8 `RiskAssessment` payload into a `CVAssessmentContext`.

Takes a dict, returns a context object. It deliberately knows nothing
about where the dict came from -- an HTTP response, a queue message, a
fixture file -- because no live delivery mechanism has been decided on
the CV side yet (primary spec section 9). When one is, the only new code
is how the JSON arrives; none of the validation below changes.

## Fail loudly, do not degrade

Spec section 16.1 draws the line, and this module implements it exactly:

- **An unrecognised `quality_flags` entry is fine.** Preserve it, keep
  going. The CV side adds flags over time and none of them ever change
  risk, so a new one is information, not an error.
- **A missing/renamed top-level key, or an unrecognised `risk_category`
  or `temporal.verdict`, is fatal.** Raise instead of building a
  half-parsed context with guessed defaults, because the next thing that
  happens to a context object is that an LLM narrates it to a patient. A
  guessed default is indistinguishable from a real measurement by then.

That mirrors the discipline the CV side already applies to itself:
`NO_PRIOR_DATA` is a first-class outcome rather than a guess, and a
ruler calibration reports `confident=False` rather than inventing a
scale.

## Version handling

`contract_version` arrived in CV-8 v1.1. An unknown MAJOR is fatal (the
payload's shape may have changed underneath us); a higher MINOR is fine
(minor bumps are additive by contract). A payload with no version at all
is pre-1.1 and is rejected, because those payloads carry the raw softmax
in `risk_reason` where the calibrated figure belongs -- exactly the bug
the version field was introduced to make detectable.
"""

from __future__ import annotations

from typing import Any

from src.rag.cv_context.schema import (
    RISK_CATEGORIES,
    TEMPORAL_VERDICTS,
    CVAssessmentContext,
    TemporalContext,
    order_by_severity,  # re-exported: callers historically imported it here
)

SUPPORTED_CONTRACT_MAJOR = 1
MINIMUM_CONTRACT_VERSION = "1.1"

_REQUIRED_TOP_LEVEL = ("lesion_id", "diagnosis", "risk_category", "risk_reason",
                       "temporal", "uncertainty", "quality_flags")
_REQUIRED_TEMPORAL = ("verdict", "magnitude", "confidence", "per_feature_deltas",
                      "compared_timestamps")


class CVContractError(ValueError):
    """
    A CV-8 payload could not be parsed safely.

    Always raised rather than returning a partially-populated context:
    the caller must be able to distinguish "no CV assessment" from "a CV
    assessment with invented fields".
    """


def _require(payload: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise CVContractError(
            f"CV-8 payload is missing required {where} key(s): "
            f"{', '.join(sorted(missing))}. Refusing to parse -- a structurally "
            "different contract must fail loudly rather than be guessed at."
        )


def _check_version(raw: Any) -> str:
    if raw is None:
        raise CVContractError(
            "CV-8 payload has no `contract_version`. Payloads predating v1.1 "
            "carry the raw softmax score in `risk_reason` where the calibrated "
            "confidence belongs; regenerate the payload from CV-8 v1.1 or later."
        )

    version = str(raw)
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError) as error:
        raise CVContractError(
            f"CV-8 `contract_version` is not a recognisable version: {version!r}."
        ) from error

    if major != SUPPORTED_CONTRACT_MAJOR:
        raise CVContractError(
            f"CV-8 contract major version {major} is not supported "
            f"(this parser understands {SUPPORTED_CONTRACT_MAJOR}.x). A major bump "
            "means the payload shape changed; update the parser rather than "
            "parsing it as though nothing moved."
        )

    return version


def _nullable_float(value: Any, field_name: str) -> float | None:
    """
    `None` means "could not be measured" and is preserved as-is; it is
    never coerced to 0.0, which would mean "measured, no change".

    All three per-feature deltas are independently nullable (spec
    section 14.1) -- not just `size`.
    """

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise CVContractError(
            f"CV-8 `per_feature_deltas.{field_name}` must be a number or null, "
            f"got {value!r}."
        ) from error


def _parse_temporal(block: Any) -> TemporalContext:
    if not isinstance(block, dict):
        raise CVContractError(
            f"CV-8 `temporal` must be an object, got {type(block).__name__}."
        )

    _require(block, _REQUIRED_TEMPORAL, "temporal")

    verdict = block["verdict"]
    if verdict not in TEMPORAL_VERDICTS:
        raise CVContractError(
            f"CV-8 `temporal.verdict` is not a recognised value: {verdict!r}. "
            f"Known verdicts: {', '.join(sorted(TEMPORAL_VERDICTS))}."
        )

    deltas = block["per_feature_deltas"]
    if not isinstance(deltas, dict):
        raise CVContractError(
            "CV-8 `temporal.per_feature_deltas` must be an object, got "
            f"{type(deltas).__name__}."
        )
    _require(deltas, ("size", "border", "color"), "per_feature_deltas")

    stamps = block["compared_timestamps"]
    if not isinstance(stamps, (list, tuple)) or len(stamps) != 2:
        raise CVContractError(
            "CV-8 `temporal.compared_timestamps` must be a 2-element list, got "
            f"{stamps!r}."
        )

    return TemporalContext(
        verdict=verdict,
        magnitude=float(block["magnitude"]),
        confidence=float(block["confidence"]),
        size_delta=_nullable_float(deltas["size"], "size"),
        border_delta=_nullable_float(deltas["border"], "border"),
        color_delta=_nullable_float(deltas["color"], "color"),
        # Opaque passthrough: never parsed as dates, both entries
        # independently nullable.
        compared_images=(stamps[0], stamps[1]),
    )


def parse_cv_assessment(payload: dict[str, Any]) -> CVAssessmentContext:
    """
    Validate and parse one CV-8 payload.

    Raises `CVContractError` on anything structurally unexpected. Note
    the argument is the contract object itself -- if you are reading
    `docs/cv8_sample_outputs/sample_outputs.json`, each entry wraps it as
    `{description, source, payload}`, so pass `entry["payload"]`.
    """

    if not isinstance(payload, dict):
        raise CVContractError(
            f"CV-8 payload must be a JSON object, got {type(payload).__name__}. "
            "If you are reading sample_outputs.json, pass entry['payload']."
        )

    _require(payload, _REQUIRED_TOP_LEVEL, "top-level")
    version = _check_version(payload.get("contract_version"))

    diagnosis = payload["diagnosis"]
    if not isinstance(diagnosis, dict):
        raise CVContractError(
            f"CV-8 `diagnosis` must be an object, got {type(diagnosis).__name__}."
        )
    _require(diagnosis, ("native_class", "probabilities"), "diagnosis")

    risk_category = payload["risk_category"]
    if risk_category not in RISK_CATEGORIES:
        raise CVContractError(
            f"CV-8 `risk_category` is not a recognised value: {risk_category!r}. "
            f"Known categories: {', '.join(sorted(RISK_CATEGORIES))}."
        )

    uncertainty = payload["uncertainty"]
    if not isinstance(uncertainty, dict):
        raise CVContractError(
            f"CV-8 `uncertainty` must be an object, got {type(uncertainty).__name__}."
        )
    _require(uncertainty, ("confidence", "requires_review"), "uncertainty")

    flags = payload["quality_flags"]
    if not isinstance(flags, (list, tuple)):
        raise CVContractError(
            f"CV-8 `quality_flags` must be a list, got {type(flags).__name__}."
        )

    return CVAssessmentContext(
        lesion_id=str(payload["lesion_id"]),
        native_class=str(diagnosis["native_class"]),
        probabilities={str(k): float(v) for k, v in diagnosis["probabilities"].items()},
        risk_category=risk_category,
        risk_reason=str(payload["risk_reason"]),
        temporal=_parse_temporal(payload["temporal"]),
        # The CALIBRATED figure -- never diagnosis.probabilities[native_class].
        confidence=float(uncertainty["confidence"]),
        requires_review=bool(uncertainty["requires_review"]),
        # Unrecognised flags are preserved, never rejected (spec section 16).
        quality_flags=tuple(str(f) for f in flags),
        contract_version=version,
    )


__all__ = ["CVContractError", "parse_cv_assessment", "order_by_severity"]
