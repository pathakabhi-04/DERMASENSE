"""
CV-7 delta/verdict tests.

All synthetic -- LesionMeasurement instances built directly with known
values, so the delta arithmetic and verdict priority logic are checked
deterministically without needing real images or a model.
"""

from __future__ import annotations

from src.temporal.delta import (
    COLOR_DELTA_E_THRESHOLD,
    GROWTH_PCT_THRESHOLD,
    LesionDelta,
    TemporalVerdict,
    compute_delta,
)
from src.temporal.measurement import LesionMeasurement


def _measurement(
    valid=True,
    area_fraction=0.05,
    compactness=1.1,
    mean_lab=(60.0, 10.0, 10.0),
    diameter_mm=5.0,
    area_mm2=20.0,
) -> LesionMeasurement:
    return LesionMeasurement(
        valid=valid,
        reason="ok" if valid else "empty mask",
        area_fraction=area_fraction if valid else None,
        compactness=compactness if valid else None,
        mean_lab=mean_lab if valid else None,
        diameter_mm=diameter_mm,
        area_mm2=area_mm2,
    )


def test_no_prior_data_when_either_visit_invalid():
    earlier = _measurement(valid=False)
    later = _measurement(valid=True)

    delta = compute_delta(earlier, later)

    assert delta.verdict == TemporalVerdict.NO_PRIOR_DATA
    assert delta.confidence == 0.0
    assert delta.size_delta_mm is None
    assert delta.border_delta is None
    assert delta.color_delta is None


def test_stable_when_nothing_crosses_threshold():
    earlier = _measurement(diameter_mm=5.0, compactness=1.1, mean_lab=(60.0, 10.0, 10.0))
    later = _measurement(diameter_mm=5.1, compactness=1.12, mean_lab=(60.5, 10.2, 10.1))

    delta = compute_delta(earlier, later)

    assert delta.verdict == TemporalVerdict.STABLE
    assert delta.confidence == 1.0  # calibration present on both sides


def test_growing_when_size_grows_past_threshold():
    earlier = _measurement(diameter_mm=5.0)
    grown = 5.0 * (1 + (GROWTH_PCT_THRESHOLD + 5) / 100.0)
    later = _measurement(diameter_mm=grown)

    delta = compute_delta(earlier, later)

    assert delta.verdict == TemporalVerdict.GROWING
    assert delta.size_pct_change > GROWTH_PCT_THRESHOLD
    assert delta.magnitude >= 1.0


def test_shrinking_when_size_shrinks_past_threshold():
    earlier = _measurement(diameter_mm=5.0)
    shrunk = 5.0 * (1 - (GROWTH_PCT_THRESHOLD + 5) / 100.0)
    later = _measurement(diameter_mm=shrunk)

    delta = compute_delta(earlier, later)

    assert delta.verdict == TemporalVerdict.SHRINKING


def test_small_absolute_growth_on_tiny_lesion_does_not_trigger_growing():
    # a lesion growing from 0.1mm to 0.2mm is a 100% relative change but
    # a trivial 0.1mm absolute change -- the abs-mm floor should hold this
    # to STABLE rather than a spurious GROWING on measurement noise.
    earlier = _measurement(diameter_mm=0.1)
    later = _measurement(diameter_mm=0.2)

    delta = compute_delta(earlier, later)

    assert delta.verdict == TemporalVerdict.STABLE


def test_changed_color_when_color_shift_past_threshold():
    earlier = _measurement(diameter_mm=5.0, mean_lab=(60.0, 10.0, 10.0))
    later = _measurement(diameter_mm=5.0, mean_lab=(60.0, 10.0 + COLOR_DELTA_E_THRESHOLD + 2, 10.0))

    delta = compute_delta(earlier, later)

    assert delta.verdict == TemporalVerdict.CHANGED_COLOR


def test_size_change_takes_priority_over_color_change():
    earlier = _measurement(diameter_mm=5.0, mean_lab=(60.0, 10.0, 10.0))
    grown = 5.0 * (1 + (GROWTH_PCT_THRESHOLD + 5) / 100.0)
    later = _measurement(
        diameter_mm=grown, mean_lab=(60.0, 10.0 + COLOR_DELTA_E_THRESHOLD + 2, 10.0)
    )

    delta = compute_delta(earlier, later)

    assert delta.verdict == TemporalVerdict.GROWING


def test_size_delta_none_when_calibration_unavailable_on_either_side():
    earlier = LesionMeasurement(
        valid=True, reason="ok", area_fraction=0.05, compactness=1.1,
        mean_lab=(60.0, 10.0, 10.0), diameter_mm=None, area_mm2=None,
    )
    later = _measurement(diameter_mm=8.0)

    delta = compute_delta(earlier, later)

    assert delta.size_delta_mm is None
    assert delta.size_pct_change is None
    # border/color are unaffected by missing calibration
    assert delta.border_delta is not None
    assert delta.color_delta is not None
    # only 2/3 feature channels available
    assert delta.confidence < 1.0


def test_border_delta_never_sets_headline_verdict_alone():
    # a large compactness (border) change with size/color both stable
    # should not, by itself, produce anything other than STABLE --
    # the locked contract has no CHANGED_BORDER verdict.
    earlier = _measurement(diameter_mm=5.0, compactness=1.0, mean_lab=(60.0, 10.0, 10.0))
    later = _measurement(diameter_mm=5.0, compactness=3.0, mean_lab=(60.0, 10.0, 10.0))

    delta = compute_delta(earlier, later)

    assert delta.verdict == TemporalVerdict.STABLE
    assert delta.border_delta == 2.0
