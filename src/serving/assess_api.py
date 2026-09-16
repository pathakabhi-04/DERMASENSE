"""
`POST /assess` -- the live feed for CV-8 output.

A transport layer over `DermaSensePipeline.predict()` and nothing more:
no new CV logic, no re-derived thresholds, no reinterpretation of any
result.

## Product mode (added 2026-09-16, Plan A)

In `full` mode every field comes from `RiskAssessment.to_dict()`
unchanged, so the JSON matches
`docs/cv8_sample_outputs/sample_outputs.json` exactly.

**The default is now `narrow`** (`src/serving/product_mode.py`), which
withholds `diagnosis`, `risk_category` and `risk_reason` from the
client. Plan A ships a product that makes no diagnostic claim, and a
client rendering those fields would break that rule regardless of how
carefully anything downstream was narrowed. The full assessment is still
computed and logged server-side for Plan C.

**Narrowing is a presentation boundary at the API edge, not a contract
change.** The CV-8 contract is untouched, and consumers that need the
full assessment -- the RAG layer, whose parser requires `diagnosis` and
`risk_category` as required keys -- must sit INSIDE that boundary and
receive the assessment object server-side, not the narrowed JSON. The
RAG does its own narrowing of what it *narrates*
(`RagPipeline(narrow_product=True)`); the two are separate mechanisms
protecting the same rule at different layers.

Built to the shape already decided in `docs/build_on_baseline_1.md`
Section A, once the prerequisite it named was actually answered:
`scripts/measure_cv8_latency.py` measured p50 0.52s on CPU against a
pre-committed 2.0s rule, so a synchronous endpoint is viable and the
async question does not need reopening.

## Deliberately absent

Section A's "explicitly not now" list, unchanged: no authentication, no
rate limiting, no horizontal scaling, no retry/queue semantics, no
response streaming, no TLS, no request logging or observability. None
of those is justified before a single real request has been exchanged
with the RAG side, and each would be easier to add correctly once there
is real traffic to shape it.

Where this runs is still an infra decision, not a CV one, and is still
open.

## The one design rule that is not just plumbing

`PipelineOutcome` has three values and the API surfaces all three
distinctly, because the orchestrator is explicit that they must not be
conflated:

    QUALITY_REJECTED -- the image was rejected before assessment
    NO_CANDIDATES    -- no lesion was found to assess
    ASSESSED         -- one or more assessments were produced

"We never looked at this" must never reach a consumer looking like "we
looked and it was fine". So a non-ASSESSED outcome returns HTTP 200 with
an empty `assessments` list and the reason named -- not an error, and
never an empty success that a caller could mistake for "no risk".

Run:

    python -m uvicorn src.serving.assess_api:app --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import json

from src.inference.orchestrator import DermaSensePipeline, PipelineOutcome
from src.risk.convergence import CONTRACT_VERSION
from src.serving.product_mode import apply_product_mode, current_mode, is_narrow
from src.temporal.calibration import RulerCalibration
from src.temporal.measurement import LesionMeasurement

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTER_CHECKPOINT = REPO_ROOT / "checkpoints/cv1_5_router/best.pt"
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)

# Loading five checkpoints costs ~9s. Doing it per request would make
# every call multi-second before any inference began, which is exactly
# what Section A's "models loaded once at startup" rule exists to avoid.
_state: dict[str, Any] = {"pipeline": None}


def load_pipeline() -> DermaSensePipeline:
    return DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        detector_weights=None,
        device="cpu",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate before loading anything: a bad mode must stop the server,
    # not surface as a 500 on the first real assessment.
    logger.info("Product mode: %s", current_mode())
    logger.info("Loading CV checkpoints...")
    _state["pipeline"] = load_pipeline()
    logger.info("CV pipeline ready.")
    yield
    _state.clear()


app = FastAPI(
    title="DermaSense CV-8 assessment API",
    version=CONTRACT_VERSION,
    description=(
        "Synchronous CV-8 risk assessment. Returns the locked contract "
        "unchanged; see docs/cv8_sample_outputs/README.md."
    ),
    lifespan=lifespan,
)


def _decode(upload: UploadFile, raw: bytes, field: str) -> np.ndarray:
    """
    Decode an uploaded image to BGR, the layout `predict()` expects.

    A bad upload is the caller's error (400), not a server fault (500).
    Before this check, an undecodable file reached `predict()` and
    surfaced as `TypeError: image_bgr must be a numpy.ndarray`, which
    tells the caller nothing about what they did wrong.
    """

    if not raw:
        raise HTTPException(400, f"'{field}' is empty.")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            400,
            f"'{field}' could not be decoded as an image "
            f"(filename={upload.filename!r}, {len(raw)} bytes). "
            "Send a PNG or JPEG.",
        )

    return image


def _parse_prior_measurement(
    raw: str | None,
) -> tuple[LesionMeasurement | None, RulerCalibration | None]:
    """
    Parse a `prior_measurement` token supplied by the caller.

    Rejected loudly rather than ignored. This value arrives from outside
    the process and feeds a real temporal verdict, so a malformed or
    partially-defaulted measurement would produce a confident-looking
    comparison against fabricated evidence -- worse than no comparison,
    which the contract already represents honestly as NO_PRIOR_DATA.
    """

    if not raw:
        return None, None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(400, f"'prior_measurement' is not valid JSON: {error}")

    if not isinstance(payload, dict):
        raise HTTPException(
            400,
            "'prior_measurement' must be the object this endpoint returned, "
            f"got {type(payload).__name__}.",
        )

    try:
        measurement = LesionMeasurement.from_dict(payload.get("measurement", payload))
        calibration = (
            RulerCalibration.from_dict(payload["calibration"])
            if isinstance(payload.get("calibration"), dict)
            else None
        )
    except ValueError as error:
        raise HTTPException(400, f"'prior_measurement' is malformed: {error}")

    return measurement, calibration


def _measurement_token(result: Any) -> dict[str, Any] | None:
    """
    The opaque token a caller stores and returns at the next visit.

    None when this image could not be measured, which is honest rather
    than unhelpful: a fabricated token would produce a confident-looking
    comparison next visit against evidence that was never computed.
    """

    if result.current_measurement is None:
        return None

    return {
        "measurement": result.current_measurement.to_dict(),
        "calibration": (
            result.current_calibration.to_dict()
            if result.current_calibration is not None
            else None
        ),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus whether the checkpoints finished loading."""

    return {
        "status": "ok",
        "contract_version": CONTRACT_VERSION,
        "pipeline_loaded": _state.get("pipeline") is not None,
    }


@app.post("/assess")
async def assess(
    image: UploadFile = File(..., description="Current photo (PNG/JPEG)."),
    prior_image: UploadFile | None = File(
        None, description="Previous visit's photo of the SAME lesion."
    ),
    lesion_id: str | None = Form(None),
    prior_timestamp: str | None = Form(None),
    current_timestamp: str | None = Form(None),
    prior_measurement: str | None = Form(
        None,
        description=(
            "The `measurement` object this endpoint returned at the previous "
            "visit, as JSON. Supplying it makes `prior_image` unnecessary and "
            "removes ~0.5s from the request."
        ),
    ),
) -> dict[str, Any]:
    """
    Assess one image, optionally against a previous visit's photo.

    `prior_image` is only compared when this image yields exactly one
    candidate; with several, CV-7 cannot tell which lesion it belongs to
    and says so via `PRIOR_IMAGE_PAIRING_AMBIGUOUS` rather than guessing.

    `*_timestamp` are opaque passthrough. CV-7 never parses them and
    does not require them to be dates; they reappear verbatim in
    `temporal.compared_timestamps`.

    ## Reusing the previous visit's measurement

    Each assessment comes back with a `measurement` object. Store it,
    and send it as `prior_measurement` at the next visit instead of
    re-uploading the previous photo. The prior image was already
    segmented and measured then, so re-measuring it is repeated work:
    ~0.5s of a ~2.2s returning-visit request on CPU.

    It is bit-identical to re-measuring -- same verdict, same magnitude,
    same per-feature deltas -- because the delta is computed from the two
    measurements and never from the pixels. Treat the object as opaque:
    send back exactly what you were given, unmodified.

    This deliberately keeps the service stateless. Who owns lesion
    history is still an open question (docs/build_on_baseline_1.md
    Section A, question 4), and round-tripping the measurement through
    the caller means that question does not have to be answered to get
    the speedup.

    Returns one assessment per detected lesion -- zero, one, or several.
    """

    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "Pipeline is still loading; retry shortly.")

    image_bgr = _decode(image, await image.read(), "image")

    prior_bgr = None
    if prior_image is not None and prior_image.filename:
        prior_bgr = _decode(prior_image, await prior_image.read(), "prior_image")

    cached_measurement, cached_calibration = _parse_prior_measurement(prior_measurement)

    result = pipeline.predict(
        image_bgr,
        lesion_id=lesion_id,
        prior_image_bgr=prior_bgr,
        prior_timestamp=prior_timestamp,
        current_timestamp=current_timestamp,
        prior_measurement=cached_measurement,
        prior_calibration=cached_calibration,
        measure_current=True,
    )

    # The assessment objects stay the locked contract, untouched: the
    # measurement token is transport-level metadata, and it describes the
    # IMAGE rather than any one lesion, so it belongs on the envelope.
    # Putting it inside an assessment would also break the property that
    # these match docs/cv8_sample_outputs/ exactly.
    assessments = [
        candidate.risk_assessment.to_dict()
        for candidate in result.candidates
        if candidate.risk_assessment is not None
    ]

    # Plan A §4: the diagnosis path keeps running and is LOGGED -- those
    # predictions beside a future biopsy result are the evaluation Plan C
    # exists to enable -- but it does not reach the client. Logged before
    # narrowing, returned after.
    if is_narrow() and assessments:
        logger.info(
            "cv8_internal %s",
            json.dumps({"assessments": assessments}, default=str),
        )
    assessments = apply_product_mode(assessments)

    return {
        "outcome": result.outcome.value,
        "product_mode": current_mode(),
        # Named so an empty list can never be misread as "assessed, no
        # risk" -- see the module docstring.
        "assessed": result.outcome is PipelineOutcome.ASSESSED,
        "num_assessments": len(assessments),
        "assessments": assessments,
        "image_quality": {
            "usable": result.quality.usable,
            "score": result.quality.quality_score,
        },
        # Store this and send it back as `prior_measurement` next visit.
        # Opaque: return exactly what you were given.
        "measurement": _measurement_token(result),
    }
