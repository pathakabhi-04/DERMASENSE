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

## Plan C endpoints (added 2026-09-16)

`/accounts`, `/consent`, `/outcomes`, `/revoke` and `/plan_c/coverage`
are the storage half of the collection loop -- see the section above
them near the bottom of this file. `/assess` gains an optional
`patient_pseudonym`: supply it and the capture is retained IF that
account consented, and the response carries the `linkage_code` to print
for the clinician. Persistence is off unless `DERMASENSE_CAPTURE_STORE`
names a directory.

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
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import json

from src.inference.orchestrator import DermaSensePipeline, PipelineOutcome
from src.risk.convergence import CONTRACT_VERSION
from src.serving.capture_store import (
    CaptureStore,
    ConsentRequired,
    UnknownLinkageCode,
)
from src.serving.linkage import LinkageError
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


# Persistence is OPT-IN, by an explicit path. A development instance
# must not quietly accumulate medical photographs because someone ran
# the server to try an endpoint: if the operator has not named a place
# for captures to live, there is no place for them to live.
CAPTURE_STORE_ENV = "DERMASENSE_CAPTURE_STORE"


def _store() -> CaptureStore:
    """The Plan C store, or 503 if this deployment has none configured."""

    store = _state.get("store")
    if store is None:
        raise HTTPException(
            503,
            f"Capture storage is not configured on this deployment. Set "
            f"{CAPTURE_STORE_ENV} to a directory to enable Plan C collection.",
        )
    return store


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
    store_root = os.environ.get(CAPTURE_STORE_ENV)
    if store_root:
        _state["store"] = CaptureStore(store_root)
        logger.info("Plan C capture store: %s", store_root)
    else:
        logger.info("Plan C capture store: disabled (%s unset)", CAPTURE_STORE_ENV)

    logger.info("Loading CV checkpoints...")
    _state["pipeline"] = load_pipeline()
    logger.info("CV pipeline ready.")
    yield
    store = _state.get("store")
    if store is not None:
        store.close()
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


def _parse_device(raw: str | None) -> dict[str, str]:
    """Device model and OS, which Plan C §2 collects to measure whether
    capture quality tracks hardware.

    Malformed input is dropped rather than rejected: this is metadata
    about the phone, and losing it must never cost the user the capture
    itself. Everything that WOULD be worth a 400 -- the image, the prior
    measurement -- is validated strictly above.
    """

    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("device metadata was not valid JSON; dropped")
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items()}


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
    body_site: str | None = Form(
        None, description="Plan C: where on the body, from the client's body map."
    ),
    device: str | None = Form(
        None, description='Plan C: {"model": ..., "os": ...} as JSON.'
    ),
    patient_pseudonym: str | None = Form(
        None,
        description=(
            "Plan C: the account this capture belongs to. Supplying it stores "
            "the capture IF that account has consented to training use, and "
            "returns the linkage code to print for the clinician."
        ),
    ),
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

    image_raw = await image.read()
    image_bgr = _decode(image, image_raw, "image")

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

    # Stored BEFORE narrowing, and only the full assessment is worth
    # storing: a prediction sitting beside a future biopsy result is the
    # evaluation this project has never been able to make
    # (decision_001). `save_capture` reads consent from the store and
    # returns None when it does not permit retention, so a declined
    # account silently stores nothing -- no record, no photograph.
    linkage_code = None
    if patient_pseudonym and _state.get("store") is not None:
        try:
            record = _state["store"].save_capture(
                patient_pseudonym,
                image_bytes=image_raw,
                lesion_id=lesion_id,
                # All of them, not the first. One photo can yield
                # several lesions and the biopsy result will name one;
                # keeping only the first would silently discard the
                # prediction the outcome turns out to be about.
                assessment={
                    "assessments": assessments,
                    # Plan C §2: the honest distribution of real-world
                    # photos, including the ones CV-1 rejected. A dataset
                    # of only the images that passed quality is exactly
                    # the clinic-conditions bias this collection exists
                    # to escape.
                    "outcome": result.outcome.value,
                    "image_quality": {
                        "usable": result.quality.usable,
                        "score": result.quality.quality_score,
                    },
                },
                body_site=body_site,
                device=_parse_device(device),
            )
        except ConsentRequired as error:
            raise HTTPException(409, str(error))
        if record is not None:
            linkage_code = record.linkage_code

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
        # Present only when the capture was actually retained. Absent
        # means nothing was stored -- no store configured, no pseudonym
        # supplied, or consent declined -- and a client must not print a
        # code for a capture that does not exist to be joined to.
        "linkage_code": linkage_code,
    }


# ----------------------------------------------------------------------
# Plan C: accounts, consent, the clinician handoff, and revocation.
#
# These endpoints are the storage layer the ship checklist listed as
# blocking. They are the API half; the UI half -- the consent screen, the
# printed clinician page, the revoke button -- is a client task, and
# `plan_a_narrow_product_spec.md` §5 governs its copy.
#
# ## Authentication, honestly
#
# There is none, as elsewhere in this file. What protects these is that
# `patient_pseudonym` (128 bits) and `linkage_code` (~40 bits) are
# unguessable rather than sequential, so holding one grants access to
# exactly one subject and reveals nothing about any other. That is
# capability security, and it is enough for a supervised pilot and NOT
# enough for public deployment: a leaked pseudonym is a permanent
# credential with no rotation path. Real auth is a prerequisite for real
# users, and saying so here is cheaper than discovering it later.
# ----------------------------------------------------------------------


@app.post("/accounts", status_code=201)
def create_account() -> dict[str, Any]:
    """Mint an account. Returns the only identity the server holds for
    this person -- random, and never derived from an email or a phone
    number (`linkage.py` on why a hash would not be pseudonymous)."""

    return {"patient_pseudonym": _store().create_account()}


@app.post("/accounts/{patient_pseudonym}/consent")
def record_consent(
    patient_pseudonym: str,
    training_use: bool = Form(
        ..., description="Whether captures may be retained to train models."
    ),
    policy_version: str = Form(
        ...,
        description=(
            "The approved consent wording the user actually agreed to. Required, "
            "with no default: the wording is a legal/ethics deliverable "
            "(plan_c_linkage_spec.md §6), and a default here would record "
            "agreement to a document that does not exist."
        ),
    ),
) -> dict[str, Any]:
    """Record a consent decision.

    `training_use` is separable from using the app: declining it still
    gets the whole product, and simply means nothing is retained.
    """

    try:
        consent = _store().record_consent(
            patient_pseudonym,
            training_use=training_use,
            policy_version=policy_version,
        )
    except ConsentRequired as error:
        raise HTTPException(404, str(error))
    except ValueError as error:
        raise HTTPException(400, str(error))

    return consent.to_dict()


@app.post("/accounts/{patient_pseudonym}/revoke")
def revoke(patient_pseudonym: str) -> dict[str, Any]:
    """Withdraw consent and delete what was retained under it.

    The receipt says what was deleted AND what revocation cannot do --
    a capture already used in training has influenced weights that
    cannot be selectively unlearned. That sentence travels in the
    response so a client cannot present revocation as more complete than
    it is.
    """

    try:
        return _store().revoke(patient_pseudonym).to_dict()
    except ConsentRequired as error:
        raise HTTPException(404, str(error))


@app.post("/outcomes", status_code=201)
def record_outcome(
    linkage_code: str = Form(..., description="The code printed on the clinician page."),
    label: str = Form(...),
    label_source: str = Form(..., description="'biopsy' or 'consensus'."),
    reported_at: str = Form(...),
) -> dict[str, Any]:
    """Attach a clinical result to the capture it came from.

    Three failures, kept distinct because each needs a different human
    response:

        400  the code failed its checksum -- re-read the form
        404  the code is valid but matches no capture here
        409  this capture already has a DIFFERENT result on file

    The 404 matters most. A valid code with no capture means the join
    has genuinely failed, and inserting the label anyway would put a row
    in the dataset that traces to no image.
    """

    try:
        outcome = _store().record_outcome(
            linkage_code,
            label=label,
            label_source=label_source,
            reported_at=reported_at,
        )
    except UnknownLinkageCode as error:
        raise HTTPException(404, str(error))
    except LinkageError as error:
        raise HTTPException(400, str(error))
    except ValueError as error:
        # Both a bad label_source and a conflicting stored result. The
        # conflict is the one worth a 409: refusing to overwrite is the
        # point, not a validation slip.
        raise HTTPException(409 if "already has outcome" in str(error) else 400, str(error))

    return outcome.to_dict()


@app.get("/plan_c/coverage")
def plan_c_coverage() -> dict[str, Any]:
    """The number Plan C lives or dies on: what fraction of captures ever
    got a label, and how many of those came from a pathologist.

    Captures are cheap and outcomes are not, so a collection can look
    healthy while almost nothing is usable. Milestone 2 of
    `plan_c_dataset_collection_spec.md` is answered from here.
    """

    return _store().linkage_coverage().to_dict()
