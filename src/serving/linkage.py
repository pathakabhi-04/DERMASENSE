"""
Plan C linkage: joining a photo taken today to a biopsy result weeks later.

`plan_c_dataset_collection_spec.md` §1 makes Plan A the collection
instrument -- user photographs a lesion, shows the summary to a GP, the
GP refers, a pathologist reports, and that label belongs to the ORIGINAL
user-captured photo. This module mints the identifier that survives that
journey.

## The constraint that shapes everything here

**A pathology lab has no API access to this system.** The join has to
survive a human writing a code on a requisition form, or a patient
reading it aloud. That rules out UUIDs (36 characters of case-sensitive
hex-and-dashes, transcribed wrong roughly every time) and rules in a
short, checksummed, unambiguous code.

So: Crockford base32 -- `0-9` plus `A-Z` minus `I`, `L`, `O`, `U`. `I`
and `L` collide with `1`, `O` collides with `0`, and dropping `U` is
what keeps the alphabet from spelling things nobody wants printed on a
medical form. Input is normalised on the way in, so `i`/`l` typed for
`1` and `o` for `0` are corrected rather than rejected.

A check character makes a mis-transcription fail loudly instead of
silently joining a melanoma result to somebody else's mole. That is the
whole reason it is there.

## What is deliberately NOT derived from anything

`patient_pseudonym` and the linkage payload are random, from
`secrets`. Nothing here is a hash of an email, a phone number or a
device id: a hash of a low-entropy identifier is reversible by
enumeration, which would make the "pseudonymous" claim false.

## Identifiers and what each is for

    patient_pseudonym   stable per account. Groups a person's lesions so
                        splits can be patient-grouped -- the defect that
                        put 25 of 132 DDI-2 test images in training
                        (cv4b_retrain_result.md) is unfixable after the
                        fact, so it is designed in here.
    lesion_id           stable per lesion, already in the CV-8 contract.
    capture_id          one photo.
    linkage_code        the human-transcribable handle for one capture.
                        This is the only one that is printed, spoken, or
                        written on a form.

The linkage code maps to the other three server-side. It carries no
information itself -- knowing a code tells you nothing about the person
without the server's record, and codes are random rather than
sequential, so one cannot be guessed from another.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Crockford base32: no I, L, O, U.
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
PREFIX = "DS"
PAYLOAD_LENGTH = 8  # 32^8 ~= 1.1e12 codes
CHECK_LENGTH = 2
# PRIME modulus, and that is the whole point. The first version used
# mod 32 with weights 1..8 and let ~5% of single-character errors
# through: at an even-weighted position a delta of 16 is invisible
# because 2*16 == 32 == 0 (mod 32). With a prime modulus, w*d == 0
# forces d == 0, so every single-character error is caught, and
# (wi-wj)(vj-vi) == 0 forces equal weights or equal values, so every
# transposition is too. 1021 needs two check characters; that is a
# fair price for a guarantee instead of a 95% hit rate.
CHECK_MODULUS = 1021
_AMBIGUOUS = {"I": "1", "L": "1", "O": "0", "U": "V"}

SCHEMA_VERSION = "1.0"


class LinkageError(ValueError):
    """A linkage code failed to parse or failed its check character."""


def _check_characters(payload: str) -> str:
    """Positionally weighted checksum over a PRIME modulus.

    Position weighting catches TRANSPOSITION -- `A7` for `7A` is the
    commonest hand-transcription error and an unweighted sum accepts it.
    The prime modulus is what makes both guarantees exact rather than
    probabilistic; see CHECK_MODULUS.
    """
    total = sum(
        (index + 1) * ALPHABET.index(char) for index, char in enumerate(payload)
    ) % CHECK_MODULUS
    return ALPHABET[total // 32] + ALPHABET[total % 32]


def new_linkage_code() -> str:
    """Mint a code. Random, not sequential: sequential codes would let
    anyone holding one enumerate other people's records."""
    payload = "".join(secrets.choice(ALPHABET) for _ in range(PAYLOAD_LENGTH))
    return f"{PREFIX}-{payload[:4]}-{payload[4:]}-{_check_characters(payload)}"


def normalise_linkage_code(raw: str) -> str:
    """Repair what a human plausibly typed, then validate.

    Case, spacing, missing dashes and the ambiguous-glyph substitutions
    are all recoverable. A bad check character is NOT -- that is the one
    failure that must reach a person rather than be guessed at.
    """
    if not isinstance(raw, str):
        raise LinkageError("linkage code must be a string")

    cleaned = re.sub(r"[^0-9A-Za-z]", "", raw).upper()
    if cleaned.startswith(PREFIX):
        cleaned = cleaned[len(PREFIX):]
    cleaned = "".join(_AMBIGUOUS.get(char, char) for char in cleaned)

    if len(cleaned) != PAYLOAD_LENGTH + CHECK_LENGTH:
        raise LinkageError(
            f"linkage code should have {PAYLOAD_LENGTH + CHECK_LENGTH} characters "
            f"after the {PREFIX} prefix, got {len(cleaned)}"
        )
    payload, check = cleaned[:PAYLOAD_LENGTH], cleaned[PAYLOAD_LENGTH:]
    if any(char not in ALPHABET for char in cleaned):
        raise LinkageError("linkage code contains characters outside the alphabet")
    if _check_characters(payload) != check:
        raise LinkageError(
            "linkage code failed its check character -- it was probably mis-typed. "
            "Re-read it rather than guessing: a wrong code would attach a clinical "
            "result to the wrong person's photo."
        )
    return f"{PREFIX}-{payload[:4]}-{payload[4:]}-{check}"


def is_valid_linkage_code(raw: str) -> bool:
    try:
        normalise_linkage_code(raw)
    except LinkageError:
        return False
    return True


def new_patient_pseudonym() -> str:
    """Random, never derived from user-identifying data -- see the module
    docstring on why a hash of an email would not be pseudonymous."""
    return f"p_{secrets.token_hex(16)}"


def new_capture_id() -> str:
    return f"c_{secrets.token_hex(12)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Consent:
    """Consent state for one account, at the time of capture.

    `training_use` is SEPARABLE from using the app
    (`plan_c_dataset_collection_spec.md` §7): consent that permits care
    but not training is the common and expensive mistake, and consent
    that bundles them is not freely given. A user who declines still
    gets the whole product; their capture is simply never retained for
    Plan C.
    """

    training_use: bool
    granted_at: str | None = None
    revoked_at: str | None = None
    policy_version: str = "unset"

    @property
    def permits_retention(self) -> bool:
        return self.training_use and self.revoked_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_use": self.training_use,
            "granted_at": self.granted_at,
            "revoked_at": self.revoked_at,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class CaptureRecord:
    """One capture, as Plan C will later join it to an outcome.

    The full CV assessment is stored here and NOT returned to the client
    (decision_001): a prediction sitting beside a future biopsy result
    is the evaluation this project has never been able to make.
    """

    linkage_code: str
    capture_id: str
    patient_pseudonym: str
    lesion_id: str | None
    captured_at: str
    consent: Consent
    assessment: dict[str, Any] | None = None
    body_site: str | None = None
    fitzpatrick_self_reported: int | None = None
    device: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "linkage_code": self.linkage_code,
            "capture_id": self.capture_id,
            "patient_pseudonym": self.patient_pseudonym,
            "lesion_id": self.lesion_id,
            "captured_at": self.captured_at,
            "consent": self.consent.to_dict(),
            "body_site": self.body_site,
            "fitzpatrick_self_reported": self.fitzpatrick_self_reported,
            "device": dict(self.device),
            "assessment": self.assessment,
        }


def build_capture_record(
    *,
    patient_pseudonym: str,
    consent: Consent,
    lesion_id: str | None = None,
    assessment: dict[str, Any] | None = None,
    body_site: str | None = None,
    fitzpatrick_self_reported: int | None = None,
    device: dict[str, str] | None = None,
) -> CaptureRecord | None:
    """Build the record Plan C joins against, or None when consent does
    not permit retention.

    Returning None rather than a redacted record is deliberate: there is
    no such thing as a partially-retained capture, and a caller that
    forgets to check gets nothing to write rather than something.
    """
    if not consent.permits_retention:
        return None

    if fitzpatrick_self_reported is not None and not 1 <= fitzpatrick_self_reported <= 6:
        raise ValueError("Fitzpatrick type is I-VI; got " f"{fitzpatrick_self_reported}")

    return CaptureRecord(
        linkage_code=new_linkage_code(),
        capture_id=new_capture_id(),
        patient_pseudonym=patient_pseudonym,
        lesion_id=lesion_id,
        captured_at=_utc_now(),
        consent=consent,
        assessment=assessment,
        body_site=body_site,
        fitzpatrick_self_reported=fitzpatrick_self_reported,
        device=dict(device or {}),
    )


@dataclass(frozen=True)
class OutcomeRecord:
    """A clinical result, arriving weeks after the capture it belongs to.

    `label_source` is kept per-record rather than assumed, because the
    asymmetry is real and permanent: melanoma gets excised and reported
    by a pathologist, while nobody biopsies an obvious seborrhoeic
    keratosis to label a dataset. Averaging those two into one
    "ground truth" column is how a dataset quietly becomes untrustworthy
    (`plan_c_dataset_collection_spec.md` §4).
    """

    linkage_code: str
    label: str
    label_source: str  # "biopsy" | "consensus"
    reported_at: str

    def __post_init__(self) -> None:
        if self.label_source not in ("biopsy", "consensus"):
            raise ValueError(
                f"label_source must be 'biopsy' or 'consensus', got {self.label_source!r}"
            )
        # Fails loudly here rather than joining to nothing later.
        normalise_linkage_code(self.linkage_code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "linkage_code": normalise_linkage_code(self.linkage_code),
            "label": self.label,
            "label_source": self.label_source,
            "reported_at": self.reported_at,
        }
