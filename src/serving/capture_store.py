"""
Plan C persistence: where a capture lives between the photo and the biopsy.

`linkage.py` mints the identifiers. This module is the thing that
actually keeps them, and it exists to close the last engineering blocker
in `plan_a_ship_checklist.md`: an account model holding the pseudonym
and consent state, captures that survive to be joined weeks later, and a
revocation path that really deletes.

SQLite, stdlib only. Not because the pilot will stay small forever, but
because the schema is the part that has to be right now -- consent
snapshots and patient grouping cannot be retrofitted onto photographs
already taken -- while the database it sits in can be swapped later
without touching a single capture's meaning.

## The four rules this module enforces

**1. Consent is snapshotted at capture, never referenced.**
`consent_events` is append-only and each `CaptureRecord` embeds the
consent as it stood when the shutter fired. If wording v2 is published
tomorrow, a capture taken under v1 must still say v1 -- otherwise the
project cannot answer "what exactly did this person agree to?", which is
the only question an ethics committee will actually ask.

**2. Nothing is written without consent, including the image.**
The consent check runs BEFORE the photo is written to disk, not after.
`build_capture_record()` returning None means there is no record AND no
file -- a declined capture must not leave a JPEG on the volume for
someone to find later and assume was fair game.

**3. A label that joins to nothing is an error, loudly.**
`record_outcome()` raises `UnknownLinkageCode` when a well-formed code
matches no capture. The checksum catches mis-typing; this catches the
other failure -- a valid code for a capture that was never stored, or
was revoked. Silently inserting an orphan label is how a dataset
acquires rows nobody can trace to an image.

**4. Export carries the patient pseudonym, always.**
25 of 132 DDI-2 test images shared a patient with training
(`cv4b_retrain_result.md`) and that is unfixable after the fact. Every
exported row carries `patient_pseudonym` so a split can be grouped, and
`export_for_training()` has no mode that omits it.

## What revocation does and does not do

Deletes: captures, images, and the outcome rows that belonged to them --
an outcome without its photograph is not clinical data, it is a label
for nothing.

Keeps: the account row and the consent event history, INCLUDING the
revocation event. Those hold a random pseudonym and a timestamp, no
clinical content, and deleting the record that revocation happened would
destroy the only proof it was honoured.

Cannot do: un-train a model. A capture already used in training has
influenced weights that cannot be selectively unlearned
(`plan_c_linkage_spec.md` §5). `revoke()` reports what it deleted and
`RevocationReceipt.caveat` states that limit, so no caller can present
revocation as more complete than it is.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.serving.linkage import (
    CaptureRecord,
    Consent,
    LinkageError,
    OutcomeRecord,
    build_capture_record,
    new_patient_pseudonym,
    normalise_linkage_code,
)

SCHEMA_VERSION = 1

REVOCATION_CAVEAT = (
    "Stored captures and images are deleted. Captures already used to train "
    "a model cannot be removed from that model's weights."
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    patient_pseudonym TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL
);

-- Append-only. Rows are never updated or deleted, including on
-- revocation: the revocation event is itself the proof it happened.
CREATE TABLE IF NOT EXISTS consent_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_pseudonym TEXT NOT NULL REFERENCES accounts(patient_pseudonym),
    training_use      INTEGER NOT NULL,
    policy_version    TEXT NOT NULL,
    granted_at        TEXT,
    revoked_at        TEXT,
    recorded_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS captures (
    capture_id        TEXT PRIMARY KEY,
    linkage_code      TEXT NOT NULL UNIQUE,
    patient_pseudonym TEXT NOT NULL,
    lesion_id         TEXT,
    captured_at       TEXT NOT NULL,
    record_json       TEXT NOT NULL,
    image_path        TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    linkage_code      TEXT PRIMARY KEY,
    label             TEXT NOT NULL,
    label_source      TEXT NOT NULL,
    reported_at       TEXT NOT NULL,
    recorded_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_captures_patient
    ON captures(patient_pseudonym);
CREATE INDEX IF NOT EXISTS idx_consent_patient
    ON consent_events(patient_pseudonym, id);
"""


class UnknownLinkageCode(LinkageError):
    """A well-formed code that matches no stored capture.

    Distinct from a checksum failure on purpose. A checksum failure means
    somebody mis-read the form and should look again. THIS means the code
    was read correctly and the capture is not here -- never stored,
    already revoked, or entered against the wrong system. Those need
    different human responses, so they are different exceptions.
    """


class ConsentRequired(RuntimeError):
    """An operation needed consent state for an account that has none."""


@dataclass(frozen=True)
class RevocationReceipt:
    """What a revocation actually removed, and what it could not."""

    patient_pseudonym: str
    captures_deleted: int
    images_deleted: int
    outcomes_deleted: int
    revoked_at: str
    caveat: str = REVOCATION_CAVEAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_pseudonym": self.patient_pseudonym,
            "captures_deleted": self.captures_deleted,
            "images_deleted": self.images_deleted,
            "outcomes_deleted": self.outcomes_deleted,
            "revoked_at": self.revoked_at,
            "caveat": self.caveat,
        }


@dataclass(frozen=True)
class LinkageCoverage:
    """How much of the collection actually became usable training data.

    This is the number Plan C lives or dies on, and it is the one most
    easily left unmeasured: captures are cheap and outcomes are not, so a
    collection can look healthy while almost nothing is labelled. Having
    it in code from day one means milestone 2 of the collection spec can
    be answered with a query instead of an estimate.
    """

    captures: int
    outcomes: int
    biopsy_confirmed: int

    @property
    def coverage(self) -> float:
        return self.outcomes / self.captures if self.captures else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "captures": self.captures,
            "outcomes": self.outcomes,
            "biopsy_confirmed": self.biopsy_confirmed,
            "coverage": round(self.coverage, 4),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_T = TypeVar("_T")


def _synchronised(method: Callable[..., _T]) -> Callable[..., _T]:
    """Serialise store access, and allow it across threads.

    Not premature: FastAPI runs synchronous endpoint handlers in a
    threadpool, so every request can touch this connection from a
    different thread. SQLite refuses that by default, and the first
    version of this module hit exactly that error -- a failure that
    would have appeared in production, not just under test. Opening with
    `check_same_thread=False` lifts the refusal; this lock is what makes
    lifting it safe, since a capture write is a row plus a file and must
    not interleave with a revocation deleting both.
    """

    @functools.wraps(method)
    def wrapper(self: "CaptureStore", *args: Any, **kwargs: Any) -> _T:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class CaptureStore:
    """Durable home for accounts, consent, captures and outcomes.

    Open one per process and reuse it; SQLite handles the concurrency a
    pilot will see, and the point of failure worth engineering against
    here is a lost join, not write throughput.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            self.root / "captures.db", check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        # Referential integrity is not decoration here: an orphaned
        # consent row would mean a capture whose permission cannot be
        # traced to an account.
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "CaptureStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- accounts and consent ------------------------------------------

    @_synchronised
    def create_account(self) -> str:
        """Mint an account. The pseudonym is the only account identity
        this module ever holds -- no email, no phone, nothing derivable
        (see `linkage.py` on why a hash would not be pseudonymous)."""
        pseudonym = new_patient_pseudonym()
        self._db.execute(
            "INSERT INTO accounts (patient_pseudonym, created_at) VALUES (?, ?)",
            (pseudonym, _utc_now()),
        )
        self._db.commit()
        return pseudonym

    @_synchronised
    def record_consent(
        self,
        patient_pseudonym: str,
        *,
        training_use: bool,
        policy_version: str,
        granted_at: str | None = None,
    ) -> Consent:
        """Append a consent decision.

        `policy_version` is REQUIRED and has no default. The wording is a
        legal/ethics deliverable (`plan_c_linkage_spec.md` §6), and a
        default here would let the system record agreement to a document
        that does not exist yet.
        """
        if not policy_version or policy_version == "unset":
            raise ValueError(
                "policy_version must name the approved consent wording the user "
                "actually agreed to; 'unset' is the placeholder, not a value."
            )
        self._require_account(patient_pseudonym)
        consent = Consent(
            training_use=training_use,
            granted_at=(granted_at or _utc_now()) if training_use else None,
            policy_version=policy_version,
        )
        self._append_consent(patient_pseudonym, consent)
        return consent

    @_synchronised
    def current_consent(self, patient_pseudonym: str) -> Consent | None:
        row = self._db.execute(
            "SELECT * FROM consent_events WHERE patient_pseudonym = ? "
            "ORDER BY id DESC LIMIT 1",
            (patient_pseudonym,),
        ).fetchone()
        if row is None:
            return None
        return Consent(
            training_use=bool(row["training_use"]),
            granted_at=row["granted_at"],
            revoked_at=row["revoked_at"],
            policy_version=row["policy_version"],
        )

    # -- captures ------------------------------------------------------

    @_synchronised
    def save_capture(
        self,
        patient_pseudonym: str,
        *,
        image_bytes: bytes | None = None,
        image_suffix: str = ".jpg",
        lesion_id: str | None = None,
        assessment: dict[str, Any] | None = None,
        body_site: str | None = None,
        fitzpatrick_self_reported: int | None = None,
        device: dict[str, str] | None = None,
    ) -> CaptureRecord | None:
        """Persist one capture, or return None when consent forbids it.

        Consent is read from the store rather than passed in, so a caller
        cannot hand over a stale or optimistic `Consent` object and have
        it believed. The record and the image are written in one
        transaction; the image is only written once the record has been
        built, so a declined capture leaves nothing behind at all.
        """
        consent = self.current_consent(patient_pseudonym)
        if consent is None:
            raise ConsentRequired(
                f"no consent decision recorded for {patient_pseudonym}; "
                "record one before capturing."
            )

        record = build_capture_record(
            patient_pseudonym=patient_pseudonym,
            consent=consent,
            lesion_id=lesion_id,
            assessment=assessment,
            body_site=body_site,
            fitzpatrick_self_reported=fitzpatrick_self_reported,
            device=device,
        )
        if record is None:
            return None

        image_path: Path | None = None
        if image_bytes:
            image_path = self.images_dir / f"{record.capture_id}{image_suffix}"
            image_path.write_bytes(image_bytes)

        try:
            self._db.execute(
                "INSERT INTO captures (capture_id, linkage_code, patient_pseudonym, "
                "lesion_id, captured_at, record_json, image_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.capture_id,
                    record.linkage_code,
                    record.patient_pseudonym,
                    record.lesion_id,
                    record.captured_at,
                    json.dumps(record.to_dict()),
                    str(image_path) if image_path else None,
                ),
            )
            self._db.commit()
        except Exception:
            # Never leave a photograph on disk with no record saying
            # whose it is or what permission covers it.
            if image_path is not None:
                image_path.unlink(missing_ok=True)
            raise

        return record

    @_synchronised
    def get_capture(self, linkage_code: str) -> CaptureRecord | None:
        row = self._capture_row(linkage_code)
        if row is None:
            return None
        payload = json.loads(row["record_json"])
        # schema_version is passed through rather than dropped: a record
        # written under an older schema must read back saying so.
        consent = payload.pop("consent")
        return CaptureRecord(consent=Consent(**consent), **payload)

    # -- outcomes ------------------------------------------------------

    @_synchronised
    def record_outcome(
        self,
        linkage_code: str,
        *,
        label: str,
        label_source: str,
        reported_at: str,
    ) -> OutcomeRecord:
        """Attach a clinical result to the capture it belongs to.

        Two different failures, deliberately distinguished:
        `LinkageError` means the code was mis-transcribed, and the
        clerk should re-read it. `UnknownLinkageCode` means it was read
        correctly and there is no such capture -- a real join failure
        that needs a human, not a retry.

        Re-entering the same result is fine; entering a DIFFERENT result
        for the same capture raises rather than overwriting. A silently
        replaced label is a dataset that changes underneath a training
        run with no record of why.
        """
        outcome = OutcomeRecord(
            linkage_code=linkage_code,
            label=label,
            label_source=label_source,
            reported_at=reported_at,
        )
        code = normalise_linkage_code(linkage_code)

        if self._capture_row(code) is None:
            raise UnknownLinkageCode(
                f"{code} is a valid code but matches no stored capture. It was "
                "either never captured here, or the capture was revoked and "
                "deleted. Do not re-enter it against a different code."
            )

        existing = self._db.execute(
            "SELECT label, label_source FROM outcomes WHERE linkage_code = ?",
            (code,),
        ).fetchone()
        if existing is not None:
            if (existing["label"], existing["label_source"]) != (label, label_source):
                raise ValueError(
                    f"{code} already has outcome {existing['label']!r} from "
                    f"{existing['label_source']!r}; refusing to overwrite it with "
                    f"{label!r} from {label_source!r}. Resolve the disagreement "
                    "before changing a stored label."
                )
            return outcome

        self._db.execute(
            "INSERT INTO outcomes (linkage_code, label, label_source, reported_at, "
            "recorded_at) VALUES (?, ?, ?, ?, ?)",
            (code, label, label_source, reported_at, _utc_now()),
        )
        self._db.commit()
        return outcome

    # -- revocation ----------------------------------------------------

    @_synchronised
    def revoke(self, patient_pseudonym: str) -> RevocationReceipt:
        """Withdraw consent and delete everything retained under it.

        Idempotent: revoking twice deletes nothing the second time and
        still returns a receipt, because a user pressing the button again
        should get confirmation rather than an error.
        """
        self._require_account(patient_pseudonym)
        revoked_at = _utc_now()

        rows = self._db.execute(
            "SELECT linkage_code, image_path FROM captures WHERE patient_pseudonym = ?",
            (patient_pseudonym,),
        ).fetchall()

        images_deleted = 0
        for row in rows:
            if row["image_path"]:
                path = Path(row["image_path"])
                if path.exists():
                    path.unlink()
                    images_deleted += 1

        codes = [row["linkage_code"] for row in rows]
        outcomes_deleted = 0
        if codes:
            placeholders = ",".join("?" * len(codes))
            outcomes_deleted = self._db.execute(
                f"DELETE FROM outcomes WHERE linkage_code IN ({placeholders})", codes
            ).rowcount
        self._db.execute(
            "DELETE FROM captures WHERE patient_pseudonym = ?", (patient_pseudonym,)
        )

        previous = self.current_consent(patient_pseudonym)
        self._append_consent(
            patient_pseudonym,
            Consent(
                training_use=False,
                granted_at=previous.granted_at if previous else None,
                revoked_at=revoked_at,
                policy_version=previous.policy_version if previous else "unset",
            ),
        )

        return RevocationReceipt(
            patient_pseudonym=patient_pseudonym,
            captures_deleted=len(rows),
            images_deleted=images_deleted,
            outcomes_deleted=max(outcomes_deleted, 0),
            revoked_at=revoked_at,
        )

    # -- export --------------------------------------------------------

    @_synchronised
    def export_for_training(self) -> list[dict[str, Any]]:
        """Labelled captures, each carrying its grouping key.

        There is no variant of this that omits `patient_pseudonym`. The
        leak it prevents (the same person in train and test) is invisible
        in every metric it corrupts, so the grouping key travels with the
        data rather than being reconstructed by whoever builds the split.
        """
        rows = self._db.execute(
            "SELECT c.linkage_code, c.capture_id, c.patient_pseudonym, c.lesion_id, "
            "c.captured_at, c.record_json, c.image_path, o.label, o.label_source, "
            "o.reported_at FROM captures c JOIN outcomes o "
            "ON o.linkage_code = c.linkage_code ORDER BY c.captured_at"
        ).fetchall()

        exported = []
        for row in rows:
            record = json.loads(row["record_json"])
            exported.append(
                {
                    "capture_id": row["capture_id"],
                    "linkage_code": row["linkage_code"],
                    "patient_pseudonym": row["patient_pseudonym"],
                    "group_key": row["patient_pseudonym"],
                    "lesion_id": row["lesion_id"],
                    "captured_at": row["captured_at"],
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "label_source": row["label_source"],
                    "reported_at": row["reported_at"],
                    "body_site": record.get("body_site"),
                    "fitzpatrick_self_reported": record.get("fitzpatrick_self_reported"),
                    "device": record.get("device", {}),
                    "assessment": record.get("assessment"),
                    "consent_policy_version": record["consent"]["policy_version"],
                }
            )
        return exported

    @_synchronised
    def linkage_coverage(self) -> LinkageCoverage:
        captures = self._db.execute("SELECT COUNT(*) AS n FROM captures").fetchone()["n"]
        outcomes = self._db.execute(
            "SELECT COUNT(*) AS n FROM outcomes o JOIN captures c "
            "ON c.linkage_code = o.linkage_code"
        ).fetchone()["n"]
        biopsy = self._db.execute(
            "SELECT COUNT(*) AS n FROM outcomes o JOIN captures c "
            "ON c.linkage_code = o.linkage_code WHERE o.label_source = 'biopsy'"
        ).fetchone()["n"]
        return LinkageCoverage(
            captures=captures, outcomes=outcomes, biopsy_confirmed=biopsy
        )

    @_synchronised
    def captures_for(self, patient_pseudonym: str) -> list[str]:
        """A list, not a generator: a lazily-consumed cursor would hold
        rows open outside the lock the decorator acquires."""
        return [
            row["linkage_code"]
            for row in self._db.execute(
                "SELECT linkage_code FROM captures WHERE patient_pseudonym = ? "
                "ORDER BY captured_at",
                (patient_pseudonym,),
            )
        ]

    # -- internals -----------------------------------------------------

    def _require_account(self, patient_pseudonym: str) -> None:
        row = self._db.execute(
            "SELECT 1 FROM accounts WHERE patient_pseudonym = ?", (patient_pseudonym,)
        ).fetchone()
        if row is None:
            raise ConsentRequired(f"no such account: {patient_pseudonym}")

    def _append_consent(self, patient_pseudonym: str, consent: Consent) -> None:
        self._db.execute(
            "INSERT INTO consent_events (patient_pseudonym, training_use, "
            "policy_version, granted_at, revoked_at, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                patient_pseudonym,
                int(consent.training_use),
                consent.policy_version,
                consent.granted_at,
                consent.revoked_at,
                _utc_now(),
            ),
        )
        self._db.commit()

    def _capture_row(self, linkage_code: str) -> sqlite3.Row | None:
        code = normalise_linkage_code(linkage_code)
        return self._db.execute(
            "SELECT * FROM captures WHERE linkage_code = ?", (code,)
        ).fetchone()
