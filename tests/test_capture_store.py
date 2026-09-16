"""
Plan C persistence.

`test_linkage.py` proves the identifier survives a human writing it on a
form. This proves the record survives the weeks between the photo and
the result -- and that a user who says no, or later changes their mind,
actually gets what they were promised.
"""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.serving.capture_store import (
    CaptureStore,
    ConsentRequired,
    UnknownLinkageCode,
)
from src.serving.linkage import LinkageError, new_linkage_code

POLICY = "consent-v1-2026-09"


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = CaptureStore(Path(self._tmp.name))
        self.addCleanup(self.store.close)

    def consented_account(self, training_use: bool = True) -> str:
        pseudonym = self.store.create_account()
        self.store.record_consent(
            pseudonym, training_use=training_use, policy_version=POLICY
        )
        return pseudonym


class ConsentGate(StoreTestCase):
    def test_capture_without_any_consent_decision_raises(self):
        """Not 'returns None' -- a capture reaching the store with no
        consent on file is a bug in the caller, not a user choice."""
        pseudonym = self.store.create_account()
        with self.assertRaises(ConsentRequired):
            self.store.save_capture(pseudonym, image_bytes=b"jpegbytes")

    def test_declined_consent_writes_no_record_and_no_image(self):
        """The image check is the point: a declined capture must not
        leave a photo on disk for someone to find and assume was fair."""
        pseudonym = self.consented_account(training_use=False)
        self.assertIsNone(self.store.save_capture(pseudonym, image_bytes=b"jpegbytes"))
        self.assertEqual(list(self.store.images_dir.iterdir()), [])
        self.assertEqual(self.store.linkage_coverage().captures, 0)

    def test_consent_is_read_from_the_store_not_trusted_from_the_caller(self):
        """save_capture takes no Consent argument, so a stale or
        optimistic object cannot be handed in and believed."""
        pseudonym = self.consented_account()
        self.store.revoke(pseudonym)
        self.assertIsNone(self.store.save_capture(pseudonym, image_bytes=b"x"))

    def test_policy_version_must_name_real_wording(self):
        """'unset' is the placeholder in Consent; accepting it here would
        record agreement to a document that does not exist yet."""
        pseudonym = self.store.create_account()
        for bad in ("", "unset"):
            with self.assertRaises(ValueError):
                self.store.record_consent(
                    pseudonym, training_use=True, policy_version=bad
                )

    def test_capture_snapshots_the_consent_wording_in_force(self):
        """A capture taken under v1 must still say v1 after v2 ships."""
        pseudonym = self.consented_account()
        record = self.store.save_capture(pseudonym, image_bytes=b"x")
        self.store.record_consent(
            pseudonym, training_use=True, policy_version="consent-v2-2027-01"
        )
        stored = self.store.get_capture(record.linkage_code)
        self.assertEqual(stored.consent.policy_version, POLICY)
        self.assertEqual(
            self.store.current_consent(pseudonym).policy_version, "consent-v2-2027-01"
        )


class Joining(StoreTestCase):
    def test_capture_round_trips_with_its_assessment(self):
        pseudonym = self.consented_account()
        assessment = {"diagnosis": {"native_class": "MEL"}, "risk_category": "HIGH"}
        record = self.store.save_capture(
            pseudonym,
            image_bytes=b"jpegbytes",
            lesion_id="lesion-1",
            assessment=assessment,
            body_site="left forearm",
            fitzpatrick_self_reported=3,
        )
        stored = self.store.get_capture(record.linkage_code)
        self.assertEqual(stored.assessment, assessment)
        self.assertEqual(stored.body_site, "left forearm")
        self.assertEqual(stored.patient_pseudonym, pseudonym)

    def test_hand_typed_code_finds_the_capture(self):
        """The whole journey is a clerk reading a code off a form."""
        pseudonym = self.consented_account()
        record = self.store.save_capture(pseudonym, image_bytes=b"x")
        typed = record.linkage_code.lower().replace("-", " ")
        outcome = self.store.record_outcome(
            typed, label="MEL", label_source="biopsy",
            reported_at="2026-11-01T00:00:00Z",
        )
        self.assertEqual(outcome.to_dict()["linkage_code"], record.linkage_code)

    def test_mistyped_and_unknown_codes_are_different_failures(self):
        """A checksum failure means 're-read the form'. A valid code with
        no capture means 'this is not our record' -- different human
        response, so a different exception."""
        with self.assertRaises(LinkageError) as mistyped:
            self.store.record_outcome(
                "DS-ZZZZ-ZZZZ-00", label="MEL", label_source="biopsy",
                reported_at="2026-11-01T00:00:00Z",
            )
        self.assertNotIsInstance(mistyped.exception, UnknownLinkageCode)

        with self.assertRaises(UnknownLinkageCode):
            self.store.record_outcome(
                new_linkage_code(), label="MEL", label_source="biopsy",
                reported_at="2026-11-01T00:00:00Z",
            )

    def test_reentering_the_same_result_is_fine(self):
        pseudonym = self.consented_account()
        record = self.store.save_capture(pseudonym, image_bytes=b"x")
        for _ in range(2):
            self.store.record_outcome(
                record.linkage_code, label="NEV", label_source="consensus",
                reported_at="2026-11-01T00:00:00Z",
            )
        self.assertEqual(self.store.linkage_coverage().outcomes, 1)

    def test_a_conflicting_result_raises_rather_than_overwriting(self):
        """A silently replaced label changes a dataset underneath a
        training run with no record of why."""
        pseudonym = self.consented_account()
        record = self.store.save_capture(pseudonym, image_bytes=b"x")
        self.store.record_outcome(
            record.linkage_code, label="NEV", label_source="consensus",
            reported_at="2026-11-01T00:00:00Z",
        )
        with self.assertRaises(ValueError):
            self.store.record_outcome(
                record.linkage_code, label="MEL", label_source="biopsy",
                reported_at="2026-12-01T00:00:00Z",
            )


class Revocation(StoreTestCase):
    def test_revocation_deletes_captures_images_and_outcomes(self):
        pseudonym = self.consented_account()
        record = self.store.save_capture(pseudonym, image_bytes=b"jpegbytes")
        self.store.record_outcome(
            record.linkage_code, label="MEL", label_source="biopsy",
            reported_at="2026-11-01T00:00:00Z",
        )

        receipt = self.store.revoke(pseudonym)

        self.assertEqual(receipt.captures_deleted, 1)
        self.assertEqual(receipt.images_deleted, 1)
        self.assertEqual(receipt.outcomes_deleted, 1)
        self.assertEqual(list(self.store.images_dir.iterdir()), [])
        self.assertIsNone(self.store.get_capture(record.linkage_code))
        self.assertEqual(self.store.export_for_training(), [])

    def test_receipt_states_the_limit_it_cannot_honour(self):
        """Revocation cannot un-train a model, and the receipt must not
        let a caller present it as though it could."""
        pseudonym = self.consented_account()
        self.store.save_capture(pseudonym, image_bytes=b"x")
        self.assertIn("weights", self.store.revoke(pseudonym).caveat)

    def test_revocation_keeps_the_proof_it_happened(self):
        """Deleting the revocation event would destroy the only evidence
        the request was honoured."""
        pseudonym = self.consented_account()
        self.store.revoke(pseudonym)
        consent = self.store.current_consent(pseudonym)
        self.assertFalse(consent.training_use)
        self.assertIsNotNone(consent.revoked_at)
        self.assertEqual(consent.policy_version, POLICY)

    def test_revoking_twice_confirms_rather_than_erroring(self):
        pseudonym = self.consented_account()
        self.store.save_capture(pseudonym, image_bytes=b"x")
        self.store.revoke(pseudonym)
        second = self.store.revoke(pseudonym)
        self.assertEqual(second.captures_deleted, 0)

    def test_revocation_touches_nobody_else(self):
        keeper = self.consented_account()
        kept = self.store.save_capture(keeper, image_bytes=b"x")
        leaver = self.consented_account()
        self.store.save_capture(leaver, image_bytes=b"y")

        self.store.revoke(leaver)

        self.assertIsNotNone(self.store.get_capture(kept.linkage_code))
        self.assertEqual(self.store.linkage_coverage().captures, 1)


class Concurrency(StoreTestCase):
    def test_captures_from_many_threads_all_survive(self):
        """FastAPI runs synchronous handlers in a threadpool, so this is
        the ordinary case, not an exotic one. The first version of the
        store raised `SQLite objects created in a thread can only be
        used in that same thread` here -- in production, not just under
        test."""
        pseudonym = self.consented_account()
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(
                pool.map(
                    lambda _: self.store.save_capture(pseudonym, image_bytes=b"x"),
                    range(40),
                )
            )
        self.assertEqual(len({r.linkage_code for r in records}), 40)
        self.assertEqual(self.store.linkage_coverage().captures, 40)
        self.assertEqual(len(list(self.images_dir_files())), 40)

    def images_dir_files(self):
        return self.store.images_dir.iterdir()


class Export(StoreTestCase):
    def test_every_exported_row_carries_its_grouping_key(self):
        """The DDI-2 leak -- same patient in train and test -- is
        invisible in every metric it corrupts and unfixable afterwards."""
        pseudonym = self.consented_account()
        for _ in range(3):
            record = self.store.save_capture(pseudonym, image_bytes=b"x")
            self.store.record_outcome(
                record.linkage_code, label="NEV", label_source="consensus",
                reported_at="2026-11-01T00:00:00Z",
            )
        rows = self.store.export_for_training()
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["group_key"] for row in rows}, {pseudonym})
        for row in rows:
            self.assertEqual(row["consent_policy_version"], POLICY)

    def test_unlabelled_captures_are_not_exported(self):
        pseudonym = self.consented_account()
        self.store.save_capture(pseudonym, image_bytes=b"x")
        self.assertEqual(self.store.export_for_training(), [])

    def test_coverage_separates_biopsy_from_consensus(self):
        """Plan C's viability is the labelled fraction, and biopsy and
        consensus labels are not interchangeable."""
        pseudonym = self.consented_account()
        codes = [
            self.store.save_capture(pseudonym, image_bytes=b"x").linkage_code
            for _ in range(4)
        ]
        self.store.record_outcome(
            codes[0], label="MEL", label_source="biopsy",
            reported_at="2026-11-01T00:00:00Z",
        )
        self.store.record_outcome(
            codes[1], label="NEV", label_source="consensus",
            reported_at="2026-11-01T00:00:00Z",
        )
        coverage = self.store.linkage_coverage()
        self.assertEqual(coverage.captures, 4)
        self.assertEqual(coverage.outcomes, 2)
        self.assertEqual(coverage.biopsy_confirmed, 1)
        self.assertEqual(coverage.coverage, 0.5)

    def test_coverage_is_zero_not_a_crash_on_an_empty_store(self):
        self.assertEqual(self.store.linkage_coverage().coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
