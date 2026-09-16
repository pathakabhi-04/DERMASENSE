"""
Plan C linkage identifiers.

The failure these guard against is specific and permanent: a
mis-transcribed code that silently joins a melanoma result to the wrong
person's photo. Everything here exists because the join passes through a
human writing a code on a form.
"""

from __future__ import annotations

import unittest

from src.serving.linkage import (
    ALPHABET,
    Consent,
    LinkageError,
    OutcomeRecord,
    build_capture_record,
    is_valid_linkage_code,
    new_linkage_code,
    new_patient_pseudonym,
    normalise_linkage_code,
)


class LinkageCodes(unittest.TestCase):
    def test_minted_codes_validate(self):
        for _ in range(200):
            self.assertTrue(is_valid_linkage_code(new_linkage_code()))

    def test_codes_are_unique_and_unguessable(self):
        codes = {new_linkage_code() for _ in range(2000)}
        self.assertEqual(len(codes), 2000)

    def test_alphabet_excludes_ambiguous_glyphs(self):
        for char in ("I", "L", "O", "U"):
            self.assertNotIn(char, ALPHABET)

    def test_human_transcription_is_repaired(self):
        """Case, spacing and missing dashes are recoverable typing noise."""
        code = new_linkage_code()
        payload = code.replace("-", "")[2:]
        for variant in (
            code.lower(),
            code.replace("-", ""),
            code.replace("-", " "),
            f"  {code}  ",
            f"ds {payload[:4]} {payload[4:8]} {payload[8:]}",
        ):
            self.assertEqual(normalise_linkage_code(variant), code, variant)

    def test_ambiguous_glyphs_are_corrected_not_rejected(self):
        """Someone reading '1' as 'I' should still land on the right record."""
        code = next(
            c for c in (new_linkage_code() for _ in range(500)) if "1" in c
        )
        self.assertEqual(normalise_linkage_code(code.replace("1", "I")), code)
        self.assertEqual(normalise_linkage_code(code.replace("1", "l")), code)

    def test_single_character_error_is_caught(self):
        """The whole point: a wrong code must fail, not join to someone else."""
        caught = 0
        attempts = 0
        for _ in range(300):
            code = new_linkage_code()
            payload = list(code.replace("-", "")[2:-2])
            for index, original in enumerate(payload):
                replacement = next(c for c in ALPHABET if c != original)
                broken = payload.copy()
                broken[index] = replacement
                candidate = "".join(broken) + code[-2:]
                attempts += 1
                if not is_valid_linkage_code(
                    f"DS-{candidate[:4]}-{candidate[4:8]}-{candidate[8:]}"
                ):
                    caught += 1
        self.assertEqual(caught, attempts, "a single-character error slipped through")

    def test_transposition_is_caught(self):
        """Swapping two adjacent characters is the commonest hand error,
        and an unweighted checksum would accept it."""
        missed = 0
        checked = 0
        for _ in range(300):
            code = new_linkage_code()
            payload = list(code.replace("-", "")[2:-2])
            for index in range(len(payload) - 1):
                if payload[index] == payload[index + 1]:
                    continue
                swapped = payload.copy()
                swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
                candidate = "".join(swapped) + code[-2:]
                checked += 1
                if is_valid_linkage_code(
                    f"DS-{candidate[:4]}-{candidate[4:8]}-{candidate[8:]}"
                ):
                    missed += 1
        self.assertGreater(checked, 0)
        self.assertEqual(missed, 0, f"{missed}/{checked} transpositions accepted")

    def test_garbage_is_rejected(self):
        for bad in ("", "DS-----", "not a code", "DS-1234-5678-99", "DS-ZZZZ-ZZZZ"):
            self.assertFalse(is_valid_linkage_code(bad), bad)

    def test_pseudonym_is_not_derived_from_anything(self):
        self.assertNotEqual(new_patient_pseudonym(), new_patient_pseudonym())


class ConsentGating(unittest.TestCase):
    def test_no_record_without_consent(self):
        """Declining training use must produce nothing to write, not a
        redacted record a caller might persist anyway."""
        record = build_capture_record(
            patient_pseudonym=new_patient_pseudonym(),
            consent=Consent(training_use=False),
            assessment={"risk_category": "HIGH"},
        )
        self.assertIsNone(record)

    def test_revoked_consent_stops_retention(self):
        record = build_capture_record(
            patient_pseudonym=new_patient_pseudonym(),
            consent=Consent(
                training_use=True, granted_at="2026-01-01T00:00:00Z",
                revoked_at="2026-06-01T00:00:00Z",
            ),
        )
        self.assertIsNone(record)

    def test_consented_record_carries_the_full_assessment(self):
        """Withheld from the client, kept for Plan C (decision_001)."""
        assessment = {"diagnosis": {"native_class": "MEL"}, "risk_category": "HIGH"}
        record = build_capture_record(
            patient_pseudonym="p_abc",
            consent=Consent(training_use=True, granted_at="2026-01-01T00:00:00Z"),
            lesion_id="lesion-1",
            assessment=assessment,
            body_site="left forearm",
            fitzpatrick_self_reported=4,
            device={"model": "Pixel 8"},
        )
        self.assertIsNotNone(record)
        payload = record.to_dict()
        self.assertEqual(payload["assessment"], assessment)
        self.assertEqual(payload["patient_pseudonym"], "p_abc")
        self.assertEqual(payload["fitzpatrick_self_reported"], 4)
        self.assertTrue(is_valid_linkage_code(payload["linkage_code"]))

    def test_impossible_fitzpatrick_type_is_rejected(self):
        for bad in (0, 7, -1):
            with self.assertRaises(ValueError):
                build_capture_record(
                    patient_pseudonym="p_abc",
                    consent=Consent(training_use=True),
                    fitzpatrick_self_reported=bad,
                )


class Outcomes(unittest.TestCase):
    def test_outcome_requires_a_valid_code(self):
        with self.assertRaises(LinkageError):
            OutcomeRecord(
                linkage_code="DS-ZZZZ-ZZZZ-00", label="MEL",
                label_source="biopsy", reported_at="2026-03-01T00:00:00Z",
            )

    def test_label_source_must_be_stated(self):
        """Biopsy and consensus labels are not interchangeable and the
        asymmetry is permanent -- benign lesions are rarely excised."""
        with self.assertRaises(ValueError):
            OutcomeRecord(
                linkage_code=new_linkage_code(), label="NEV",
                label_source="guess", reported_at="2026-03-01T00:00:00Z",
            )

    def test_outcome_normalises_a_hand_typed_code(self):
        code = new_linkage_code()
        outcome = OutcomeRecord(
            linkage_code=code.lower().replace("-", " "), label="MEL",
            label_source="biopsy", reported_at="2026-03-01T00:00:00Z",
        )
        self.assertEqual(outcome.to_dict()["linkage_code"], code)


if __name__ == "__main__":
    unittest.main()
