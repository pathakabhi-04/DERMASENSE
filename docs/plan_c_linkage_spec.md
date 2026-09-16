# Plan C linkage and consent: design

**Date:** 2026-09-16
**Status:** identifier design IMPLEMENTED (`src/serving/linkage.py`,
16 tests). Consent **requirements** specified here; consent **wording**
is not, and must not be written by this project's engineering (§6).
**Resolves:** the "cannot be retrofitted" blocker in
`plan_a_ship_checklist.md`.

## 1. Why this had to be settled before shipping

Consent and the linkage identifier cannot be added to photographs that
have already been taken. Ship without them and every capture in the
interim is unusable for Plan C — not merely unlabelled, but unusable,
because no lawful basis exists to retain it for training and no handle
exists to attach a result to.

Everything else in Plan A can be added in version two. This cannot.

## 2. The constraint that shaped the design

**A pathology lab has no API access to this system.** The join runs:

```
user photographs a lesion       (app)
      -> shows the summary to a GP     (paper or screen)
      -> GP refers                     (referral letter)
      -> excision and histopathology   (lab, no integration)
      -> result                        (weeks later)
```

At two points a human transcribes the identifier by hand. That single
fact rules out UUIDs — 36 case-sensitive characters, mis-transcribed
routinely — and rules in a short, checksummed, unambiguous code.

## 3. The identifiers

| identifier | scope | purpose |
|---|---|---|
| `patient_pseudonym` | account | groups a person's lesions so splits can be patient-grouped |
| `lesion_id` | lesion | already in the CV-8 contract |
| `capture_id` | one photo | |
| `linkage_code` | one capture | **the only one ever printed, spoken, or written down** |

`patient_pseudonym` exists because of a specific defect this project
already hit: 25 of 132 DDI-2 test images shared a patient with training
(`cv4b_retrain_result.md`), and DDI and Fitzpatrick carry no patient
identifier at all, so the same leak may exist there undetectably. That
is unfixable after collection. It is designed in here.

Nothing is derived from user-identifying data. A hash of an email or a
phone number is reversible by enumeration, which would make the
"pseudonymous" claim false; all identifiers come from `secrets`.

## 4. Linkage code format

```
DS-XXXX-XXXX-CC
```

**Alphabet — Crockford base32**, `0-9` and `A-Z` minus `I`, `L`, `O`,
`U`. `I`/`L` collide with `1`, `O` with `0`, and dropping `U` keeps the
alphabet from spelling things nobody wants on a medical form. On input,
`i`/`l`→`1` and `o`→`0` are *corrected*, not rejected: a human reading
handwriting should still land on the right record.

**Check characters — two, over a prime modulus (1021).** This was
changed during implementation, and the reason is worth recording. The
first version used a single check character mod 32, and its own test
caught it letting **118 of 2,400 single-character errors through
(~5%)**: at an even-weighted position a delta of 16 is invisible,
because 2×16 ≡ 0 (mod 32). With a prime modulus:

- single-character error → `w·d ≡ 0` forces `d ≡ 0`. **Always caught.**
- transposition → `(wᵢ−wⱼ)(vⱼ−vᵢ) ≡ 0` forces equal weights or equal
  values. **Always caught.**

Both are asserted exhaustively rather than sampled
(`tests/test_linkage.py`). The cost is one extra character, which is a
fair price: the failure this prevents is a melanoma result silently
attached to a different person's photograph.

Capacity is 32⁸ ≈ 1.1×10¹² codes, minted randomly rather than
sequentially so that holding one code tells you nothing about any other.

## 5. Consent, as implemented

`Consent(training_use, granted_at, revoked_at, policy_version)`, and one
rule enforced in code:

> **`build_capture_record()` returns `None` when consent does not permit
> retention.**

Returning `None` rather than a redacted record is deliberate — there is
no such thing as a partially-retained capture, and a caller that forgets
to check gets nothing to write rather than something.

**Training use is separable from using the app.** Consent that permits
care but not training is the common and expensive mistake; consent that
*bundles* them is not freely given. A user who declines gets the entire
product — capture assistance, tracking, the clinician page — and their
capture is simply never retained.

**Revocation** sets `revoked_at`, after which no new records are written
and stored records are deletable.

> **Revocation cannot un-train a model.** A capture already used in
> training has influenced weights that cannot be selectively unlearned.
> Consent copy must say so plainly rather than implying a deletion that
> is not achievable. Flagged here because the honest version is less
> comfortable than the usual wording.

## 6. Where engineering stops

This document specifies what consent must **cover**. It does not, and
must not, supply the wording:

- ethics/IRB approval covering **model training**, not merely service provision
- retention and secondary use, separable and revocable
- linkage to a clinical outcome, and who holds the join
- whether the dataset may ever be released — which has to be in the
  consent from day one, since it cannot be added retroactively either
- the honest limit on revocation (above)

**That wording is a legal and ethics task.** Nothing in this repo should
be treated as sufficient consent text, and `policy_version` exists on
`Consent` precisely so the approved wording a user actually agreed to is
recorded per capture.

## 7. Outcome records

`OutcomeRecord(linkage_code, label, label_source, reported_at)`, where
`label_source` is `biopsy` or `consensus` and is **required**.

The asymmetry is permanent and must not be averaged away: melanoma is
excised and reported by a pathologist, while nobody biopsies an obvious
seborrhoeic keratosis to label a dataset. A single undifferentiated
"ground truth" column is how a dataset quietly becomes untrustworthy.
The constructor also validates the linkage code, so a mis-typed join
fails at entry rather than silently matching nothing.

## 8. Still outstanding

Implemented here: identifiers, consent gating, record schemas, checksum
guarantees.

Not implemented, and not this module's job — the storage and UI layer
that calls it:

- persisting `CaptureRecord` and the image itself
- the account model holding `patient_pseudonym` and consent state
- printing the linkage code on the clinician page
- the revocation path that deletes stored records
- body-site tap and self-reported Fitzpatrick type at capture

Milestone 2 of `plan_c_dataset_collection_spec.md` — "linkage
demonstrably works end-to-end on 10 cases" — is the gate that proves
this design survives contact with an actual clinic, and it should be
run early, while changing the format is still cheap.
