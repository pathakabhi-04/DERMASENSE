# CV-8 Sample Outputs — for the RAG collaborator

**What this is:** `sample_outputs.json` in this directory contains 5
real outputs of the CV pipeline's final step (`RiskAssessment.to_dict()`
in `src/risk/convergence.py`), run on real checkpoints and real images.
Nothing here is hand-written or edited after the run — regenerate it
any time with `python -m scripts.generate_cv8_sample_outputs`.

This is the exact object your ingestion code should parse. It is
produced per detected lesion candidate, not per image (an image can
have zero, one, or more of these if it contains multiple lesions).

## Schema

```json
{
  "lesion_id": "string",
  "diagnosis": {
    "native_class": "string",
    "probabilities": { "<class>": 0.0, "...": 0.0 }
  },
  "risk_category": "LOW | MEDIUM | HIGH",
  "risk_reason": "short machine-generated string, not LLM-authored",
  "temporal": {
    "verdict": "STABLE | GROWING | SHRINKING | CHANGED_COLOR | NO_PRIOR_DATA",
    "magnitude": 0.0,
    "confidence": 0.0,
    "per_feature_deltas": { "size": 0.0, "border": 0.0, "color": 0.0 },
    "compared_timestamps": ["string", "string"]
  },
  "uncertainty": {
    "confidence": 0.0,
    "requires_review": true
  },
  "quality_flags": ["string", "..."]
}
```

This matches the contract locked in
`docs/cv7_temporal_rag_integration_spec.md` exactly, with the field
behaviors below spelled out concretely against the 5 real examples.

## Two things your parser MUST handle, not edge cases you can skip

1. **`temporal.per_feature_deltas.size` is very often `null`, not a
   float.** Real-world users essentially never have a physical ruler
   in their photo (this pipeline's only source of a real-world size
   scale), so size-based comparison is rare by design — see example 2
   below, where `border`/`color` are populated but `size` is `null`
   even though a real, valid comparison happened. **A `0.0` here would
   have meant "confirmed no size change" — `null` means "couldn't be
   measured," which is a different and important distinction.** Do
   not coerce `null` to `0`.
2. **`diagnosis.native_class` is currently the PAD-UFES 6-class
   taxonomy** (`ACK, BCC, MEL, NEV, SCC, SEK`), not the 8-class ISIC
   taxonomy shown in the locked contract's own example
   (`MEL | BCC | SCC | AK | NV | BKL | DF | VASC`). This is a known,
   still-open discrepancy — flagged in
   `docs/cv7_temporal_rag_integration_spec.md`, not silently resolved.
   **Build your parser against the 6-class set actually in these
   samples**, and expect a follow-up conversation before this changes.

## What each example demonstrates (all real, not curated for a story)

1. **First visit, no prior image at all.** `temporal.verdict` is
   `NO_PRIOR_DATA`, `quality_flags` contains `NO_TEMPORAL_COMPARISON`.
   This is what most first-time uploads will look like.
2. **Returning visit, stable.** A real second visit of the same
   lesion; CV-7 ran and found no meaningful change. `uncertainty.confidence`
   is `0.667` (2 of 3 feature channels — border+color; no ruler, so no
   size channel), not `1.0`.
3. **Returning visit, risk escalated by CV-7.** The diagnosis alone
   (`NEV`) would have been `MONITOR`/`LOW`. A real `CHANGED_COLOR`
   verdict pushed `risk_category` to `MEDIUM` and forced
   `requires_review: true` — read `risk_reason` to see this stated
   explicitly ("escalated to MEDIUM due to CHANGED_COLOR"). **CV-7 can
   only ever push risk up, never down** — a `STABLE` or `NO_PRIOR_DATA`
   verdict never lowers what CV-4 alone would have said.
4. **A prior image was supplied, but the comparison still couldn't
   happen.** `temporal.verdict` is `NO_PRIOR_DATA` even though a real
   second image was given — CV-3 didn't find a lesion mask in the
   current image (`quality_flags` includes `DEGENERATE_MASK`). This is
   a different failure mode from example 1 (there, no prior image
   existed at all) — both currently produce the same `NO_PRIOR_DATA`
   shape, since the contract has no separate slot for the distinction.
5. **A disclosed quality flag alongside a normal result.** `LOW_CROP_BLUR`
   is present, but it never changed `risk_category` or
   `requires_review` — `quality_flags` are disclosure only, for you to
   optionally mention in narration, never a hidden gate.

## What `quality_flags` can contain today

`DEGENERATE_MASK`, `MASK_TOUCHES_BORDER`, `LOW_CROP_CONTRAST`,
`LOW_CROP_BLUR`, `ENSEMBLE_DISAGREEMENT`, `NO_TEMPORAL_COMPARISON`,
`TEMPORAL_NO_PRIOR_DATA`, `TEMPORAL_LOW_CONFIDENCE`,
`PRIOR_IMAGE_PAIRING_AMBIGUOUS` (multiple lesions detected in one
image with a prior image supplied — CV-7 can't tell which one it
belongs to, so it's skipped). This list can grow; treat unrecognized
flags as informational, not an error.

## What is NOT in scope of this file

- No delivery mechanism (API/queue/file drop) is decided yet — this is
  a static example set for you to design against, not a live feed.
- `risk_reason` is guaranteed machine-generated, never LLM text — safe
  to quote directly, but it is a short technical string, not meant to
  be shown to an end user verbatim.
