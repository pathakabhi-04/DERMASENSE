# CV-8 Sample Outputs — for the RAG collaborator

**What this is:** `sample_outputs.json` in this directory contains 5
real outputs of the CV pipeline's final step (`RiskAssessment.to_dict()`
in `src/risk/convergence.py`), run on real checkpoints and real images.
Nothing here is hand-written or edited after the run — regenerate it
any time with `python -m scripts.generate_cv8_sample_outputs`.

This is the exact object your ingestion code should parse. It is
produced per detected lesion candidate, not per image (an image can
have zero, one, or more of these if it contains multiple lesions).

**File shape note:** the JSON is a list of 5 *wrapper* objects, each
`{description, source, payload}`. The contract object is `payload` —
parse `entry["payload"]`, not the top-level entry.

## Schema

```json
{
  "contract_version": "1.1",
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
    "per_feature_deltas": {
      "size": 0.0,
      "border": 0.0,
      "color": 0.0
    },
    "compared_timestamps": ["string|null", "string|null"]
  },
  "uncertainty": {
    "confidence": 0.0,
    "requires_review": true
  },
  "quality_flags": ["string", "..."]
}
```

Every one of the three `per_feature_deltas` values is independently
nullable, and both `compared_timestamps` entries are independently
nullable. See the sections below.

## `contract_version` — new in 1.1

Added so your parser can detect a breaking change explicitly instead of
inferring it from a missing key.

- **Fail loudly on an unrecognized MAJOR** (a `2.x` you have not been
  told about). Do not proceed with a partially-parsed context object
  full of guessed defaults.
- **A higher MINOR is safe to proceed on** — minor bumps are additive.

**What changed in 1.1** (if you built against the earlier, unversioned
payloads, re-read these — the values moved):

- `risk_reason` now quotes the **calibrated** confidence, not the raw
  softmax. The number in the string changed for all 5 examples.
- `risk_reason` no longer states `magnitude` numerically.
- `contract_version` itself was added.

Nothing else changed: every probability, risk category, temporal block,
and quality-flag list is byte-identical to the pre-1.1 samples.

## Three confidence-shaped numbers — they are NOT interchangeable

This is the single easiest thing to get wrong, and the earlier version
of this README got it wrong itself. There are three:

| Field | What it is | Show to a user? |
|---|---|---|
| `uncertainty.confidence` | CV-6's **calibrated** confidence (post-hoc temperature scaling) | **Yes — this is the one** |
| `diagnosis.probabilities[native_class]` | the **raw** classifier softmax score | No — internal |
| `temporal.confidence` | what fraction of CV-7's 3 feature channels were measurable | No — internal |

- **`uncertainty.confidence` is the only figure to narrate as "how
  confident is this assessment."** That is what calibration exists to
  produce. In these 5 examples it sits 1.8–7.7 points below the raw softmax.
- `temporal.confidence` is a *coverage* number, not a certainty. In
  example 2 it is `0.667` because 2 of 3 channels (border + color; no
  ruler, so no size) were available. It says nothing about how sure the
  verdict is.

As of 1.1, `risk_reason` also quotes the calibrated figure, so the
string and `uncertainty.confidence` agree. Before 1.1 they did not.

## `temporal.per_feature_deltas` is a MIXED-UNIT dict

The three values are **not** on a common scale and are **not**
comparable to each other:

| Key | Unit | Threshold that matters |
|---|---|---|
| `size` | real **millimetres** of diameter change | 20% change, with an absolute floor |
| `border` | unitless compactness difference | 3.0 |
| `color` | CIE Lab ΔE distance | 24.0 |

Only `temporal.magnitude` is threshold-normalized (`1.0` == exactly at
the threshold that triggers escalation).

**A nonzero delta does NOT mean a change was detected.** Example 2 is
`STABLE` with `color: 20.5` — below the 24.0 threshold. The thresholds
live in `src/temporal/delta.py` and are deliberately not part of this
contract, so a consumer holding only the payload **cannot interpret
these numbers**. Treat all three as disclosure/audit values: none of
them is patient-facing, and neither is `magnitude`. If change magnitude
must appear in an explanation, state it qualitatively ("the color
change was well above the threshold the system uses to flag a
meaningful change") and let `verdict` carry the meaning.

## `temporal.compared_timestamps` is opaque passthrough

These are whatever strings the **caller** handed the pipeline.
`TemporalPipeline.assess_pair` never parses them and CV-7 never derives
them. In this sample set they are image filenames, because the source
dataset records visit **order**, not visit **dates** — using a
fabricated date would have been worse.

**Do not parse them as dates, and do not compute an interval from
them.** Both entries are independently nullable (example 1 is
`[null, null]`).

## Two more things your parser MUST handle

1. **`temporal.per_feature_deltas.size` is very often `null`, not a
   float.** Real-world users essentially never have a physical ruler
   in their photo (this pipeline's only source of a real-world size
   scale), so size-based comparison is rare by design — see example 2
   below, where `border`/`color` are populated but `size` is `null`
   even though a real, valid comparison happened. **A `0.0` here would
   have meant "confirmed no size change" — `null` means "couldn't be
   measured," which is a different and important distinction.** Do
   not coerce `null` to `0`.

   **All three fields are independently nullable, not just `size`.**
   When `verdict` is `NO_PRIOR_DATA`, all three are `null` together
   (examples 1 and 4) — no comparison happened at all. Do not assume
   `border`/`color` are always populated "because they don't need a
   ruler"; that only holds when a real comparison ran.
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
   `NO_PRIOR_DATA`, `quality_flags` contains `NO_TEMPORAL_COMPARISON`,
   all three `per_feature_deltas` are `null`, and
   `compared_timestamps` is `[null, null]`. This is what most
   first-time uploads will look like.
2. **Returning visit, stable.** A real second visit of the same
   lesion; CV-7 ran and found no meaningful change. `temporal.confidence`
   is `0.667` (2 of 3 feature channels — border+color; no ruler, so no
   size channel), not `1.0`. Note `uncertainty.confidence` is a
   different number entirely (`0.836`) — see the confidence table above.
3. **Returning visit, risk escalated by CV-7.** The diagnosis alone
   (`NEV`) would have been `MONITOR`/`LOW`. A real `CHANGED_COLOR`
   verdict pushed `risk_category` to `MEDIUM` and forced
   `requires_review: true` — read `risk_reason` to see this stated
   explicitly ("escalated to MEDIUM due to CHANGED_COLOR exceeding its
   flagging threshold"). **CV-7 can only ever push risk up, never
   down** — a `STABLE` or `NO_PRIOR_DATA` verdict never lowers what
   CV-4 alone would have said.
4. **A prior image was supplied, but the comparison still couldn't
   happen.** `temporal.verdict` is `NO_PRIOR_DATA` even though a real
   second image was given — CV-3 didn't find a lesion mask in the
   current image (`quality_flags` includes `DEGENERATE_MASK`). This is
   a different failure mode from example 1 (there, no prior image
   existed at all) — both currently produce the same `NO_PRIOR_DATA`
   shape, since the contract has no separate slot for the distinction.
   **Use `quality_flags` to tell them apart, never `verdict` alone**:
   example 1 carries `NO_TEMPORAL_COMPARISON`, example 4 carries
   `TEMPORAL_NO_PRIOR_DATA`. Only example 4 justifies saying a
   comparison was attempted.
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

**This list will go stale** — the authoritative source is
`_evidence_quality_flags` plus the `quality_flags.append` calls in
`src/risk/convergence.py`. An unknown flag is never an error, so a
stale list here costs you nothing as long as you don't hard-code it as
an allowed-values enum.

A structurally different payload is a different matter: a **missing or
renamed top-level key**, or an unrecognized `risk_category` /
`verdict` value, should **fail loudly** rather than parse into guessed
defaults. That mirrors the fail-loud discipline the CV side uses
throughout (`NO_PRIOR_DATA` as a first-class outcome rather than a
guess; a ruler calibration that reports `confident=False` rather than
fabricating a scale).

## What is NOT in scope of this file

- No delivery mechanism (API/queue/file drop) is decided yet — this is
  a static example set for you to design against, not a live feed. Your
  parser should take a JSON object/dict and be agnostic to how it
  arrived; that way the eventual transport decision costs you no
  parsing changes. See `docs/build_on_baseline_1.md` Section A.
- `risk_reason` is guaranteed machine-generated, never LLM text. As of
  1.1 every number in it is safe to show, but it is still a terse
  technical string containing internal vocabulary — the
  `URGENT_EVALUATION` / `EVALUATE_SOON` / `MONITOR` / `UNKNOWN` tokens
  come from `ProductAction` in `src/risk/action_mapping.py`, and map to
  `risk_category` as `URGENT_EVALUATION -> HIGH`, `EVALUATE_SOON ->
  MEDIUM`, `MONITOR -> LOW`, `UNKNOWN -> HIGH` (fail-safe). Translate
  it; don't show it verbatim.
