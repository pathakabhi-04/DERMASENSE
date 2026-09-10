# CV-8 contract delta — v1.1, for the RAG side

**From:** CV side
**Date:** 2026-09-10
**Applies to:** `docs/cv8_sample_outputs/sample_outputs.json` (regenerated) and `RiskAssessment.to_dict()` in `src/risk/convergence.py`
**Read this before writing the CV-8 JSON parser** (your Step 3 / §26 Tests 1-7).

Your `RAG_EVOLUTION_AND_CV_INTEGRATION_STRATEGY_v2.md` is accurate about
the architecture and I'm not asking for any change to it. But checking
it against what `to_dict()` *actually* emitted turned up four mismatches
between the document's assumptions and the real payload — three of them
bugs on my side. I've fixed those and regenerated the fixtures rather
than let you build a parser against them.

Net effect on you: **re-pull the fixtures, and read §2 before writing
any confidence-narration code.** Nothing about your architecture,
phase plan, or generation approach changes.

---

## 1. `contract_version` now exists — your §16.1 ask

Every payload now carries a top-level `"contract_version": "1.1"`.

Your §16.1 recommended this and I'd already committed to it in
`docs/build_on_baseline_1.md` Section A; it's in. Suggested handling,
matching your own fail-loud rule:

- **Unrecognized MAJOR** (a `2.x` you haven't been told about) → fail
  loudly. Don't build a partially-parsed `CVAssessmentContext` from
  guessed defaults.
- **Higher MINOR** → safe to proceed. Minor bumps are additive.

This replaces inferring a breaking change from a missing key, which was
the only option before.

---

## 2. `risk_reason` was quoting the WRONG confidence number — please re-read

**This is the one that would have burned you.** Your §11.1 is correct
and I've implemented to it, but it had a gap I only found by reading
the real strings.

Your §11.1 rule: a narrated confidence figure must be
`uncertainty.confidence` (calibrated), never
`diagnosis.probabilities[native_class]` (raw softmax). Correct.

The gap: §11.1 polices the **field**. But your §12 also hands
`risk_reason` to the LLM as evidence to translate into user-facing
language — and `risk_reason` had the **raw** number baked into its
text. So a spec-compliant implementation would still have narrated the
uncalibrated figure, while believing it was showing the calibrated one.
No check on your side could have caught it, because the string is
opaque prose to you.

What the five real examples looked like before and after:

| Example | Old `risk_reason` said | Calibrated (correct) | Gap |
|---|---|---|---|
| BCC, first visit | 46% | **44%** | 1.8 pts |
| ACK, stable | 90% | **84%** | 6.3 pts |
| NEV, changed colour | 81% | **73%** | 7.7 pts |
| ACK, degenerate mask | 44% | **41%** | 3.1 pts |
| ACK, low blur | 57% | **52%** | 4.9 pts |

**Fixed:** `risk_reason` now quotes `uncertainty.confidence`, so the
string and the field always agree. You can narrate either safely.

**Still the safest rule on your side:** take the number from
`uncertainty.confidence`, not by regexing it out of the prose. The
field is the contract; the string is a convenience.

### Bonus: there are THREE confidence-shaped numbers, not two

Your §11.1 names two. There's a third, and my own README had it
mislabelled until today:

| Field | What it is | Narrate? |
|---|---|---|
| `uncertainty.confidence` | CV-6's calibrated confidence | **Yes — this one** |
| `diagnosis.probabilities[native_class]` | raw classifier softmax | No |
| `temporal.confidence` | **coverage** — what fraction of CV-7's 3 feature channels were measurable | No |

`temporal.confidence` is the trap: it looks like a certainty but it's a
denominator. Example 2 has `0.667` purely because 2 of 3 channels
(border + colour; no ruler, so no size) were available. It says nothing
about how sure the verdict is. Don't narrate it as confidence in the
temporal finding.

---

## 3. `magnitude` no longer appears as a number in `risk_reason`

Your §13.1 rule — never narrate `magnitude` as a physical quantity — is
right, and `risk_reason` was violating it:

```
old: "NEV -> MONITOR (81% confidence); escalated to MEDIUM due to
      CHANGED_COLOR (magnitude 1.37)"
new: "NEV -> MONITOR (73% confidence); escalated to MEDIUM due to
      CHANGED_COLOR exceeding its flagging threshold"
```

The raw value is still on `temporal.magnitude` if you want it
internally.

---

## 4. NEW — `per_feature_deltas` is mixed-unit, and your §13.1 doesn't cover it

Your §13.1 rules on `magnitude` only. It doesn't say what the three
per-feature deltas are, and they are **not** on a common scale, **not**
comparable to each other, and **not** threshold-normalized the way
`magnitude` is:

| Key | Unit | Threshold that matters |
|---|---|---|
| `size` | real **millimetres** of diameter change | 20% change, plus an absolute floor |
| `border` | unitless compactness difference | 3.0 |
| `color` | CIE Lab ΔE distance | 24.0 |

**The consequence, and why this matters more than it looks:** a nonzero
delta does **not** mean a change was detected. Example 2 is `STABLE`
with `color: 20.5` — below the 24.0 threshold. The thresholds live in
`src/temporal/delta.py` and are deliberately *not* part of the
contract, so holding only the payload you **cannot** tell `20.5` from
`32.9` in meaning.

**Suggested rule for your side, extending your own §13.1:** treat all
three deltas exactly as you treat `magnitude` — disclosure/audit values,
never patient-facing, never narrated numerically. Let `verdict` carry
the meaning. If you want a qualitative phrase, derive it from `verdict`,
not from the delta values.

If you *do* end up wanting per-feature narration, tell me and I'll ship
either the thresholds or a per-feature `crossed_threshold` boolean in a
1.2 — I'd rather add that than have you infer it. I haven't built it
speculatively.

---

## 5. NEW — `compared_timestamps` is opaque passthrough, not timestamps

Your §11 types it `["string", "string"]` and your Phase 3 plans to
narrate timestamps. Careful:

`TemporalPipeline.assess_pair` takes these as arbitrary caller-supplied
strings and **never parses them**. CV-7 doesn't derive them and doesn't
require them to be dates. In the delivered fixtures they are image
filenames, because the source dataset records visit **order**, not visit
**dates** — a fabricated date would have been worse.

- **Don't parse them as dates. Don't compute an interval from them.**
- Both entries are **independently nullable** (example 1 is `[null, null]`).

The field is correctly designed as passthrough; it's the *name* that
oversells it. When a real deployment supplies real ISO timestamps,
they'll flow through unchanged — but you can't assume that today.

---

## 6. Confirming your §14.1 — you were right, and here's the check

Your §14.1 (all three `per_feature_deltas` independently nullable, not
just `size`) is correct and matches the fixtures exactly:

- Examples 1 and 4: all three `null` together, both with
  `verdict == NO_PRIOR_DATA`.
- Examples 2, 3, 5: `size` `null`, `border`/`color` populated.

Your §17 (use `verdict` AND `quality_flags` together, never `verdict`
alone) is also confirmed: example 1 carries `NO_TEMPORAL_COMPARISON`
(no prior image ever existed), example 4 carries `TEMPORAL_NO_PRIOR_DATA`
+ `DEGENERATE_MASK` (a real prior image was supplied, CV-3 found no mask
in the current photo). Only example 4 justifies saying a comparison was
attempted. Both produce identical `temporal` blocks.

---

## 7. Small one: the fixtures file wraps the payload

`sample_outputs.json` is a list of 5 **wrapper** objects:

```json
{ "description": "...", "source": "...", "payload": { ...the contract... } }
```

Your §9 and §26 describe it as "5 real `to_dict()` outputs", which reads
as a bare list. Parse `entry["payload"]`. Twenty-minute bug, not a
design issue — flagging so it costs you zero minutes.

---

## 8. `quality_flags` — don't hard-code the list

Your §16 rule (accept arbitrary strings, preserve unknown flags, never
fail on a new one) is right and unchanged. One addition: the list in my
README **will** go stale. The authoritative source is
`_evidence_quality_flags` plus the `quality_flags.append` calls in
`src/risk/convergence.py`. A stale list costs you nothing as long as you
don't turn it into an allowed-values enum that rejects unknowns.

Structural changes are the opposite case, and your §16.1 has it right:
a missing/renamed top-level key or an unrecognized `risk_category` /
`verdict` value should fail loudly rather than parse into defaults.

---

## What I have NOT changed, deliberately

- **The 6-class vs 8-class taxonomy** (`ACK BCC MEL NEV SCC SEK` vs
  `MEL BCC SCC AK NV BKL DF VASC`). Still open, still jointly owned,
  still not guessed at. Build against the 6-class set that's actually
  in the payload, per your §18.
- **`ProductAction` vocabulary in `risk_reason`.** The tokens
  `URGENT_EVALUATION` / `EVALUATE_SOON` / `MONITOR` / `UNKNOWN` are
  internal and appear in every reason string, but aren't in your §11
  schema. I kept them rather than restructure the contract before you've
  written the parser. Mapping to `risk_category`, if you need it:
  `URGENT_EVALUATION -> HIGH`, `EVALUATE_SOON -> MEDIUM`,
  `MONITOR -> LOW`, `UNKNOWN -> HIGH` (fail-safe). Source of truth is
  `src/risk/action_mapping.py`. If you'd rather narrate from structured
  fields than parse prose, say so and I'll add them in a 1.2.
- **No live delivery mechanism.** Your §9 is right that this doesn't
  block you: keep the parser taking a dict, agnostic to transport. The
  one thing gating my sync-vs-async decision is a latency measurement
  on my side, not a decision on yours.

---

## What I'd ask from you

1. **Re-pull `docs/cv8_sample_outputs/`** (both files) before writing
   the parser. The old fixtures encode the wrong confidence.
2. **Answer one question when convenient:** do you want per-feature
   narration (§4)? If yes I'll ship thresholds or crossed-threshold
   booleans in a 1.2. If no, the "never narrate deltas numerically" rule
   above is sufficient and I'll build nothing.
3. **Your §26 Test 6** (fail loudly on a missing key / unrecognized
   enum) is the one that protects us both — worth not skipping. With
   `contract_version` in place it's now cheap to write.

Nothing here blocks your Phase 1 gate (the 16-query, 100%-pass run).
That's independent of the CV side — please don't wait on me for it.
