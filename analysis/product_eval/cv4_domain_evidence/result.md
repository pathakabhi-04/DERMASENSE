# CV-4 Out-of-Domain Behavior — Result

**Status:** Investigation complete, evidence layer implemented. Spec:
`docs/cv4_domain_evidence_spec.md`.

## Diagnosis

The 32.7% URGENT_EVALUATION / 85% review rate on the wide-field branch
(`analysis/product_eval/cv1_cv4_assembly/result.md`) is driven by a
12.9% per-candidate high-risk rate, dominated by BCC (357 of 3,082
candidates).

Two upstream explanations were tested and ruled out: CV-2 detection
confidence does not separate high-risk from non-high-risk predictions
(0.529 vs 0.534 mean), and CV-3 mask evidence does not either (0.237 vs
0.282 area fraction). Overall CV-4 confidence is not dramatically
different in- vs out-of-domain (0.733 vs 0.760 mean) — this is not a
blanket miscalibration.

**The failure is class-specific.** BCC and ACK confidence drops sharply
out of domain (BCC 0.578 vs 0.743 in-domain; ACK 0.590 vs 0.773) while
MEL and NEV stay close to their in-domain levels (0.672 vs 0.809; 0.812
vs 0.874).

**Visual audit (36 crops — top-12 highest-confidence BCC/MEL, plus 12
random BCC and 12 random MEL, all sampled from the exact crop CV-4
received) confirms two structurally different populations.** MEL
predictions are consistently coherent — dark, irregular pigmented blobs
with real internal structure, including in the unbiased random sample.
BCC predictions are frequently incoherent — diffuse low-contrast
pink/red patches with no clear border, hair-dominated crops, one linear
scratch mark.

**A quantifiable signal separates them.** Crop-level contrast
(`src/quality/signals.py::contrast_signal`, applied to the crop rather
than the source image) on a 150 BCC / 39 MEL calibration sample: BCC
mean 0.208 (58.7% below 0.20), MEL mean 0.337 (5.1% below 0.20). ACK
shows the same pattern as BCC (mean 0.165).

This is consistent with, not a duplicate of, the existing SCC/BCC
embedding-overlap finding (`analysis/scc_bcc/`, not fixable by
reweighting/SupCon). That investigation characterized in-domain
confusability geometry; this identifies a previously-untested,
orthogonal lever: low-information crops land disproportionately in that
same weak region out of domain.

## What was deliberately not done

- **No retraining.** The SCC/BCC question is already closed
  (`analysis/scc_bcc/`); reopening it was out of scope, and no labeled
  TBP data exists to retrain against regardless.
- **No filtering.** A low-contrast crop can still be a genuine faint
  lesion. Dropping candidates would recreate the silent-miss failure
  mode the `PipelineOutcome` design exists to prevent.
- **No change to the aggregation rule.** Still most-severe-wins; a
  screening tool must not hide a high-risk candidate behind an averaged
  verdict.

## Fix implemented

`crop_blur` and `crop_contrast` added to `CandidateResult`
(`src/inference/orchestrator.py`), computed on the exact RGB crop CV-4
receives, using CV-1's existing signal functions
(`src/quality/signals.py`) — no new signal design, no new inference
pass. Recorded for **every** candidate regardless of predicted class,
matching the CV-3-mask precedent (evidence alongside the diagnosis,
never gating it) and deliberately not hardcoded to BCC — a disclosure
mechanism a reviewer or a future CV-6/CV-8 consumer can use to discount
an unreliable diagnosis without ever hiding the candidate.

## Full-scale confirmation (1,000-image seeded sample, same set used
throughout this branch's evaluation; run post-CV-1-recalibration, so
outcome counts differ slightly from the original assembly result —
QUALITY_REJECTED dropped 21.5%→12.4% as expected from that fix)

3,336 candidates. The pattern holds almost exactly, not an artifact of
the 150+39 calibration sample:

| class | n | crop_contrast mean | % below 0.20 | confidence mean |
|---|---|---|---|---|
| MEL | 42 | 0.381 | 2.4% | 0.670 |
| NEV | 1808 | 0.303 | 22.3% | 0.808 |
| BCC | 403 | 0.207 | 52.1% | 0.576 |
| SEK | 838 | 0.204 | 60.0% | 0.666 |
| ACK | 244 | 0.149 | **88.1%** | 0.604 |

ACK is even more extreme at full scale than the calibration sample
indicated (mean 0.149 vs 0.165; 88.1% below 0.20 vs the smaller sample's
directional signal). BCC and MEL both replicate closely (0.207 vs 0.208;
0.381 vs 0.337). The two classes with the lowest confidence out of
domain (BCC, ACK) are exactly the two with the lowest crop contrast —
the mechanism identified in calibration is the same mechanism operating
at scale, not a small-sample coincidence.

## Decision

Per the spec's anti-rabbit-hole boundary: this is an evidence-layer
addition, not a component with an accuracy gate. The classification and
action distribution are unchanged by design — only what accompanies the
diagnosis changed. Stopping here; acting on the evidence (e.g. a CV-6
abstention policy) is a separately-scoped future task.
