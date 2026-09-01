# CV-4 Out-of-Domain Behavior — Investigation and Evidence Layer Spec

**Status:** Investigation complete (see findings below); fix
pre-committed before implementation, per the same discipline as prior
specs.

## Why

The CV-1→CV-4 assembly measured 32.7% of assessed wide-field images
escalating to URGENT_EVALUATION and 85% requiring review, driven by a
12.9% per-candidate high-risk rate (357 BCC, 39 MEL, 1 SCC out of 3,082
candidates) — implausible for a routine screening population, and
flagged as CV-4 running out of domain on TBP crops
(`analysis/product_eval/cv1_cv4_assembly/result.md`).

## What was ruled out first

No ground truth exists for iToBoS, so accuracy cannot be measured
directly. Two cheap hypotheses were tested and rejected before further
investigation:

- **CV-2 detection confidence** does not separate high-risk from
  non-high-risk predictions (0.529 vs 0.534 mean) — not a
  garbage-detection-in problem.
- **CV-3 mask evidence** (area fraction, degenerate rate, border-touch)
  does not separate them either (0.237 vs 0.282 area fraction) — not
  explained by segmentation failure.
- **Overall confidence** is not dramatically different in/out of domain
  (0.733 vs 0.760 mean) — not a blanket miscalibration.

## What the investigation found

**The failure is class-specific, not general.** Per-class mean
confidence out of domain: BCC 0.578, ACK 0.590 (both markedly lower than
their in-domain PAD-UFES means of 0.743 and 0.773) versus MEL 0.672 and
NEV 0.812 (closer to in-domain). Visual audit (36 crops: top-12
highest-confidence BCC/MEL, plus 12 random BCC and 12 random MEL) shows
two structurally different populations:

- **MEL predictions are visually coherent** — dark, irregular, roundish
  pigmented blobs with real internal structure, across every sample
  checked including an unbiased random draw. Whatever their true
  diagnosis, these are genuine lesion-shaped candidates.
- **BCC predictions are frequently visually incoherent** — diffuse
  low-contrast pink/red patches with no clear border, hair-dominated
  crops with little visible skin, and in one case a linear
  scratch/mark. Many do not look like a lesion at all.

**A quantifiable signal separates them: crop-level contrast.**
`src/quality/signals.py::contrast_signal`, already used by CV-1 on
whole source images, applied instead to the exact crop CV-4 receives
(n=150 BCC, 39 MEL, seeded sample): BCC mean 0.208 (58.7% fall below
0.20), MEL mean 0.337 (only 5.1% below 0.20). ACK shows the same pattern
as BCC (mean 0.165). This is a genuinely new signal — it does not
overlap with the CV-2/CV-3 evidence already ruled out above
(`corr(crop_contrast, mask_area_fraction)` is not the same measurement;
mask evidence looks at the segmented region, crop_contrast looks at the
raw crop's own texture).

**Consistent with, not a duplicate of, the existing SCC/BCC finding.**
`analysis/scc_bcc/` already established BCC/SCC representation overlap
in-domain, not fixable by reweighting/SupCon, accepted as a known
limitation. This investigation does not reopen that — it identifies a
different, previously-untested lever: low-information crops (blurry,
low-contrast, hair-dominated) land disproportionately in that already-
known weak region of CV-4's decision space. The mitigation below does
not touch CV-4's weights or retrain anything.

## What this is NOT going to fix

**CV-4 will not be retrained or redesigned.** The SCC/BCC embedding
overlap is already documented as not fixable by reweighting/SupCon
alone (`analysis/scc_bcc/`) — reopening that question is out of scope.
No labeled TBP data exists to retrain against even if it were in scope.

**Candidates will not be filtered or dropped.** A low-contrast crop is
not necessarily a non-lesion — it could be a genuine but faint lesion.
Dropping it would recreate exactly the silent-miss failure mode the
`PipelineOutcome` design was built to prevent
(`docs/cv1_cv4_assembly_spec.md`). The fix must not let evidence turn
into a gate.

**The conservative aggregation rule will not be weakened.** Established
in the assembly work: a screening tool must not hide a high-risk
candidate behind an averaged verdict. Unchanged here.

## The fix: crop-quality evidence, general-purpose

Add crop-level quality signals (blur, contrast — reusing
`src/quality/signals.py` directly, no new signal design) to
`CandidateResult`, computed on the exact RGB crop CV-4 receives.
**Recorded for every candidate regardless of predicted class** — not
special-cased to BCC. The BCC/ACK correlation is a finding to document,
not a rule to hardcode; a future class could exhibit the same failure
mode and the evidence should already be there to see it. This mirrors
the CV-3 mask precedent exactly: evidence alongside the diagnosis, never
gating it.

Consumers (a human reviewer, or eventually CV-6 uncertainty / CV-8 risk
context) can use `crop_contrast` to discount a low-input-quality
diagnosis without the pipeline ever hiding the candidate. This is a
disclosure mechanism, not a filter.

## Pre-committed evaluation

Re-run the wide-field structural propagation eval
(`scripts/evaluate_pipeline_end_to_end.py --split itobos`) with the new
fields recorded, on the same 1,000-image seeded sample already used, and
confirm the correlation holds at that scale (not just the 150+39
calibration sample): report crop_contrast by predicted class, and the
fraction of BCC/ACK predictions falling below 0.20 vs. other classes.

**No pass/fail gate here** — this is an evidence-layer addition, not a
component with a accuracy target to clear. The deliverable is the
recorded field plus the documented finding, not a changed action
distribution (the fix is disclosure, so the raw classification/action
numbers should NOT change — that is the point: the *diagnosis* is
unchanged, only what accompanies it for review).

## Anti-rabbit-hole boundary

One new evidence field (crop-level blur + contrast), computed once, on
the crop already available in the orchestrator (no new inference pass).
Documented as a finding, not chased into a retraining project or a
filtering mechanism. If a future session wants to act on this evidence
(e.g., build a CV-6-style abstention policy), that is separately scoped,
not assumed here.
