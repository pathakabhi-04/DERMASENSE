# CV-7 Stage 1 Evaluation — Clinical Signal Validity

**Status: DONE. Result: Stage 1 shows a real, statistically significant
signal. Decision: no Stage 2 (learned model / RunPod training) on this
basis.**

This answers question 2 from `docs/cv7_temporal_technical_spec.md`'s
evaluation plan — the question left open when the spec was written,
because the malignant-enrichment data wasn't staged yet. It now is (99+
malignant-outcome lesions), so this was run.

## Pre-committed question, sample, and decision rule

Written into `scripts/evaluate_cv7_stage1.py`'s docstring **before**
running, per this project's bounded-experiment discipline:

- **Question:** do lesion pairs with a malignant outcome diagnosis
  (melanoma/BCC/SCC) show a higher rate of non-STABLE verdicts, and/or
  higher delta magnitude, than benign-outcome pairs?
- **Sample:** ALL malignant-outcome lesions with ≥2 visits in the
  staged data (no further sub-sampling — few enough that all of them
  were used), vs. a bounded random sample of up to 300 benign-outcome
  lesions (seed=17) — a fair comparison group, not cherry-picked.
- **Decision rule:** Fisher's exact test on non-STABLE vs. STABLE
  counts, and Mann-Whitney U on magnitude distributions, both two-sided,
  p<0.05, fixed before running. NO_PRIOR_DATA pairs excluded from both
  (that reflects CV-3 mask availability, not a temporal signal, and
  would confound the comparison). Significance in the expected
  direction (malignant higher) → real signal, sufficient to proceed
  without Stage 2. No significance → does not automatically trigger
  Stage 2 (must be weighed against known instrument limitations first).
  Wrong-direction significance → reported as-is.

## Results

| | malignant | benign |
|---|---|---|
| Pairs processed | 101 | 300 |
| Excluded (NO_PRIOR_DATA) | 12 | 16 |
| Scored | 89 | 284 |
| Non-STABLE count | 23 | 39 |
| **Non-STABLE rate** | **25.8%** | **13.7%** |

**Fisher's exact test:** odds ratio = 2.19, **p = 0.0135** (significant,
malignant higher — the expected direction).

**Mann-Whitney U on magnitude:** U = 17398.0, **p = 8.2×10⁻⁸**
(highly significant, malignant higher).

Both pre-registered tests are significant in the expected direction.

## Interpretation

Per the decision rule fixed in advance: **Stage 1's classical
measurements do discriminate between malignant- and benign-outcome
lesions on this data.** Lesion pairs with a malignant outcome are
roughly twice as likely to register a non-STABLE verdict, and their
overall delta-magnitude distribution is shifted meaningfully higher —
this holds even though `magnitude` is a normalized composite across
three differently-scaled features (size/border/color ratios to their
own thresholds), not a single physical unit, and even though most of
the underlying signal here is almost certainly coming from
border/color deltas rather than size (size-change detection remains
gated by calibration's 4.0% single-image coverage, ~0.3% for both
visits of a pair — see `delta_calibration_result.md`).

**Caveats, stated plainly:**
- n=89 scored malignant pairs is not large. The result is significant,
  not definitive — this is one bounded evaluation, not a clinical
  validation study.
- Only one consecutive visit pair per lesion was used (matching the
  same convention as the delta-threshold calibration run), not a
  lesion's full visit history.
- This does not, by itself, establish *why* malignant lesions show more
  measured change (real biological evolution vs. e.g. malignant
  lesions being biopsied/monitored more closely and thus photographed
  under more variable conditions is a plausible confound, given the
  already-documented lighting/capture-condition noise in the color
  channel). Establishing causation was never the pre-committed
  question — discriminative signal was — but this confound should be
  kept in mind by whoever weights this evidence in CV-8.
- No correction for multiple comparisons was applied (2 pre-registered
  tests); both are significant well past a Bonferroni-corrected
  threshold anyway (p=0.0135 and p=8.2e-8 vs. 0.025), so this does not
  change the outcome.

## Decision

**Stage 1 (the classical, deterministic pipeline — calibration.py +
measurement.py + delta.py + pipeline.py) is sufficient.** Per the
technical spec's own anti-rabbit-hole boundary — "Stage 2 is not
attempted preemptively... only if Stage 1's own evaluation shows a
genuine, specific insufficiency" — that insufficiency was not found.
**No Stage 2 learned model, and therefore no RunPod GPU training, is
needed for CV-7 on the basis of this evaluation.** CV-7 proceeds
directly to CV-8 integration as a classical, deterministic component,
same as CV-1.5 remained Stage-1-only after its own evaluation found no
need to escalate.

This decision can be revisited if CV-8 integration surfaces a specific
insufficiency Stage 1 can't address (e.g. a need for size-change
detection at higher coverage than calibration's 4% ceiling allows) —
but that would be a new, specific trigger, not a default assumption
that "a learned model would probably do better."
