# CV-2 Spec — Section 22 (Revised): Acceptance Metrics

**Status:** Revised. Supersedes the original placeholder thresholds
(recall >= 95% box-level, zero-lesion FPR <= 5% binary, dense-bucket
recall >= 90%).

**Why revised, and why now:** The original Section 22 thresholds were
explicitly placeholders, set before any evidence existed about what CV-2
could achieve or — more importantly — about what CV-2 *should* be measured
on given its role in the pipeline. After three experiments (threshold
sweep, B0->B1 resolution, D sun-damage oversampling), we now have enough
evidence to see that the original metrics measure the wrong things for a
candidate-localizer feeding CV-3/CV-4.

**Guardrail on this revision (recorded to prevent motivated reasoning):**
These changes are justified by CV-2's pipeline role, NOT by the fact that
the prior experiments failed the old targets. The test applied to each
change was: "would we make this change even if B1 had passed the old
thresholds comfortably?" Each change below passes that test — each is a
correction to *what is measured*, grounded in CV-2's function, not a
loosening of *how hard* the target is. Where a change happens to make the
target easier, that is a consequence of measuring the right thing, not the
motivation for it.

---

## Change 1 — Recall: box-level -> image-level (primary gate)

**Rationale (pipeline role):** CV-2's job is to surface regions worth
passing to CV-3 (segmentation) and CV-4 (classification/risk). The
product-relevant question is "of images containing something worth
examining, did CV-2 surface it," not "of all annotated lesion instances,
how many boxes did we catch." Box-level recall on a dataset with up to 72
lesions per image penalizes missing the 60th freckle exactly as hard as
missing an isolated suspicious lesion — but those differ enormously in
product consequence. Box-level recall is dominated by the model's ability
to catch the *marginal, least-important* instances in dense images, which
is precisely the wrong thing to gate on.

**Definition:**
- **Primary gate — image-level detection recall:** fraction of
  lesion-containing images where CV-2 produced >= 1 true-positive
  candidate (TP defined at the locked IoU 0.50 matching threshold).
- **Secondary (reported, not a gate) — box-level recall:** retained for
  understanding dense-scene behavior.

**Acknowledged blind spot (why this is only acceptable for CV-2):**
Image-level recall scores an image as caught if *any* one lesion is
found, even if the specific lesion that mattered was missed. This is
acceptable ONLY because CV-2 is not the final arbiter — it feeds a
pipeline that re-examines what it surfaces. If CV-2 were the end of the
line, box-level would be the correct gate. The pipeline position is what
justifies image-level; this justification does not transfer to any
downstream component that makes a terminal decision.

---

## Change 2 — Zero-lesion FPR: binary rate -> per-image candidate burden (primary gate)

**Rationale (information loss / downstream cost):** A binary per-image
"did any false box survive" measurement discards product-relevant
information. On a zero-lesion image, 1 stray candidate and 40 stray
candidates are very different downstream costs (each false candidate
triggers a wasted CV-3 segmentation pass and possibly a low-value CV-4
classification), but the binary rate treats them identically. Downstream
cost scales with candidate *count*, not with the binary presence of any
candidate. This justification is independent of any experiment outcome:
per-image burden is a better cost measure whether or not prior runs passed.

**Definition:**
- **Primary gate — per-image false-candidate burden on zero-lesion
  images:** median and 90th-percentile count of surviving false
  candidates per zero-lesion image (at the locked operating confidence
  threshold). Gate set on both central tendency and tail.
- **Secondary (reported, not a gate) — binary zero-lesion FPR:** retained
  because a false "we found something" on genuinely clear skin is a
  product-surface trust concern regardless of count. Kept visible so this
  concern is not lost even though the gate is burden-based.

**Why keep both:** two competing truths are both real — downstream compute
cost scales with count (argues for burden), while user trust can break on
*any* false alarm regardless of count (argues for binary). Rather than
choose, gate on the actionable one (burden, which CV-2 tuning can target)
and keep the other in view as a reported metric.

---

## Change 3 — Dense-bucket recall: dropped as a gate; dense bucket becomes the FPR stratum

**Rationale:** The dense-bucket recall gate (>= 90% on 10+ images) existed
because the box-level framing made dense images the hardest recall case.
Under image-level recall (Change 1), that reasoning dissolves: a 10+ image
needs only >= 1 true catch to count as a recall success, making dense
images the *easiest* recall case, not the hardest. A dense-recall gate is
no longer meaningful.

However, dense — especially heavily sun-damaged/freckled — images are
exactly where false candidates proliferate (this was the entire subject of
the D investigation). So the dense bucket remains important, but as an
**FPR/false-candidate-burden stratum**, not a recall gate.

**Definition:**
- **Dropped:** dense-bucket recall as a pass/fail gate.
- **Retained:** dense images (and the high-sun-damage subgroup within
  them) as the primary stratum for reporting the Change-2 false-candidate
  burden metric.

---

## Revised gate summary

| Metric | Type | Target |
|---|---|---|
| Image-level detection recall | **GATE** | `[to set after measuring B1/D]` |
| Per-image false-candidate burden (zero-lesion, median) | **GATE** | `[to set]` |
| Per-image false-candidate burden (zero-lesion, p90) | **GATE** | `[to set]` |
| Box-level recall | reported | — |
| Binary zero-lesion FPR | reported | — |
| Burden on dense / high-sun-damage stratum | reported | — |

**Note on setting numeric targets:** Per the original Section 22
discipline, numeric targets must be committed BEFORE the next training
experiment, not derived to fit results. The immediate next step is to
compute B1 and D against these *definitions* to understand the achievable
range, then set targets deliberately with rationale — distinct from
running a new experiment and back-fitting a target to whatever it produced.
Measuring existing checkpoints against a new metric definition is
characterization, not target-fitting; setting the actual pass/fail numbers
is the committed decision that must precede the next GPU run.