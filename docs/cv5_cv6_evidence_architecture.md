# CV-5 / CV-6: How They Were Built, and Why They Fit

**Purpose of this document:** CV-5 (Explainability) and CV-6
(Uncertainty) were built in the same session, immediately after the
CV-1→CV-4 assembly and its two follow-on investigations (CV-1
recalibration, CV-4 domain evidence). This document is the consolidated
record of how they were built and — more importantly — *why* they are
not two unrelated features bolted onto the pipeline, but two
applications of one architectural pattern that has now been used four
times. The individual specs (`docs/cv5_explainability_spec.md`,
`docs/cv6_uncertainty_spec.md`) record the pre-committed design and
acceptance criteria for each; this document is the cross-cutting view.

---

## 1. The pipeline, with CV-5/CV-6 in place

```
                    image (BGR, as cv2.imread returns it)
                                  |
                                  v
                         +-----------------+
                         |       CV-1      |   unusable
                         |   quality gate  |------------> QUALITY_REJECTED
                         +-----------------+               (outcome, not
                                  | usable                   a silent drop)
                                  v
                         +-----------------+
                         |      CV-1.5     |
                         |  domain router  |
                         +-----------------+
                        pre_framed |  | wide_field
                +------------------+  +------------------+
                v                                         v
   candidate = whole frame                       +-----------------+
                |                                 |       CV-2      |  0 boxes
                |                                 |     detection   |----> NO_CANDIDATES
                |                                 +-----------------+      (outcome, not
                |                                         | N boxes           "cleared")
                |                                         v
                +<----------------------------------------+
                                  |
                ============ per candidate ================================
                |                                                          |
                |         crop_and_normalize(margin=0.25)                  |
                |                       |                                  |
                |          +------------+------------+                     |
                |          v                         v                     |
                |   +-------------+           +-------------+              |
                |   |     CV-3    |           |     CV-4    |              |
                |   | segmentation|           |  classifier |              |
                |   +-------------+           +-------------+              |
                |          | mask                    | diagnosis           |
                |          v                          v + action + gate    |
                |   mask_evidence()          NativePrediction               |
                |   area / degenerate /              |                     |
                |   border-touch                     +--> crop_contrast,   |
                |          |                          |    crop_blur       |
                |          |                          |    (src/quality/   |
                |          |                          |     signals,       |
                |          |                          |     reused)        |
                |          |                          |                    |
                |          |                          +--> CV-6 evidence   |
                |          |                          |    (opt-in):       |
                |          |                          |    ensemble agree/ |
                |          |                          |    distance,       |
                |          |                          |    calibrated conf.|
                |          v                          v                    |
                |   +--------------------------------------------+         |
                |   |               CandidateResult               |         |
                |   |  every signal above flattened to a scalar,  |         |
                |   |  merged via .to_dict() -- NONE of it is ever |         |
                |   |  fed back into CV-4's own input              |         |
                |   +--------------------------------------------+         |
                |                                                          |
                ============================================================
                                  |
                                  v
               aggregate (most-severe action wins) --> PipelineResult [ASSESSED]


      -------------------- on demand, NOT in the hot path ---------------------

   PipelineResult + original image
               |
               v
     explain_candidate()  (src/explainability)
               |
               +--> recompute CV-3 mask (cheap, deterministic, same weights)
               +--> Grad-CAM on CV-4 (needs a backward pass --
               |    bypasses NativePredictor.predict()'s @torch.no_grad())
               v
     ExplanationResult
       - mask_contour_overlay      )  two images, don't fit
       - gradcam_heatmap_overlay   )  CandidateResult's CSV-scalar shape
       - gradcam_mask_iou  <-- first-ever CV-3-vs-CV-4 agreement check


      ------------------------------ not yet built ------------------------------

     CandidateResult evidence (CV-3, crop_contrast, CV-6, CV-5's IoU)
                    +  CV-7 (temporal -- blocked, see docs/cv7_temporal_blockers.md)
                                  |
                                  v
                         +-----------------+
                         |       CV-8      |   <-- the ONLY place all of this
                         |   risk engine   |       is allowed to become a
                         |  (convergent)   |       product decision
                         +-----------------+
```

---

## 2. The one pattern, used four times

Every evidence signal added this session follows the same rule, first
stated when CV-3's mask was designed:

> **CV-4 drives risk, so it must depend on as few upstream failure
> points as possible.** Evidence is recorded; it never reshapes what a
> downstream component sees, and it never decides anything on its own.

This was not re-derived each time — it was applied, deliberately, as
the same rule to four different additions in sequence:

| # | Signal | Where it's computed | What it would be tempting to do instead | Why that was rejected |
|---|---|---|---|---|
| 1 | CV-3 mask evidence (`mask_area_fraction`, `mask_degenerate`, `mask_touches_border`) | `src/segmentation/inference.py::mask_evidence` | Use the mask to re-crop or black out CV-4's input | A bad mask (measured ~22% unreliable on TBP crops) could crop away the exact tissue CV-4 needed, turning a segmentation miss into a diagnosis false-reassurance — this product's worst failure mode |
| 2 | `crop_contrast` / `crop_blur` | `src/quality/signals.py` (CV-1's own signals, reapplied to the CV-4 crop) | Filter out low-contrast candidates before they reach CV-4 | A faint crop can still be a genuine lesion; dropping it recreates the silent-miss failure mode `PipelineOutcome` exists to prevent |
| 3 | CV-6: ensemble agreement/distance, calibrated confidence | `src/uncertainty/` | Wire a confidence threshold into `src/risk/safety_gate.py` | Collapsing evidence into a decision is CV-8's job (the convergent risk engine), not any individual phase's — building it here would pre-empt CV-8 with one signal's view, before CV-5/CV-7 exist to converge with |
| 4 | CV-5: `gradcam_mask_iou` | `src/explainability/evidence.py` | Use low IoU to flag or suppress a diagnosis | Same reasoning as #3 — an explanation's job is to inform a human or CV-8, not to gate CV-4 |

Four additions, one rule, applied without exception. That consistency
*is* the architectural coherence — not that CV-5 and CV-6 share code
(they mostly don't), but that neither could have been added in a way
that violates what CV-3's evidence already established.

---

## 3. How CV-6 (Uncertainty) was actually built

**Order of work:** spec (`docs/cv6_uncertainty_spec.md`, pre-committed
before writing code) → implementation → unit tests (no checkpoints) →
integration tests (real checkpoints) → full-scale evaluation → result
doc → commit.

**Reuse, not new invention:**
- `expected_calibration_error` — moved verbatim (logic unchanged) from
  `scripts/evaluate_c1_vs_f1_product.py::compute_ece`, which had been a
  one-off script function with no `src/` presence until now.
- The ensemble is two checkpoints that already existed on disk
  (`pad_ufes_c1_partial_finetune_seed{42,123}_best.pt`) — no new
  training. `load_ensemble` is a thin wrapper around the existing
  `NativePredictor.from_checkpoint`, called twice.
- `apply_temperature` needed no change to `NativePredictor`/`native.py`
  at all: temperature scaling is normally applied to logits
  (`softmax(z/T)`), but `predict()` never exposes logits. The identity
  `softmax(log(p)/T) == softmax(z/T)` (the unknown softmax
  normalization constant cancels) means it can be computed from
  probabilities alone — a derivation, not a workaround, that avoided
  touching CV-4's inference code a second time in the same session.

**Cost discipline:** ensemble evidence is opt-in
(`additional_ensemble_checkpoints` / `--ensemble`), off by default,
because running a second classifier roughly doubles CV-4 inference cost
per candidate. Calibration is always-on because it's cheap (no extra
model inference, just arithmetic on probabilities already computed).

**What the evaluation actually found, and why it's evidence of good
design, not a weak result:** the pre-committed question was whether
ensemble disagreement would independently corroborate `crop_contrast`'s
finding that BCC/ACK collapse out-of-domain. It partially did (BCC) and
partially didn't (ACK agreed confidently instead of disagreeing; MEL
disagreed most despite being visually fine). The spec had explicitly
pre-committed to documenting a "no" honestly rather than forcing a
cleaner story — and the resulting finding (two complementary, not
redundant, signals) is now the concrete justification for why CV-8
needs multiple convergent inputs rather than one. See
`analysis/product_eval/cv6_uncertainty/result.md`.

---

## 4. How CV-5 (Explainability) was actually built

**Order of work:** spec → one additive model method, verified safe
before anything else was built on top of it → Grad-CAM → overlays →
cross-check evidence → unit tests → integration test → **visual audit
against real images** (not just shape/range assertions) → result doc →
commit.

**The one piece of new model surgery, done carefully:**
`src/models/native_classifier.py`'s `SharedResNet18Backbone` /
`SharedResNet50Backbone` wrap `resnet.children()` as a single
`nn.Sequential` that includes the final `AdaptiveAvgPool2d`, so there
was no existing way to get the pre-pool conv feature map Grad-CAM needs.
The fix was one additive method,
`forward_conv_features()` (`self.features[:-1](x)`), added without
touching `forward()`. Before writing anything else, this was verified
two ways:
1. `F.adaptive_avg_pool2d(forward_conv_features(x), 1)` reconstructs
   `forward(x)` exactly (`torch.allclose`, not "close enough").
2. The three existing CV-4 regression tests
   (`test_native_inference_reproduction.py`,
   `test_phase4_inference_pipeline.py`,
   `test_phase4_known_dangerous_cases.py`) still pass unmodified.

Only after both checks passed did Grad-CAM get built on top of it — the
one change to the component every other decision this session was
built to protect got the most scrutiny, proportionate to that role.

**Reuse:** `mask_contour_overlay` is the exact contour-drawing pattern
already proven in
`scripts/validate_cv3_domain_itobos.py::draw_overlay`
(`cv2.findContours` + `cv2.drawContours`), not a reimplementation.
`gradcam_heatmap_overlay` is genuinely new — no colormap-overlay
precedent existed anywhere in the repo — so it got the extra scrutiny of
a **visual audit**: 3 real crops (MEL/BCC/NEV), heatmap and mask
contour inspected directly, not just asserted to have the right shape.
In all three, the heatmap's hotspot lands on the lesion itself, not
background skin or hair — the kind of alignment bug (a resize or
transpose mistake) that only a human eye catches, matching the same
visual-audit discipline used for CV-1.5's Stage 2 checkpoint and the
CV-4 domain-evidence work.

**Why it stayed out of the hot path:** `explain_candidate()` needs a
backward pass (real cost) and returns images (don't fit the CSV-scalar
`CandidateResult` pattern every other evidence signal uses). It is a
separate, on-demand entry point a caller invokes with a `PipelineResult`
already in hand, not something `DermaSensePipeline.predict()` runs for
every candidate automatically.

---

## 5. What "coherent with the rest of the system" means concretely

Not an abstract claim — five checkable properties, all true as of this
commit:

1. **No component's regression tests changed behavior.** CV-4's 7
   existing tests, the 34 pipeline-assembly/capture-guidance/CV-1
   tests, all pass unmodified after both CV-5 and CV-6 landed. Adding
   evidence never altered an existing decision.
2. **Nothing was duplicated.** Every reusable piece (CV-1's
   blur/contrast signals, the mask-contour pattern, `compute_ece`,
   `NativePredictor.from_checkpoint`) was reused, not reimplemented.
3. **Every evidence signal reaches the same place, the same way.**
   `CandidateResult.to_dict()` is the single flattening point all
   CV-3/CV-4/CV-6 scalar evidence goes through — one place a future
   CV-8 (or any analysis script) reads from, not four different
   conventions.
4. **Cost is opt-in wherever it's non-trivial.** CV-2 is optional
   (`detector_weights=None`), CV-6's ensemble is optional
   (`additional_ensemble_checkpoints=None`), CV-5 is entirely outside
   `predict()`. Nothing expensive runs unless something asked for it.
5. **Nothing decides.** Every new signal — `crop_contrast`,
   `ensemble_agree`, `calibrated_confidence`, `gradcam_mask_iou` — is
   readable, loggable, auditable, and inert. `src/risk/safety_gate.py`
   is byte-for-byte unchanged since before this session started.

That fifth property is the actual answer to "how does this fit the rest
of the system": it doesn't yet produce a better product decision. It
produces the raw material CV-8 will need to, once CV-8 exists. Building
that convergence now, one signal at a time, inside CV-5 or CV-6, would
have been building CV-8 by accident — which is exactly the mistake this
document's Section 2 shows was avoided, four times in a row.
