# CV-5 Explainability — Result

**Status:** Implemented per `docs/cv5_explainability_spec.md`. Visual
audit confirms correct spatial alignment; no downstream regression.

## What was built

- `src/models/native_classifier.py` — additive `forward_conv_features()`
  on both `SharedResNet18Backbone`/`SharedResNet50Backbone`, exposing
  the pre-pool conv feature map (`[B,512,7,7]` for resnet18@224).
  Verified exactly reconstructs the existing `forward()` output via
  `F.adaptive_avg_pool2d` (`tests/test_explainability.py::test_forward_conv_features_reconstructs_forward_exactly`)
  — confirms the change is additive, not a behavioral risk to CV-4.
- `src/explainability/gradcam.py::compute_gradcam` — standard Grad-CAM,
  bypassing `NativePredictor.predict()` (which is `@torch.no_grad()`)
  to reach `predictor.model` directly with gradients enabled.
- `src/explainability/overlay.py` — `mask_contour_overlay` (direct
  reuse of `scripts/validate_cv3_domain_itobos.py::draw_overlay`'s
  contour pattern) and `gradcam_heatmap_overlay` (new — `cv2.applyColorMap`
  + `cv2.addWeighted`, no colormap-overlay precedent existed in-repo).
- `src/explainability/evidence.py::gradcam_mask_iou` — a genuinely new
  cross-check: does CV-4's attention overlap with CV-3's segmented
  region? CV-3 and CV-4 have never been compared to each other before
  (the mask never gates CV-4's input, per the locked architecture) —
  this is the first measurement of whether they agree.
- `src/explainability/__init__.py::explain_candidate` — the public
  entry point, tying the above together. **Deliberately not wired into
  `DermaSensePipeline.predict()`'s hot path** — Grad-CAM needs a
  backward pass (meaningfully more expensive than CV-3/CV-6's
  forward-only evidence), and overlay images don't fit
  `CandidateResult`'s scalar-evidence pattern. Invoked on demand by a
  caller that already has a `PipelineResult`.

## Regression check

`tests/test_native_inference_reproduction.py`,
`tests/test_phase4_inference_pipeline.py`,
`tests/test_phase4_known_dangerous_cases.py` — all pass unmodified
after the `native_classifier.py` change (7/7). This is the check the
spec named as the one that mattered: CV-4's existing behavior is
unchanged.

## Visual audit

3 crops (MEL, BCC, NEV predictions) — mask contour + Grad-CAM heatmap
inspected directly. In all three, the heatmap's hotspot lands on the
lesion itself, not on background skin or hair, and is spatially
coherent with the mask contour (centered on the pigmented/lesion region
in each case, including a multi-toned BCC crop where the heatmap
weighted toward the darker sub-region rather than the whole crop
indiscriminately). This is the same kind of go/no-go visual check used
for CV-1.5 and CV-4's domain evidence — it confirms no flip/transpose/
resize bug, which numeric shape/range assertions alone would not catch.

## Decision

Per the spec's anti-rabbit-hole boundary: two overlays, one cross-check
signal, evaluated once against a real visual sample. No SHAP/LIME,
no Grad-CAM++/ScoreCAM, no counterfactual explanations added — plain
Grad-CAM was sufficient on this sample and nothing indicates otherwise.
`gradcam_mask_iou` is recorded as a new measurable quantity but not
acted on (e.g., no policy built around low-IoU candidates) — same
evidence-not-decision principle as CV-3's mask and CV-6's signals,
reserved for CV-8.
