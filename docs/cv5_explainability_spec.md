# CV-5 Explainability — Spec

**Status:** Implemented (2026-09-01). Result:
`analysis/product_eval/cv5_explainability/result.md`. This document is
kept as-written (committed before implementation) — the "Why not now"
section below described the state at spec-writing time; the deferred
implementation happened later the same day.

## What CV-5 is

The human-facing explanation of a CV-4 diagnosis: what did the system
focus on, and why does it look concerning? Sits after CV-4, feeds CV-8
convergently alongside CV-6/CV-7 (`docs/project_state.md`). The only
prior commitment: CV-3's segmentation mask is CV-5's input, recorded as
evidence on every candidate but never consumed by anything yet
(`docs/cv1_cv4_assembly_spec.md`, decision 1 — "The mask is recorded
alongside the diagnosis... for CV-5 explainability"). Nothing else about
CV-5's output was decided anywhere before this document.

## Design: two complementary overlays, both inference-time only

**1. Mask overlay — direct reuse, zero new design.** The exact pattern
already proven in `scripts/validate_cv3_domain_itobos.py::draw_overlay`:
`cv2.findContours` + `cv2.drawContours` on the CV-3 mask, drawn on the
crop CV-4 saw. Shows "here's the region CV-3 segmented" — cheap,
already validated, no model changes.

**2. Grad-CAM heatmap — shows what CV-4 itself attended to**, which the
mask overlay cannot: CV-3 and CV-4 are independent components (per the
locked decision that CV-3 never gates CV-4's input), so CV-3's mask
does not necessarily reflect what pixels actually drove CV-4's
diagnosis. A classifier-focused explanation needs the classifier's own
attention, not the segmenter's.

## Why this needs real (if small) implementation work, not a hook

Investigated directly, not assumed:

- `src/models/native_classifier.py`'s `SharedResNetXXBackbone` wraps
  torchvision ResNet as `self.features = nn.Sequential(*list(backbone.children())[:-1])`
  — this single `Sequential` includes `AdaptiveAvgPool2d`, so the
  pre-pool conv feature map Grad-CAM needs (`[B,512,7,7]` for resnet18@224)
  is not separately exposed. Getting it means either positional-index
  hooking (`backbone.features[7]`, fragile — breaks silently if the
  Sequential's construction ever changes) or a small, deliberate
  addition: a `forward_conv_features()` method.
- `src/inference/native.py::NativePredictor.predict()` is
  `@torch.no_grad()` and returns only the `NativePrediction` dataclass
  — no logits, no activations, no gradient path. Grad-CAM requires
  `.backward()`, so it cannot go through `predict()`; it needs a
  separate gradient-enabled path reaching `predictor.model` directly.
- No heatmap-rendering code exists anywhere in the repo
  (`cv2.applyColorMap` + `cv2.addWeighted`, the standard approach, has
  no precedent here — only the contour-overlay pattern above).

None of this needs training or new data. All of it is bounded,
inference-time, testable immediately against the existing frozen
checkpoint. It is real engineering effort, not a trivial reuse.

## Why not implemented in the same session as CV-6

`src/models/native_classifier.py` is the one file this entire
CV-1→CV-4 assembly effort was careful never to touch — CV-4 is the
"substantially established" baseline everything else (CV-1.5's routing,
CV-2's detection, CV-3's segmentation) was built and evidenced against
without altering. Even an additive, non-breaking method deserves its
own focused pass — implemented, tested against the frozen checkpoint,
and verified not to change `NativePredictor`'s existing behavior —
rather than being folded into CV-6's implementation turn.

## Pre-committed design decisions for implementation

1. **Additive only.** The new `forward_conv_features()`-style method
   must not change `forward()`'s existing signature, return shape, or
   behavior. `tests/test_native_inference_reproduction.py` (existing,
   passing) must still pass unmodified after the change — that is the
   regression check.
2. **Evidence, not input.** Like CV-3's mask, CV-5's output is recorded
   evidence for a human reviewer / future CV-8, never fed back into
   CV-4 or used to alter `product_action`/`gate_decision`. Same
   principle as CV-6 (`docs/cv6_uncertainty_spec.md`) and the CV-3-mask
   decision before it.
3. **No new dependency for colormap rendering** — `cv2.applyColorMap`
   is already available via the existing OpenCV dependency; no new
   library needed.

## Anti-rabbit-hole boundary

Two overlays, both inference-time, no training. Do not extend to
counterfactual explanations, SHAP/LIME, or multi-layer CAM variants
(Grad-CAM++, ScoreCAM) without evidence that plain Grad-CAM is
insufficient — that evidence does not exist yet because nothing has
been built. Ship the simplest version, evaluate it once against a
real, visually-checkable sample (following the same visual-audit
pattern used for CV-1.5 Stage 2 and the CV-4 domain evidence work), and
stop.
