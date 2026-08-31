# CV-2 Experiment E — YOLO11s Capacity Test

**Goal:** Test whether increased model capacity (YOLO11s, ~9.4M params, vs
YOLO11n's ~2.6M) closes the image-level recall gap from B1's 0.809 toward
the committed >= 0.90 gate (Section 22 finalized).

**Hypothesis:** The ~19% of lesion-containing images where B1 surfaces
nothing are missed because a nano-capacity model at the extreme small-object
scale of this dataset (median lesion normalized area 0.00097) lacks the
representational capacity to reliably fire on faint/small lesions. A larger
backbone may recover some of these.

**This is a single-variable change from B1** except where VRAM forces a
batch-size change (see caveat).

---

## Configuration

Match B1 exactly except the backbone:

| Param | B1 value | E value | Same as B1? |
|---|---|---|---|
| weights | yolo11n.pt | **yolo11s.pt** | CHANGED (the variable) |
| imgsz | 1280 | 1280 | yes |
| epochs | 50 | 50 | yes |
| batch | 8 | 8 (see caveat) | maybe |
| seed | 42 | 42 | yes |
| optimizer | auto | auto | yes |
| lr0 | 0.01 | 0.01 | yes |
| data | cv2_itobos.yaml (original train.txt) | cv2_itobos.yaml | yes |

**Important:** use the ORIGINAL `configs/cv2_itobos.yaml` (standard
train.txt), NOT the D oversampled list. Experiment E tests capacity in
isolation; combining it with oversampling would confound two variables. The
D result already showed oversampling doesn't help, so it is correctly
excluded here.

---

## VRAM caveat (check before the long run)

YOLO11s is ~3.6x the parameters of nano. At imgsz=1280, batch=8 may exceed
GPU memory. Before committing to a 50-epoch run:

1. Launch and watch the first few iterations. If it OOMs, reduce batch to 4.
2. If batch must change to 4, that is technically a second variable. Note
   it explicitly in the result writeup. It does not invalidate the
   experiment (image-level recall is robust to batch size), but it must be
   recorded so the comparison to B1 is honest rather than silently
   confounded.

Preferred: keep batch=8 if it fits. Fall back to 4 only if forced.

---

## Launch command (on pod, inside tmux)

```bash
# verify sync first (per standing discipline)
git fetch origin
git log --oneline origin/main -3   # confirm the finalized-Section-22 commit is present
git pull origin main

# confirm yolo11s.pt is available (ultralytics auto-downloads if absent,
# but that needs network; confirm before a long session)
ls -la yolo11s.pt 2>/dev/null || echo "yolo11s.pt will auto-download on first use"

# launch in tmux
tmux new -s cv2_e
python -m scripts.train_cv2 --weights yolo11s.pt --data configs/cv2_itobos.yaml --imgsz 1280 --name e_yolo11s
# confirm startup scan line references the ORIGINAL train.txt (6778 images),
# NOT the oversampled list, then detach: Ctrl+b, d
```

**Pre-launch check on build_detector:** confirm
`src/detection/model.py`'s `build_detector(weights=..., pretrained=True)`
actually passes the weights string through to the YOLO constructor and does
not hardcode the nano architecture anywhere. If it just does
`YOLO(weights)`, `--weights yolo11s.pt` is sufficient. If it hardcodes a
model config, that needs a one-line fix first. Check with:
`grep -n "yolo11\|YOLO(" src/detection/model.py`

---

## Evaluation (after training)

Run the full pipeline against E's checkpoint, same as B1/D:

```bash
# 1. locked-metrics eval (original evaluate_cv2, for continuity)
python -m scripts.evaluate_cv2 --weights runs/cv2/e_yolo11s/weights/best.pt --split val --imgsz 1280 --conf 0.25

# 2. prediction diagnostics (needed for revised-metric measurement)
python -m scripts.analyze_cv2_predictions --weights runs/cv2/e_yolo11s/weights/best.pt --split val --imgsz 1280 --output evaluation/cv2/prediction_diagnostics/e_yolo11s

# 3. pull predictions.csv locally, then run the REVISED metric on it:
#    python scripts/measure_cv2_revised_metrics.py --conf 0.25
#    (add e_yolo11s to PRED_FILES in that script)
```

**The decision metric is image-level recall from step 3**, evaluated
against the committed >= 0.90 gate and the stopping rule in Section 22
finalized. Not Ultralytics' internal mAP (which we've established does not
track the product metrics), and not the box-level recall from step 1.

---

## Decision (per Section 22 committed stopping rule)

- image-level recall >= 0.90 -> **CV-2 PASSES**, stop, move to
  CV-2 -> CV-3 interface validation.
- 0.85 <= recall < 0.90 -> one more experiment permitted (tiling/SAHI),
  then stop regardless.
- recall ~unchanged from 0.81 -> not a capacity problem; accept 0.81 floor,
  document, move on. No 11m/11l/augmentation escalation.

Also confirm E does not regress the burden gates (median <= 1, p90 <= 2) —
a larger model firing more freely could raise false-candidate burden even
while improving recall. Both must hold for a pass.