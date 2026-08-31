"""
CV-1.5 domain router validation (Stage 1: classical heuristic).

Runs src.routing.heuristic.route_image against the pre-committed
held-out evaluation set (docs/cv1_5_router_spec.md): a fixed, seeded
sample of 150 PAD-UFES-20 test images (label pre_framed) and 150 iToBoS
test images (label wide_field). Reports per-class accuracy against the
spec's >=90%-per-class gate.

Ground-truth caveat: labels are a proxy (dataset identity), not
per-image-verified framing -- see the spec before interpreting results.

Usage:
    python -m scripts.validate_cv1_5_router
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from src.routing.heuristic import route_image

REPO_ROOT = Path(__file__).resolve().parents[1]

PAD_UFES_TEST = REPO_ROOT / "data/splits/pad_ufes/test.csv"
ITOBOS_TEST = REPO_ROOT / "data/splits/itobos_detection/test.csv"
OUTPUT_DIR = REPO_ROOT / "analysis/quality/cv1_5_router"

SAMPLE_SIZE_PER_CLASS = 150
RANDOM_SEED = 42
PER_CLASS_GATE = 0.90


def build_eval_set() -> pd.DataFrame:
    pad = pd.read_csv(PAD_UFES_TEST).sample(
        n=SAMPLE_SIZE_PER_CLASS, random_state=RANDOM_SEED
    )
    pad = pad[["image_path"]].copy()
    pad["label"] = "pre_framed"

    ito = pd.read_csv(ITOBOS_TEST).drop_duplicates("image_id")
    ito = ito.sample(n=SAMPLE_SIZE_PER_CLASS, random_state=RANDOM_SEED)
    ito = ito[["image_path"]].copy()
    ito["label"] = "wide_field"

    return pd.concat([pad, ito], ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_set = build_eval_set()
    eval_set.to_csv(OUTPUT_DIR / "eval_set.csv", index=False)

    records = []
    for _, row in eval_set.iterrows():
        img_path = REPO_ROOT / row["image_path"]
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        predicted = route_image(image_bgr)
        records.append(
            {
                "image_path": row["image_path"],
                "label": row["label"],
                "predicted": predicted,
                "correct": predicted == row["label"],
            }
        )

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    per_class = results.groupby("label")["correct"].mean()
    pre_framed_acc = float(per_class.get("pre_framed", 0.0))
    wide_field_acc = float(per_class.get("wide_field", 0.0))
    overall_acc = float(results["correct"].mean())

    passed = pre_framed_acc >= PER_CLASS_GATE and wide_field_acc >= PER_CLASS_GATE

    confusion = pd.crosstab(results["label"], results["predicted"])

    summary_lines = [
        "CV-1.5 Domain Router -- Stage 1 (Heuristic) Validation",
        "=" * 60,
        "",
        f"Eval set: {len(results)} images "
        f"({SAMPLE_SIZE_PER_CLASS} pre_framed + {SAMPLE_SIZE_PER_CLASS} wide_field, "
        f"seed={RANDOM_SEED})",
        "",
        f"pre_framed accuracy: {pre_framed_acc:.3f}  (gate >= {PER_CLASS_GATE})",
        f"wide_field accuracy: {wide_field_acc:.3f}  (gate >= {PER_CLASS_GATE})",
        f"overall accuracy:    {overall_acc:.3f}",
        "",
        "Confusion matrix (rows=true label, cols=predicted):",
        confusion.to_string(),
        "",
        f"RESULT: {'PASS' if passed else 'FAIL'} -- "
        + (
            "Stage 1 heuristic clears both per-class gates. This is CV-1.5."
            if passed
            else "at least one class is below the 90% gate. Per "
            "docs/cv1_5_router_spec.md, this means escalating to Stage 2 "
            "(learned classifier) -- do not iterate heuristic thresholds "
            "further. Flag to the user before starting a Stage 2 training run."
        ),
    ]
    summary_text = "\n".join(summary_lines)
    (OUTPUT_DIR / "summary.txt").write_text(summary_text + "\n")

    print(summary_text)
    print()
    print(f"Output written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
