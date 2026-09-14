"""
Resolve the CV-1.5 routing question raised by external_6class_result.md:
the router sends 45% of real clinical close-ups to `wide_field`, where
CV-2 then finds nothing 30.8% of the time. Is that routing helping or
hurting?

## Why this is a measurement, not a training run

There is no ground-truth framing label for DDI/DDI-2/Fitzpatrick, so
"is the router correct?" is unanswerable as posed. The question that IS
answerable, and is the one that matters, is operational:

    Does forcing every image down the pre-framed branch route more
    melanomas to a clinician than letting the router decide?

That needs no framing labels -- only the outcome we already care about.

## Pre-committed before running

  Arm A  router decides            (already measured: external_predictions.csv)
  Arm B  forced pre_framed         (this script)

  Primary metric: melanomas reaching a clinician, UNCONDITIONAL over all
  107 -- an image the pipeline never assessed counts as not reaching one,
  because a silent NO_CANDIDATES is exactly the failure being examined.

  Cost metric:    benign (NEV+SEK) referral rate. A routing change that
                  refers everything is not an improvement.

  Decision rule:
    B - A >= +5 points melanoma, and benign referral rises <= +10 points
        -> the router is a net negative on this input class. Recommend
           defaulting close-up product input to pre_framed, or retraining
           CV-1.5 with a third source.
    |B - A| < 5 points
        -> routing is not the lever. Stop here and look at CV-2 instead.
    B - A <= -5 points
        -> the wide_field branch is doing real work. Leave the router
           alone; the 30.8% no-candidate rate is CV-2's problem, not
           CV-1.5's.

  Diagnostic (secondary, not a gate): whole-frame `mask_area_fraction`
  from CV-3 for every image, split by what the router said. If images
  the router called wide_field really do have smaller lesion-to-frame
  ratios, the router is tracking something real even where it costs
  outcomes. PAD-UFES's own median is 0.312
  (domain_composition_audit.md) as the pre-framed anchor.

The router is patched at module scope inside this script rather than
adding a framing override to `DermaSensePipeline.predict()` -- an
experiment must not leave a test hook in the production inference path.

    PYTHONPATH=. python3 scripts/resolve_cv1_5_routing.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

import src.inference.orchestrator as orchestrator_module
from src.inference.orchestrator import DermaSensePipeline, PipelineOutcome
from src.training.metrics import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SET = REPO_ROOT / "analysis/quality/mel_sensitivity/external_6class_eval_set.csv"
ARM_A = REPO_ROOT / "analysis/product_eval/cv1_cv4_assembly/external_predictions.csv"
OUT_DIR = REPO_ROOT / "analysis/quality/mel_sensitivity"

ROUTER_CHECKPOINT = REPO_ROOT / "checkpoints/cv1_5_router/best.pt"
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)

BENIGN_PREDICTIONS = {"NEV", "SEK"}
PAD_UFES_MEDIAN_AREA_FRACTION = 0.312  # domain_composition_audit.md


def melanoma_reach_rate(per_image: pd.DataFrame, total_melanomas: int) -> tuple[int, float]:
    """Unconditional: never-assessed melanomas count as not reaching a clinician."""
    assessed = per_image[
        (per_image["true_class"] == "MEL")
        & (per_image["outcome"] == PipelineOutcome.ASSESSED.value)
    ]
    reached = int((~assessed["predicted_class"].isin(BENIGN_PREDICTIONS)).sum())
    return reached, reached / total_melanomas


def benign_referral_rate(per_image: pd.DataFrame) -> float:
    assessed = per_image[
        per_image["true_class"].isin(BENIGN_PREDICTIONS)
        & (per_image["outcome"] == PipelineOutcome.ASSESSED.value)
    ]
    if assessed.empty:
        return float("nan")
    return float((~assessed["predicted_class"].isin(BENIGN_PREDICTIONS)).mean())


def run_forced_pre_framed(table: pd.DataFrame, device: str) -> pd.DataFrame:
    # Force the routing decision. Contained to this process.
    orchestrator_module.route_image_classifier = lambda *a, **k: "pre_framed"

    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        detector_weights=None,  # never needed: every image is pre_framed here
        device=device,
    )

    records = []
    out_path = OUT_DIR / "cv1_5_routing_forced_preframed.csv"
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        result = pipeline.predict(image)
        record = {
            "image_id": Path(row["image_path"]).stem,
            "true_class": row["label"],
            "source": row["source"],
            "outcome": result.outcome.value,
            "framing": result.framing,
        }
        if result.candidates:
            candidate = result.candidates[0]
            record["predicted_class"] = candidate.predicted_class
            record["mask_area_fraction"] = candidate.mask_area_fraction
        records.append(record)
        if position % 100 == 0 or position == len(table):
            print(f"  {position}/{len(table)}", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(out_path, index=False)  # persist BEFORE validating
    # QUALITY_REJECTED returns before routing runs at all, so framing is
    # legitimately None there -- only rows that reached the router must
    # be pre_framed. (An earlier version of this check ignored that and
    # discarded a completed 792-image run over 6 quality rejections.)
    routed = frame[frame["outcome"] != PipelineOutcome.QUALITY_REJECTED.value]
    unexpected = routed[routed["framing"] != "pre_framed"]
    if len(unexpected):
        raise RuntimeError(
            f"forcing failed: {len(unexpected)} routed images were not pre_framed"
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    table = pd.read_csv(EXTERNAL_SET)
    total_melanomas = int((table["label"] == "MEL").sum())
    print(f"{len(table)} images, {total_melanomas} melanomas\n")

    arm_a = pd.read_csv(ARM_A).drop_duplicates("image_id")
    print("Arm B: forcing pre_framed for every image...")
    arm_b = run_forced_pre_framed(table, args.device)

    a_n, a_rate = melanoma_reach_rate(arm_a, total_melanomas)
    b_n, b_rate = melanoma_reach_rate(arm_b, total_melanomas)
    a_low, a_high = wilson_interval(a_n, total_melanomas)
    b_low, b_high = wilson_interval(b_n, total_melanomas)
    a_benign, b_benign = benign_referral_rate(arm_a), benign_referral_rate(arm_b)

    delta = (b_rate - a_rate) * 100
    benign_delta = (b_benign - a_benign) * 100

    print("\n" + "=" * 72)
    print("MELANOMA REACHING A CLINICIAN (unconditional, n=%d)" % total_melanomas)
    print("=" * 72)
    print(f"  Arm A  router decides   : {a_n:3}/{total_melanomas} = {a_rate:.4f} "
          f"[{a_low:.2f}, {a_high:.2f}]   benign referral {a_benign:.4f}")
    print(f"  Arm B  forced pre_framed: {b_n:3}/{total_melanomas} = {b_rate:.4f} "
          f"[{b_low:.2f}, {b_high:.2f}]   benign referral {b_benign:.4f}")
    print(f"\n  delta: {delta:+.1f} points melanoma, {benign_delta:+.1f} points benign referral")

    for arm, frame in (("A", arm_a), ("B", arm_b)):
        counts = frame["outcome"].value_counts().to_dict()
        print(f"  arm {arm} outcomes: {counts}")

    if delta >= 5 and benign_delta <= 10:
        verdict = ("ROUTER IS A NET NEGATIVE on this input class -- default close-up "
                   "product input to pre_framed, or retrain CV-1.5 with a third source")
    elif abs(delta) < 5:
        verdict = ("ROUTING IS NOT THE LEVER -- stop here, look at CV-2's "
                   "no-candidate rate instead")
    elif delta <= -5:
        verdict = "WIDE_FIELD BRANCH IS DOING REAL WORK -- leave CV-1.5 alone"
    else:
        verdict = (f"melanoma {delta:+.1f} but benign referral {benign_delta:+.1f} "
                   "exceeds the +10 budget -- gain was bought by referring more benign")
    print(f"\nVERDICT: {verdict}")

    print("\n" + "=" * 72)
    print("DIAGNOSTIC: whole-frame lesion area fraction by what the router said")
    print("=" * 72)
    router_says = arm_a.set_index("image_id")["framing"].to_dict()
    arm_b = arm_b.assign(router_framing=arm_b["image_id"].map(router_says))
    measured = arm_b.dropna(subset=["mask_area_fraction", "router_framing"])
    for framing, subset in measured.groupby("router_framing"):
        print(f"  router said {framing:11} n={len(subset):4}  "
              f"median area fraction {subset['mask_area_fraction'].median():.4f}")
    print(f"  PAD-UFES (true pre-framed) reference median: {PAD_UFES_MEDIAN_AREA_FRACTION:.4f}")
    print("\n  If wide_field-routed images really do have a smaller lesion-to-frame")
    print("  ratio, the router is tracking something real even where it costs outcomes.")


if __name__ == "__main__":
    main()
