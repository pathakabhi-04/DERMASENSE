"""
CV-1 -> CV-4 assembled pipeline evaluation.

Two branches, two different kinds of measurement -- see
docs/cv1_cv4_assembly_spec.md.

  pre_framed (PAD-UFES test, 352 images, has diagnosis labels)
      Full product metrics, reusing the existing Tier-1..4 taxonomy.
      This is a REGRESSION CHECK, not a new accuracy claim: the
      assembled pipeline must not degrade CV-4's known standalone
      baseline (Macro-F1 0.5996, 32/352 Tier-1 errors). Because the
      pre-framed branch feeds CV-4 essentially the whole image, results
      should be near-identical; a material deviation IS the finding.

  wide_field (iToBoS test, NO diagnosis labels)
      Structural propagation only -- quality-rejection rate, routing
      distribution, zero-candidate (silent-miss) rate, candidates per
      image, action distribution. NO accuracy claim is made, because
      there is no ground truth to make one against.

Usage:
    python -m scripts.evaluate_pipeline_end_to_end --split pad_ufes
    python -m scripts.evaluate_pipeline_end_to_end --split itobos --limit 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch

from src.inference.orchestrator import DermaSensePipeline, PipelineOutcome
from src.risk.action_mapping import diagnosis_to_action

REPO_ROOT = Path(__file__).resolve().parents[1]

PAD_UFES_TEST = REPO_ROOT / "data/splits/pad_ufes/test.csv"
ITOBOS_TEST = REPO_ROOT / "data/splits/itobos_detection/test.csv"

ROUTER_CHECKPOINT = REPO_ROOT / "checkpoints/cv1_5_router/best.pt"
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)
SECOND_CLASSIFIER_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed123_best.pt"
)
DETECTOR_WEIGHTS = REPO_ROOT / "runs/cv2/b1_1280/weights/best.pt"

OUTPUT_DIR = REPO_ROOT / "analysis/product_eval/cv1_cv4_assembly"

# Reused verbatim from scripts/analyze_phase4_safety_bottleneck.py so the
# regression comparison is apples-to-apples with the CV-4 baseline.
HIGH_RISK = frozenset({"BCC", "MEL", "SCC"})

# CV-4 standalone per-image predictions, used as the regression baseline.
# Comparing against a stored *number* would be survivorship-biased: the
# assembly can drop images at CV-1/CV-2, so its metrics cover a subset.
# The baseline must be recomputed on exactly the images the assembly
# actually scored.
BASELINE_PREDICTIONS = (
    REPO_ROOT / "analysis/product_eval/c1_f1_test_predictions.csv"
)
BASELINE_PREDICTION_COLUMN = "c1_pred"


def classify_error(true_class: str, predicted_class: str) -> str:
    if true_class == predicted_class:
        return "CORRECT"
    if true_class in HIGH_RISK:
        if predicted_class not in HIGH_RISK:
            return "TIER_1"
        return "TIER_2"
    if predicted_class in HIGH_RISK:
        return "TIER_3"
    return "TIER_4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the assembled CV-1 -> CV-4 pipeline."
    )
    p.add_argument(
        "--split", choices=["pad_ufes", "itobos"], default="pad_ufes"
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "cap images processed (0 = all). iToBoS test is 8481 images. "
            "Sampled at random with --seed rather than taken from the "
            "head, so the subset is not correlated with acquisition order."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--ensemble",
        action="store_true",
        help=(
            "run the seed123 checkpoint alongside seed42 for CV-6 "
            "ensemble-disagreement evidence (docs/cv6_uncertainty_spec.md). "
            "Off by default -- roughly doubles CV-4 inference cost."
        ),
    )
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def macro_f1(frame: pd.DataFrame) -> float:
    """Macro-F1 over the classes present in the ground truth."""
    scores = []
    for class_name in sorted(frame["true_class"].unique()):
        tp = int(
            (
                (frame["true_class"] == class_name)
                & (frame["predicted_class"] == class_name)
            ).sum()
        )
        fp = int(
            (
                (frame["true_class"] != class_name)
                & (frame["predicted_class"] == class_name)
            ).sum()
        )
        fn = int(
            (
                (frame["true_class"] == class_name)
                & (frame["predicted_class"] != class_name)
            ).sum()
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        scores.append(f1)
    return float(sum(scores) / len(scores)) if scores else 0.0


def run_split(
    pipeline: DermaSensePipeline, table: pd.DataFrame, has_labels: bool
) -> pd.DataFrame:
    records = []

    for position, (_, row) in enumerate(table.iterrows(), start=1):
        image_path = REPO_ROOT / row["image_path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        result = pipeline.predict(image_bgr)

        base = {
            "image_id": row.get("image_id", image_path.stem),
            "outcome": result.outcome.value,
            "framing": result.framing,
            "quality_usable": result.quality.usable,
            "quality_score": round(result.quality.quality_score, 4),
            "num_candidates": len(result.candidates),
            "image_action": result.product_action.value,
            "requires_review": result.requires_review,
            "num_suggestions": len(result.suggestions),
        }

        if has_labels:
            true_class = str(row["native_diagnosis"]).strip().upper()
            base["true_class"] = true_class
            base["true_action"] = diagnosis_to_action(true_class).value

        if result.candidates:
            # One row per candidate; the image-level fields repeat.
            for candidate in result.candidates:
                record = dict(base)
                record.update(candidate.to_dict())
                if has_labels:
                    record["error_tier"] = classify_error(
                        record["true_class"], candidate.predicted_class
                    )
                records.append(record)
        else:
            record = dict(base)
            record["predicted_class"] = None
            if has_labels:
                # An unassessed high-risk lesion is the worst case, and
                # it is not a Tier-1 misclassification -- it is a
                # non-assessment. Kept as its own category so it can
                # never be averaged into classification accuracy.
                record["error_tier"] = "NOT_ASSESSED"
            records.append(record)

        if position % 50 == 0:
            print(f"  {position}/{len(table)} images")

    return pd.DataFrame(records)


def _tier1_count(frame: pd.DataFrame) -> int:
    return sum(
        classify_error(row.true_class, row.predicted_class) == "TIER_1"
        for row in frame.itertuples()
    )


def summarize_pre_framed(results: pd.DataFrame) -> str:
    assessed = results[
        results["outcome"] == PipelineOutcome.ASSESSED.value
    ].dropna(subset=["predicted_class"])
    per_image = results.drop_duplicates("image_id")

    images = per_image["image_id"].nunique()
    tier_counts = results["error_tier"].value_counts().to_dict()
    not_assessed = int(tier_counts.get("NOT_ASSESSED", 0))

    scored = assessed[["image_id", "true_class", "predicted_class"]]
    f1 = macro_f1(scored)
    tier1 = _tier1_count(scored)

    lines = [
        "CV-1 -> CV-4 Assembly -- Pre-framed branch (PAD-UFES test)",
        "=" * 62,
        "",
        f"Images: {images}",
        f"Assessed: {len(scored)}   Not assessed: {not_assessed}",
        "",
    ]

    # Regression check on exactly the images the assembly scored.
    if BASELINE_PREDICTIONS.exists():
        baseline = pd.read_csv(BASELINE_PREDICTIONS).rename(
            columns={BASELINE_PREDICTION_COLUMN: "predicted_class"}
        )[["image_id", "true_class", "predicted_class"]]
        baseline_same = baseline[
            baseline["image_id"].isin(set(scored["image_id"]))
        ]

        merged = baseline_same.merge(
            scored, on="image_id", suffixes=("_base", "_asm")
        )
        agreement = (
            merged["predicted_class_base"] == merged["predicted_class_asm"]
        ).mean()

        lines += [
            "REGRESSION CHECK vs CV-4 standalone, same images only",
            "(comparing against a whole-set baseline number would be",
            " survivorship-biased -- the assembly can drop images at CV-1)",
            "",
            f"  {'':<18}{'assembly':>10}{'CV-4 alone':>13}",
            f"  {'Macro-F1':<18}{f1:>10.4f}{macro_f1(baseline_same):>13.4f}",
            f"  {'Tier-1 errors':<18}{tier1:>10}{_tier1_count(baseline_same):>13}",
            "",
            f"  Per-image agreement: {agreement:.1%}",
            f"  Verdict: {'NO DEGRADATION' if agreement >= 0.99 else 'DIVERGENCE -- investigate'}",
            "",
        ]
    else:
        lines += [
            f"Macro-F1 {f1:.4f}   Tier-1 {tier1}",
            "(baseline predictions unavailable; no regression check run)",
            "",
        ]

    lines.append(
        "Error tiers (Tier-1 = high-risk predicted as non-high-risk):"
    )
    for tier in sorted(tier_counts):
        lines.append(f"  {tier:<14} {tier_counts[tier]}")

    # Non-assessment is the outcome that never reaches a diagnosis, so
    # report its composition rather than letting it vanish from metrics.
    dropped = per_image[per_image["outcome"] != PipelineOutcome.ASSESSED.value]
    if len(dropped):
        high_risk_dropped = int(dropped["true_class"].isin(HIGH_RISK).sum())
        base_rate = per_image["true_class"].isin(HIGH_RISK).mean()
        lines += [
            "",
            f"Not assessed: {len(dropped)}/{images} ({len(dropped)/images:.1%})",
            f"  by outcome: {dropped['outcome'].value_counts().to_dict()}",
            f"  high-risk among dropped: {high_risk_dropped}"
            f" ({high_risk_dropped/len(dropped):.1%};"
            f" base rate {base_rate:.1%})",
        ]

    lines += [
        "",
        "Note: the pre-framed branch does not exercise CV-2. It routes",
        "straight to CV-3 -> CV-4, which is why this is a regression",
        "check on CV-4's known behaviour rather than a new claim.",
    ]
    return "\n".join(lines)


def summarize_wide_field(results: pd.DataFrame) -> str:
    per_image = results.drop_duplicates("image_id")
    total = len(per_image)

    outcome_counts = per_image["outcome"].value_counts().to_dict()
    routing_counts = per_image["framing"].value_counts(dropna=False).to_dict()
    action_counts = per_image["image_action"].value_counts().to_dict()

    no_candidates = int(
        outcome_counts.get(PipelineOutcome.NO_CANDIDATES.value, 0)
    )
    assessed = per_image[
        per_image["outcome"] == PipelineOutcome.ASSESSED.value
    ]

    lines = [
        "CV-1 -> CV-4 Assembly -- Wide-field branch (iToBoS test)",
        "=" * 62,
        "",
        "NO ACCURACY CLAIM IS MADE. iToBoS carries no diagnosis labels,",
        "so nothing here is scored against ground truth. These are",
        "structural propagation counts only.",
        "",
        f"Images: {total}",
        "",
        "Outcomes:",
    ]
    for outcome in sorted(outcome_counts):
        share = outcome_counts[outcome] / total
        lines.append(f"  {outcome:<18} {outcome_counts[outcome]:>5}  ({share:.1%})")

    lines += [
        "",
        f"Silent-miss rate (routed wide_field, zero candidates): "
        f"{no_candidates / total:.1%}",
        "",
        "Routing distribution:",
    ]
    for framing in sorted(routing_counts, key=lambda value: str(value)):
        lines.append(f"  {str(framing):<18} {routing_counts[framing]}")

    if len(assessed):
        lines += [
            "",
            f"Candidates per assessed image: "
            f"mean {assessed['num_candidates'].mean():.2f}, "
            f"max {int(assessed['num_candidates'].max())}",
        ]

    lines += ["", "Image-level action distribution:"]
    for action in sorted(action_counts):
        lines.append(f"  {action:<20} {action_counts[action]}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    is_pad_ufes = args.split == "pad_ufes"
    table = pd.read_csv(PAD_UFES_TEST if is_pad_ufes else ITOBOS_TEST)
    if not is_pad_ufes:
        table = table.drop_duplicates("image_id")
    if args.limit and args.limit < len(table):
        # Random, seeded -- a head() slice could track acquisition order
        # (site, batch, body region) and bias the structural rates.
        table = table.sample(n=args.limit, random_state=args.seed)

    # The detector is only needed for the wide-field branch.
    detector_weights = None if is_pad_ufes else DETECTOR_WEIGHTS
    if detector_weights is not None and not Path(detector_weights).exists():
        raise FileNotFoundError(
            f"CV-2 weights not found: {detector_weights}"
        )

    print(f"Device: {device}   Split: {args.split}   Images: {len(table)}")

    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        additional_ensemble_checkpoints=(
            (SECOND_CLASSIFIER_CHECKPOINT,) if args.ensemble else None
        ),
        detector_weights=detector_weights,
        device=device,
    )

    results = run_split(pipeline, table, has_labels=is_pad_ufes)

    prefix = "pad_ufes" if is_pad_ufes else "itobos"
    predictions_path = args.output / f"{prefix}_predictions.csv"
    summary_path = args.output / f"{prefix}_summary.txt"

    results.to_csv(predictions_path, index=False)

    summary = (
        summarize_pre_framed(results)
        if is_pad_ufes
        else summarize_wide_field(results)
    )
    summary_path.write_text(summary + "\n")

    print()
    print(summary)
    print()
    print(f"Predictions: {predictions_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
