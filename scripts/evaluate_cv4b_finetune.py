"""
The single, final test-set evaluation for a CV-4b fine-tune run
(design §8). This is the ONLY script that reads a test split.

It takes the checkpoint that was selected on validation and reports the
three pre-committed criteria, then states pass/fail against them without
re-deciding what they are.

    PYTHONPATH=. python3 scripts/evaluate_cv4b_finetune.py --fold pooled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from scripts.finetune_cv4b_backbone import (
    DEFAULT_BUNDLE,
    FOLDS,
    RUN_ROOT,
    BinaryLesionDataset,
    Row,
    build_model,
    bundle_rows,
)
from src.data.transforms import build_eval_transform
from src.training.metrics import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "analysis/quality/mel_sensitivity"

# design §8 -- all three must hold. Copied here as constants so the
# evaluation cannot quietly drift from what was pre-committed.
LOSO_AUC_TARGET = 0.80
# The multi-domain run pre-committed a different bar
# (docs/cv4b_domain_generalization_design.md §6): 0.75, because its claim
# is "domain count is a mechanism", not "this ships". dg_pad is reported
# but NOT gated -- 9 melanomas is the measurement problem that started
# this whole line of work.
DG_AUC_TARGET = 0.75
DG_UNGATED_FOLDS = ("dg_pad",)
ISIC_AUC_FLOOR = 0.85
BENIGN_REFERRAL_CEILING = 0.45
TARGET_BENIGN_REFERRAL = 0.40  # operating-point budget, chosen on val


@torch.no_grad()
def score_rows(model, head, rows: list[Row], device, batch_size: int, workers: int) -> np.ndarray:
    model.eval()
    head.eval()
    loader = DataLoader(
        BinaryLesionDataset(rows, build_eval_transform()),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
    )
    scores = []
    for batch in loader:
        features = model.extract_features(batch["image"].to(device, non_blocking=True))
        scores.append(torch.sigmoid(head["pooled"](features).squeeze(1)).cpu().numpy())
    return np.concatenate(scores)


def summarise(rows: list[Row], scores: np.ndarray, threshold: float, label: str) -> dict:
    labels = np.array([row.label for row in rows])
    melanoma = np.array([row.is_melanoma for row in rows])
    referred = scores >= threshold

    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    benign_referred = float(referred[labels == 0].mean()) if (labels == 0).any() else float("nan")
    n_melanoma = int(melanoma.sum())
    if n_melanoma:
        melanoma_routed = float(referred[melanoma].mean())
        low, high = wilson_interval(int(referred[melanoma].sum()), n_melanoma)
    else:
        melanoma_routed, low, high = float("nan"), 0.0, 0.0

    print(f"  {label:<24} n={len(rows):>5}  AUC={auc:.4f}  "
          f"benign_referred={benign_referred:.4f}  "
          f"mel_routed={melanoma_routed:.4f} [{low:.2f},{high:.2f}] (n_mel={n_melanoma})")
    return {"split": label, "n": len(rows), "auc": auc, "benign_referred": benign_referred,
            "melanoma_routed": melanoma_routed, "melanoma_ci": [low, high],
            "n_melanoma": n_melanoma}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="pooled", choices=FOLDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR,
                        help="repo copy of the result JSON; the authoritative copy is always "
                             "written to <run-root>/<fold>/result.json as well")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    best_path = args.run_root / args.fold / "best.pt"
    if not best_path.exists():
        raise SystemExit(f"no selected checkpoint at {best_path}; run the fine-tune first")

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model, head, _ = build_model(device, args.data_root)
    model.load_state_dict(payload["model"])
    # `head` is a ModuleDict (pooled + per-domain auxiliaries); only the
    # pooled head is scored. build_model already constructed the right
    # shape, so load into it rather than rebuilding one by hand.
    head.load_state_dict(payload["head"])
    print(f"loaded epoch {payload['epoch']} (selected on {payload['selection_metric']}, "
          f"val {payload['metrics']['non_isic_auc']:.4f})")

    val_rows = bundle_rows(args.data_root, args.fold, "val")
    test_rows = bundle_rows(args.data_root, args.fold, "test")
    isic_test = [row for row in test_rows if row.source == "isic2019"]
    non_isic_test = [row for row in test_rows if row.source != "isic2019"]

    # Operating point comes from VAL, never from test.
    val_scores = score_rows(model, head, val_rows, device, args.batch_size, args.workers)
    val_labels = np.array([row.label for row in val_rows])
    threshold = float(np.quantile(val_scores[val_labels == 0], 1 - TARGET_BENIGN_REFERRAL))
    print(f"operating point from val ({TARGET_BENIGN_REFERRAL:.0%} benign budget): {threshold:.4f}")

    print("\nTEST (first and only use):")
    isic_result = summarise(
        isic_test, score_rows(model, head, isic_test, device, args.batch_size, args.workers),
        threshold, "ISIC (in-domain)",
    )
    non_isic_scores = score_rows(model, head, non_isic_test, device, args.batch_size, args.workers)
    non_isic_result = summarise(non_isic_test, non_isic_scores, threshold, "non-ISIC (combined)")

    per_source = []
    sources = np.array([row.source for row in non_isic_test])
    for source in sorted(set(sources)):
        mask = sources == source
        per_source.append(summarise(
            [row for row, keep in zip(non_isic_test, mask) if keep],
            non_isic_scores[mask], threshold, source,
        ))

    is_dg = args.fold.startswith("dg_")
    target = DG_AUC_TARGET if is_dg else LOSO_AUC_TARGET
    criteria = {
        f"held-out-domain AUC >= {target}": non_isic_result["auc"] >= target,
        f"ISIC test AUC >= {ISIC_AUC_FLOOR}":
            isic_result["auc"] >= ISIC_AUC_FLOOR,
        f"held-out benign referral <= {BENIGN_REFERRAL_CEILING}":
            non_isic_result["benign_referred"] <= BENIGN_REFERRAL_CEILING,
    }
    if args.fold in DG_UNGATED_FOLDS:
        print(f"\nNOTE: {args.fold} is REPORTED, NOT GATED (design §5) -- "
              f"{non_isic_result['n_melanoma']} melanomas is too few to gate on.")
    print("\n" + "=" * 72)
    for name, met in criteria.items():
        print(f"  {'MET    ' if met else 'NOT MET'}  {name}")
    success = all(criteria.values())
    print(f"OVERALL: {'SUCCESS' if success else 'NOT MET'} "
          f"({'all three' if success else 'see design §8 for what each outcome means'})")
    print("=" * 72)

    document = json.dumps({
        "fold": args.fold, "epoch": int(payload["epoch"]), "threshold": threshold,
        "isic_test": isic_result, "non_isic_test": non_isic_result,
        "per_source": per_source,
        "criteria": {name: bool(met) for name, met in criteria.items()},
        "success": success,
    }, indent=2)

    # Written beside the checkpoint FIRST and always. On a rented pod the
    # repo clone may sit on container-local disk that dies with the pod,
    # and this JSON is the actual deliverable of the whole exercise --
    # losing it would mean re-running the GPU job to recover a number that
    # had already been computed.
    run_copy = args.run_root / args.fold / "result.json"
    run_copy.parent.mkdir(parents=True, exist_ok=True)
    run_copy.write_text(document)
    print(f"\nresult -> {run_copy}  (beside the checkpoint; survives the pod)")

    repo_copy = args.result_dir / f"cv4b_finetune_{args.fold}_result.json"
    if repo_copy.resolve() != run_copy.resolve():
        repo_copy.parent.mkdir(parents=True, exist_ok=True)
        repo_copy.write_text(document)
        print(f"result -> {repo_copy}  (repo copy, for committing)")


if __name__ == "__main__":
    main()
