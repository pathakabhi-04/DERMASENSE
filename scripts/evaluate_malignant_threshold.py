"""
Can routing on summed malignant probability beat routing on argmax?

The pipeline currently routes `probabilities -> argmax -> class ->
action`, which discards the distribution. A melanoma scoring
P(NEV)=0.35, P(MEL)=0.30, P(BCC)=0.20 is routed MONITOR despite
P(malignant)=0.50.

This sweeps a threshold on P(MEL)+P(BCC)+P(SCC) and reports, for each
threshold, melanoma routing (sensitivity) against benign referral (the
cost). No retraining -- same checkpoint, same forward pass.

    PYTHONPATH=. python3 scripts/evaluate_malignant_threshold.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.torch_dataset import CVDatasetTorch
from src.inference.native import NativePredictor

MALIGNANT = ("MEL", "BCC", "SCC")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)
# ISIC labels whose clinical entity is benign. Used only to price the
# cost side; no taxonomy mapping is applied to the data.
BENIGN_ISIC = ("NV", "BKL")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--dataset", default="isic2019")
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path,
                   default=Path("analysis/quality/mel_sensitivity/malignant_threshold.json"))
    return p.parse_args()


def score(predictor, dataset, positions):
    """Summed malignant probability, plus the argmax action, per image."""
    out = []
    for n, i in enumerate(positions, 1):
        with torch.no_grad():
            r = predictor.predict(dataset[i]["image"])
        out.append({
            "p_malignant": sum(r.probabilities.get(c, 0.0) for c in MALIGNANT),
            "argmax_class": r.predicted_class,
            "argmax_routes": r.product_action.value != "MONITOR",
        })
        if n % 250 == 0:
            print(f"    ...{n}/{len(positions)}")
    return out


def main() -> None:
    args = parse_args()
    ds = CVDatasetTorch(dataset_id=args.dataset, split=args.split, verify_images=False)

    mel = [i for i in range(len(ds)) if ds.get_diagnosis(i) == "MEL"]
    benign = [i for i in range(len(ds)) if ds.get_diagnosis(i) in BENIGN_ISIC]
    print(f"  melanomas {len(mel)}   benign (NV/BKL) {len(benign)}")

    predictor = NativePredictor.from_checkpoint(args.checkpoint, device="cpu")
    print("  scoring melanomas...");  mel_s = score(predictor, ds, mel)
    print("  scoring benign...");     ben_s = score(predictor, ds, benign)

    base_sens = sum(s["argmax_routes"] for s in mel_s) / len(mel_s)
    base_refer = sum(s["argmax_routes"] for s in ben_s) / len(ben_s)

    print("\n" + "=" * 72)
    print(f"  BASELINE (argmax):  melanoma routed {base_sens:.3f}   "
          f"benign referred {base_refer:.3f}")
    print("=" * 72)
    print(f"  {'threshold':>9}  {'MEL routed':>11}  {'benign referred':>16}  {'vs baseline':>12}")
    print("  " + "-" * 66)

    rows = []
    for t in [i / 20 for i in range(1, 20)]:
        sens = sum(s["p_malignant"] >= t for s in mel_s) / len(mel_s)
        refer = sum(s["p_malignant"] >= t for s in ben_s) / len(ben_s)
        rows.append({"threshold": t, "mel_routed": sens, "benign_referred": refer})
        flag = "  <-- meets rule" if sens >= 0.90 and refer <= 0.50 else ""
        print(f"  {t:>9.2f}  {sens:>11.3f}  {refer:>16.3f}  "
              f"{sens - base_sens:>+11.3f}{flag}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"baseline": {"mel_routed": base_sens, "benign_referred": base_refer},
         "sweep": rows}, indent=2))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
