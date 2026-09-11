"""
Melanoma behaviour of the DEPLOYED (PAD-UFES-transferred) checkpoint,
measured on a sample large enough to mean something.

Companion to `analysis/quality/mel_sensitivity/isic_in_domain_result.md`,
which measured the ISIC-trained 8-class model in-domain: MEL recall
0.5661, with 35.4% of melanomas routing to MONITOR.

This measures the checkpoint the pipeline actually ships
(`pad_ufes_c1_partial_finetune_seed42`, 6-class) on the same 666
melanomas.

## What this is and is not

The 6-class head cannot emit ISIC's 8 labels, so this does NOT compute
8-class accuracy and does NOT invent a taxonomy mapping (still jointly
open -- see docs/cv8_sample_outputs/README.md). It asks one question
that needs no mapping:

    Of images a dermatologist labelled MELANOMA, what product action
    does the shipped model produce?

MEL ground truth is unambiguous. Predictions live in the 6-class space
and map to actions through src/risk/action_mapping.py, which is the
mapping the product already uses.

## The honest caveat

This is CROSS-DOMAIN: the checkpoint was fine-tuned on PAD-UFES clinical
photographs and is evaluated here on ISIC dermoscopy. A low number
therefore confounds two things -- what the PAD transfer cost, and what
the domain shift costs. It is a LOWER bound on the checkpoint's
melanoma behaviour in its own domain.

The alternative, PAD-UFES's own test split, has **9 melanomas**, whose
95% CI spans roughly 30-93%. A confounded measurement on 666 is more
informative than an unconfounded one on 9.

    PYTHONPATH=. python3 scripts/evaluate_pad_checkpoint_on_isic_mel.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from src.data.torch_dataset import CVDatasetTorch
from src.inference.native import NativePredictor
from src.risk.action_mapping import ProductAction, diagnosis_to_action

DEFAULT_CHECKPOINT = Path(
    "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)

# Actions that send the user to a clinician. MONITOR does not.
SEES_CLINICIAN = {
    ProductAction.URGENT_EVALUATION,
    ProductAction.EVALUATE_SOON,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Melanoma routing behaviour of a PAD-UFES checkpoint on ISIC."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", default="isic2019")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = CVDatasetTorch(
        dataset_id=args.dataset, split=args.split, verify_images=False
    )
    class_names = list(dataset.class_names)
    if "MEL" not in class_names:
        raise SystemExit(f"No MEL class in {args.dataset}/{args.split}: {class_names}")
    mel_index = class_names.index("MEL")

    # get_diagnosis() reads metadata only -- it does not decode the image,
    # so selecting the melanomas costs nothing.
    mel_positions = [
        i for i in range(len(dataset)) if dataset.get_diagnosis(i) == "MEL"
    ]

    print(f"Dataset    : {args.dataset}/{args.split}  ({len(dataset)} images)")
    print(f"Melanomas  : {len(mel_positions)}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Domain     : CROSS-DOMAIN (PAD-UFES clinical -> ISIC dermoscopy)\n")

    predictor = NativePredictor.from_checkpoint(args.checkpoint, device=args.device)

    predicted = Counter()
    actions = Counter()

    for seen, position in enumerate(mel_positions, start=1):
        sample = dataset[position]
        with torch.no_grad():
            result = predictor.predict(sample["image"])
        predicted[result.predicted_class] += 1
        actions[result.product_action] += 1
        if seen % 100 == 0:
            print(f"  ...{seen}/{len(mel_positions)}")

    total = len(mel_positions)
    print("\n" + "=" * 66)
    print("PREDICTED CLASS for images a dermatologist labelled MELANOMA")
    print("=" * 66)
    for name, count in predicted.most_common():
        print(f"  {name:5} {count:5}  ({100 * count / total:5.1f}%)")

    print("\n" + "=" * 66)
    print("PRODUCT ACTION (src/risk/action_mapping.py)")
    print("=" * 66)
    for action, count in actions.most_common():
        print(f"  {action.value:20} {count:5}  ({100 * count / total:5.1f}%)")

    to_clinician = sum(c for a, c in actions.items() if a in SEES_CLINICIAN)
    monitored = actions.get(ProductAction.MONITOR, 0)

    print("\n" + "=" * 66)
    print(f"  MEL recall (predicted MEL) : {predicted.get('MEL', 0)}/{total} = "
          f"{predicted.get('MEL', 0) / total:.4f}")
    print(f"  Sent to a clinician        : {to_clinician}/{total} = "
          f"{to_clinician / total:.4f}")
    print(f"  Told 'low risk' (MONITOR)  : {monitored}/{total} = "
          f"{monitored / total:.4f}   <-- the number that matters")
    print("=" * 66)


if __name__ == "__main__":
    main()
