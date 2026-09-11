"""
Cache ISIC2019 backbone features for the binary referral head.

Same 2048-d penultimate representation, same deployed checkpoint, as the
cached test-split features -- so a probe fitted on one split can be
evaluated on the other without a representation mismatch.

    PYTHONPATH=. python3 scripts/extract_isic_features.py --split val
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import CVDatasetTorch
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)

CHECKPOINT = Path("checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt")
OUT_DIR = Path("analysis/scc_bcc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--dataset", default="isic2019")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def collate(batch):
    return (
        torch.stack([b["image"] for b in batch]),
        torch.tensor([b["target"] for b in batch]),
    )


def main() -> None:
    args = parse_args()
    dataset = CVDatasetTorch(
        dataset_id=args.dataset, split=args.split, verify_images=False
    )
    print(f"  {args.dataset}/{args.split}: {len(dataset)} images")

    model = DermaSenseNativeClassifier(
        NativeClassifierConfig(backbone="resnet50", pretrained=False, dropout=0.0)
    )
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck.get("state_dict")))
    model.eval()

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate,
    )

    feats, targs = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for n, (images, targets) in enumerate(loader, 1):
            f = model.forward_features(images)
            if f.ndim > 2:
                f = torch.flatten(f, start_dim=1)
            feats.append(f.cpu().numpy())
            targs.append(targets.numpy())
            if n % 20 == 0:
                done = n * args.batch_size
                rate = done / (time.perf_counter() - start)
                print(f"    {min(done, len(dataset))}/{len(dataset)}  "
                      f"{rate:.1f} img/s  ~{(len(dataset)-done)/max(rate,1e-9)/60:.1f} min left",
                      flush=True)

    features = np.concatenate(feats).astype(np.float32)
    targets = np.concatenate(targs).astype(np.int64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.dataset}_{args.split}_backbone_features.npz"
    np.savez_compressed(out, features=features, targets=targets)
    print(f"\n  wrote {out}  {features.shape}  in {(time.perf_counter()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
