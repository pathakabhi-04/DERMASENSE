"""Experiment 3 CV-4 arm: ViT-B/16 on ISIC 2019 (docs/academic_vit_spec.md §3.1).

The CV-4 weighted recipe (configs/cv_resnet50_weighted.yaml on main) with the
spec's one ViT change: backbone lr 3e-5 (head 1e-4), 1-epoch linear warmup,
then constant. AdamW wd 1e-4, batch 32, 10 epochs, sqrt-inverse-frequency
weighted CE, CV-4 augmentation, best val macro-F1, fp16, seed 42. The test
split is never read here (see evaluate_cv4.py).

    PYTHONPATH=. python3 scripts/academic_vit/train_cv4.py \
        --data-root /workspace/dermasense_academic/data/academic_vit_bundle \
        --run-dir   /workspace/dermasense_academic/runs/academic_vit/cv4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.academic.data import IMAGENET_MEAN, IMAGENET_STD, ISIC2019_CLASSES
from src.academic.metrics import macro_f1, sqrt_inverse_frequency_weights
from src.academic.vit import ViTClassifier


def cv4_transform(train: bool) -> transforms.Compose:
    """main's build_train_transform / build_eval_transform with the default config."""
    ops = []
    if train:
        ops = [transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
               transforms.RandomRotation(15.0), transforms.ColorJitter(0.10, 0.10, 0.10, 0.02)]
    return transforms.Compose(ops + [transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class IsicBundle(Dataset):
    def __init__(self, root: Path, split: str, train: bool | None = None) -> None:
        df = pd.read_csv(Path(root) / "dataset.csv")
        self.df = df[(df.source == "isic2019") & (df.split == split)].reset_index(drop=True)
        self.root = Path(root)
        self.targets = [ISIC2019_CLASSES.index(c) for c in self.df.label]
        self.tf = cv4_transform(split == "train" if train is None else train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        with Image.open(self.root / self.df.image[i]) as im:
            return self.tf(im.convert("RGB")), self.targets[i]


def _worker_init(_):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-pretrained", action="store_true", help="Smoke tests only.")
    p.add_argument("--max-train-batches", type=int, default=None, help="Smoke tests only.")
    p.add_argument("--max-val-batches", type=int, default=None, help="Smoke tests only.")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if (a.run_dir / "DONE.json").exists():
        print(f"{a.run_dir} already complete; skipping.")
        return
    a.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"

    train_ds, val_ds = IsicBundle(a.data_root, "train"), IsicBundle(a.data_root, "val")
    weights = sqrt_inverse_frequency_weights(train_ds.targets, len(ISIC2019_CLASSES))

    def sub(ds, n):
        return ds if n is None else Subset(ds, range(min(len(ds), n * a.batch_size)))

    gen = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(sub(train_ds, a.max_train_batches), batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, generator=gen, worker_init_fn=_worker_init,
                              pin_memory=amp, persistent_workers=a.num_workers > 0)
    val_loader = DataLoader(sub(val_ds, a.max_val_batches), batch_size=a.batch_size, num_workers=a.num_workers)

    model = ViTClassifier(pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.param_groups(a.backbone_lr, a.head_lr), weight_decay=a.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ce = nn.CrossEntropyLoss(weight=weights.to(dev))
    warmup_iters = int(a.warmup_epochs * len(train_loader))

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              "device": str(dev), "amp": amp, "warmup_iters": warmup_iters,
              "class_weights": [round(float(w), 6) for w in weights], "torch": torch.__version__}
    (a.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    history, best, start, it = [], float("-inf"), 1, 0
    last = a.run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        gen.set_state(ck["generator"])
        torch.set_rng_state(ck["torch_rng"])
        history, best, start, it = ck["history"], ck["best"], ck["epoch"] + 1, ck["iteration"]
        print(f"Resumed from epoch {ck['epoch']}")

    print(f"device={dev} amp={amp} train={len(train_loader.dataset)} val={len(val_loader.dataset)} warmup_iters={warmup_iters}")
    for epoch in range(start, a.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            factor = min(1.0, 0.001 + (1 - 0.001) * it / warmup_iters) if warmup_iters else 1.0
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * factor
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = ce(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            loss_sum += loss.item() * len(y)
            n += len(y)

        model.eval()
        ys, ps, vloss, vn = [], [], 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    logits = model(x)
                vloss += ce(logits.float(), y).item() * len(y)
                vn += len(y)
                ys.append(y.cpu())
                ps.append(logits.argmax(1).cpu())
        f1 = macro_f1(torch.cat(ys).numpy(), torch.cat(ps).numpy(), len(ISIC2019_CLASSES))
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": vloss / vn, "val_macro_f1": f1,
               "lr_backbone": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        improved = f1 > best
        if improved:
            best = f1
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_f1": f1, "config": config},
                       a.run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "generator": gen.get_state(), "torch_rng": torch.get_rng_state(), "epoch": epoch,
                    "iteration": it, "best": best, "history": history}, last)
        (a.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch:03d}/{a.epochs:03d} train_loss={rec['train_loss']:.4f} val_loss={rec['val_loss']:.4f} "
              f"val_macro_f1={f1:.4f} ({rec['seconds']}s)" + (" *" if improved else ""), flush=True)

    best_epoch = max(history, key=lambda r: r["val_macro_f1"])["epoch"]
    (a.run_dir / "DONE.json").write_text(json.dumps({"best_epoch": best_epoch, "val_macro_f1": best}, indent=2))
    print(f"Done. best epoch {best_epoch}, val_macro_f1={best:.4f}")


if __name__ == "__main__":
    main()
