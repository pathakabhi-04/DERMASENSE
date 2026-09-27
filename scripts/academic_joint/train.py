"""Train one (arm, seed) run of the joint seg+cls experiment.

docs/academic_joint_seg_cls_spec.md, Section 5 (arms) and 5.1 (settings).

    PYTHONPATH=. python3 scripts/academic_joint/train.py --arm J --seed 42 \
        --data-root /workspace/dermasense_academic/data/academic_joint_bundle \
        --run-root  /workspace/dermasense_academic/runs/academic_joint

Writes <run-root>/<arm>_seed<seed>/{last.pt,best.pt,history.json,config.json,DONE.json}.
last.pt is written every epoch and a rerun resumes from it; a run with
DONE.json is skipped. The test split is never read here.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.academic.data import ISIC2019_CLASSES, HamJointDataset, SegTestDataset
from src.academic.metrics import BCEDiceLoss, macro_f1, per_image_dice, sqrt_inverse_frequency_weights
from src.academic.model import ARMS, AcademicModel

LAMBDA_SEG = 1.0  # fixed by the spec, not swept


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--dataset", choices=("ham", "isic2018"), default="ham",
                   help="isic2018 (arm S only) is the Experiment 2 CV-3 run.")
    p.add_argument("--augment", choices=("cv4", "none"), default="cv4")
    p.add_argument("--backbone", choices=("resnet50", "vit_b_16"), default="resnet50",
                   help="vit_b_16 (arm S only) is the Experiment 3 CV-3 run.")
    p.add_argument("--encoder-lr", type=float, default=None,
                   help="Separate lr for the pretrained trunk (Experiment 3); default: --learning-rate for all.")
    p.add_argument("--warmup-epochs", type=float, default=0.0,
                   help="Linear warmup from 0.001x, per iteration (Experiment 3); default none.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--no-pretrained", action="store_true", help="Smoke tests only.")
    p.add_argument("--max-train-batches", type=int, default=None, help="Smoke tests only.")
    p.add_argument("--max-val-batches", type=int, default=None, help="Smoke tests only.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def make_loader(ds, *, batch_size, num_workers, shuffle, generator=None, max_batches=None) -> DataLoader:
    if max_batches is not None:
        ds = Subset(ds, range(min(len(ds), max_batches * batch_size)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init,
        generator=generator,
        drop_last=False,
    )


def compute_loss(arm, out, batch, ce, seg_loss):
    parts = {}
    if arm in ("C", "J"):
        parts["cls"] = ce(out["cls"].float(), batch["target"])
    if arm in ("S", "J"):
        parts["seg"] = seg_loss(out["seg"], batch["mask"])
    total = parts.get("cls", 0.0) + LAMBDA_SEG * parts.get("seg", 0.0)
    return total, parts


@torch.no_grad()
def validate(model, loader, device, arm, ce, seg_loss, use_amp) -> dict:
    model.eval()
    loss_sum, n = 0.0, 0
    preds, trues, dices = [], [], []
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch["image"])
        loss, _ = compute_loss(arm, out, batch, ce, seg_loss)
        bs = batch["image"].shape[0]
        loss_sum += loss.detach().item() * bs
        n += bs
        if "cls" in out:
            preds.append(out["cls"].argmax(1).cpu())
            trues.append(batch["target"].cpu())
        if "seg" in out:
            dices.append(per_image_dice(out["seg"], batch["mask"]).cpu())
    result = {"val_loss": loss_sum / n}
    if preds:
        result["val_macro_f1"] = macro_f1(torch.cat(trues).numpy(), torch.cat(preds).numpy(), len(ISIC2019_CLASSES))
    if dices:
        result["val_dice"] = float(torch.cat(dices).mean())
    return result


def main() -> None:
    args = parse_args()
    run_dir = args.run_root / f"{args.arm}_seed{args.seed}"
    if (run_dir / "DONE.json").exists():
        print(f"{run_dir} already complete; skipping.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    use_amp = device.type == "cuda"
    select_key = "val_dice" if args.arm == "S" else "val_macro_f1"

    augment = args.augment == "cv4"
    if args.dataset == "ham":
        train_ds = HamJointDataset(args.data_root, "train", image_size=args.image_size, train=augment)
        val_ds = HamJointDataset(args.data_root, "val", image_size=args.image_size)
    else:
        if args.arm != "S":
            raise SystemExit("--dataset isic2018 has masks only; use --arm S")
        train_ds = SegTestDataset(args.data_root, image_size=args.image_size, split="train", augment=augment)
        val_ds = SegTestDataset(args.data_root, image_size=args.image_size, split="val")

    # A dedicated generator fixes shuffling and per-worker augmentation seeds,
    # so for a given seed all three arms see the same batches and augmentations.
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                               shuffle=True, generator=gen, max_batches=args.max_train_batches)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, max_batches=args.max_val_batches)

    model = AcademicModel(args.arm, pretrained=not args.no_pretrained, seed=args.seed,
                          backbone=args.backbone, image_size=args.image_size).to(device)
    class_weights = (sqrt_inverse_frequency_weights(train_ds.targets, len(ISIC2019_CLASSES))
                     if args.arm != "S" else torch.ones(len(ISIC2019_CLASSES)))
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    seg_loss = BCEDiceLoss()
    params = (model.param_groups(args.encoder_lr, args.learning_rate) if args.encoder_lr is not None
              else model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    warmup_iters = int(args.warmup_epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "lambda_seg": LAMBDA_SEG,
        "select_key": select_key,
        "device": str(device),
        "use_amp": use_amp,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "class_weights": [round(float(w), 6) for w in class_weights],
        "torch": torch.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    history: list[dict] = []
    best, start_epoch = float("-inf"), 1
    last_path = run_dir / "last.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        gen.set_state(ckpt["generator"])
        torch.set_rng_state(ckpt["torch_rng"])
        history, best, start_epoch = ckpt["history"], ckpt["best"], ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best {select_key}={best:.4f})")
    it = (start_epoch - 1) * len(train_loader)

    print(f"arm={args.arm} seed={args.seed} device={device} amp={use_amp} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} select={select_key}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        part_sums: dict[str, float] = {}
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if warmup_iters:
                factor = min(1.0, 0.001 + (1 - 0.001) * it / warmup_iters)
                for g, lr in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = lr * factor
            it += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch["image"])
            loss, parts = compute_loss(args.arm, out, batch, ce, seg_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = batch["image"].shape[0]
            loss_sum += loss.detach().item() * bs
            n += bs
            for k, v in parts.items():
                part_sums[k] = part_sums.get(k, 0.0) + v.detach().item() * bs

        record = {"epoch": epoch, "train_loss": loss_sum / n,
                  **{f"train_loss_{k}": v / n for k, v in part_sums.items()}}
        record.update(validate(model, val_loader, device, args.arm, ce, seg_loss, use_amp))
        record["seconds"] = round(time.time() - t0, 1)
        history.append(record)

        improved = record[select_key] > best
        if improved:
            best = record[select_key]
            torch.save({"model": model.state_dict(), "arm": args.arm, "seed": args.seed, "epoch": epoch,
                        select_key: best, "config": config}, run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "generator": gen.get_state(),
                    "torch_rng": torch.get_rng_state(), "epoch": epoch, "best": best,
                    "history": history}, last_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        shown = " ".join(f"{k}={v:.4f}" for k, v in record.items() if k not in ("epoch", "seconds"))
        print(f"Epoch {epoch:03d}/{args.epochs:03d} {shown} ({record['seconds']}s)"
              + (" *" if improved else ""), flush=True)

    best_epoch = max(history, key=lambda r: r[select_key])["epoch"]
    (run_dir / "DONE.json").write_text(json.dumps(
        {"arm": args.arm, "seed": args.seed, "best_epoch": best_epoch, select_key: best,
         "epochs": args.epochs}, indent=2))
    print(f"Done. best epoch {best_epoch}, {select_key}={best:.4f}")


if __name__ == "__main__":
    main()
