"""Train the CV-2 Faster R-CNN (ResNet-50 FPN v2) on iToBoS train.

docs/academic_resnet50_three_task_spec.md, Sections 3 and 3.1. The final
epoch is the model; val is not used during training (it is the evaluation
set). last.pt is written every epoch, and a rerun resumes from it.

    PYTHONPATH=. python3 scripts/academic_r50/train_cv2.py \
        --data-root /workspace/dermasense_academic/data/academic_r50_bundle \
        --run-dir   /workspace/dermasense_academic/runs/academic_r50/cv2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.academic.detection import ITobosDataset, build_detector, collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--milestones", type=int, nargs=2, default=(31, 42))
    p.add_argument("--warmup-iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-pretrained", action="store_true", help="Smoke tests only.")
    p.add_argument("--max-train-batches", type=int, default=None, help="Smoke tests only.")
    return p.parse_args()


def _worker_init(_):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def main() -> None:
    args = parse_args()
    if (args.run_dir / "DONE.json").exists():
        print(f"{args.run_dir} already complete; skipping.")
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    ds = ITobosDataset(args.data_root, "train", train=True)
    if args.max_train_batches is not None:
        ds = Subset(ds, range(min(len(ds), args.max_train_batches * args.batch_size)))
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate, worker_init_fn=_worker_init, generator=gen,
                        pin_memory=use_amp, persistent_workers=args.num_workers > 0)

    model = build_detector(pretrained=not args.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(args.milestones), gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "device": str(device), "use_amp": use_amp, "train_images": len(ds),
              "torch": torch.__version__}
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2))

    history, start_epoch, it = [], 1, 0
    last = args.run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        gen.set_state(ck["generator"])
        torch.set_rng_state(ck["torch_rng"])
        history, start_epoch, it = ck["history"], ck["epoch"] + 1, ck["iteration"]
        print(f"Resumed from epoch {ck['epoch']}")

    print(f"device={device} amp={use_amp} train_images={len(ds)} batches/epoch={len(loader)}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, sums, n = time.time(), {}, 0
        for images, targets, _ in loader:
            if it <= args.warmup_iters:  # linear warmup, reference recipe; ends at exactly 1.0
                alpha = it / args.warmup_iters
                factor = 0.001 * (1 - alpha) + alpha
                for g in optimizer.param_groups:
                    g["lr"] = g["initial_lr"] * factor
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                losses = model(images, targets)
            loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {losses}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            it += 1
            n += 1
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        scheduler.step()
        record = {"epoch": epoch, **{k: v / n for k, v in sums.items()},
                  "loss": sum(sums.values()) / n, "lr": optimizer.param_groups[0]["lr"],
                  "seconds": round(time.time() - t0, 1)}
        history.append(record)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "generator": gen.get_state(), "torch_rng": torch.get_rng_state(),
                    "epoch": epoch, "iteration": it, "history": history}, last)
        (args.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch:03d}/{args.epochs:03d} loss={record['loss']:.4f} "
              + " ".join(f"{k}={v / n:.4f}" for k, v in sums.items())
              + f" lr={record['lr']:.5f} ({record['seconds']}s)", flush=True)

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, args.run_dir / "final.pt")
    (args.run_dir / "DONE.json").write_text(json.dumps({"epochs": args.epochs}, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
