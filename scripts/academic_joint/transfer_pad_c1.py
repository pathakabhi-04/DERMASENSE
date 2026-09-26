"""Secondary metric: PAD-UFES transfer of a C or J encoder (spec Section 6).

Applies the CV-4 C1 partial fine-tune protocol
(scripts/experiment_c1_partial_finetune.py on main) to the encoder of a
finished run, with the same hyperparameters:

    freeze everything; train layer4 + a new 6-class linear head;
    frozen stages kept in eval mode (BatchNorm statistics frozen);
    AdamW, layer4 lr 1e-5, head lr 1e-4, weight decay 1e-4;
    batch 32, up to 30 epochs, early stop after 7 without val improvement;
    sqrt inverse-frequency class weights; 224x224 input, CV-4 augmentation;
    best state by val macro-F1, then scored once on PAD-UFES test.

The fine-tune seed is the run's seed, so C and J transfers are paired.
The segmentation decoder of a J run is discarded here.

    PYTHONPATH=. python3 scripts/academic_joint/transfer_pad_c1.py \
        --run-dir /workspace/dermasense_academic/runs/academic_joint/J_seed42 \
        --data-root /workspace/dermasense_academic/data/academic_joint_bundle
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.academic.data import PAD_CLASSES, PadDataset  # noqa: E402
from src.academic.metrics import macro_f1, per_class_f1  # noqa: E402
from src.academic.model import AcademicModel, ResNet50Encoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    p.add_argument("--head-learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None, help="Smoke tests only.")
    return p.parse_args()


class PadClassifier(nn.Module):
    def __init__(self, encoder: ResNet50Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(ResNet50Encoder.feature_dim, len(PAD_CLASSES))

    def forward(self, x):
        return self.head(torch.flatten(F.adaptive_avg_pool2d(self.encoder(x)[-1], 1), 1))

    def set_train_mode(self) -> None:
        self.eval()
        self.encoder.layer4.train()
        self.head.train()


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    loss_sum, n, ys, ps = 0.0, 0, [], []
    for b in loader:
        x, y = b["image"].to(device), b["target"].to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y)) * len(y)
        n += len(y)
        ys.append(y.cpu())
        ps.append(logits.argmax(1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {"loss": loss_sum / n, "macro_f1": macro_f1(y, p, len(PAD_CLASSES)),
            "accuracy": float((y == p).mean()), "y": y, "p": p}


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    if ckpt["arm"] not in ("C", "J"):
        raise SystemExit("PAD transfer applies to C and J encoders only.")
    seed = ckpt["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: adaptive_avg_pool2d backward has no deterministic CUDA kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = AcademicModel(ckpt["arm"], pretrained=False, seed=seed)
    source.load_state_dict(ckpt["model"])
    model = PadClassifier(source.encoder).to(device)

    for prm in model.parameters():
        prm.requires_grad = False
    for prm in list(model.encoder.layer4.parameters()) + list(model.head.parameters()):
        prm.requires_grad = True

    def subset(ds):
        if args.max_batches is None:
            return ds
        return Subset(ds, range(min(len(ds), args.max_batches * args.batch_size)))

    train_ds = PadDataset(args.data_root, "train", image_size=args.image_size)
    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(PAD_CLASSES)).float()
    weights = torch.sqrt(counts.sum() / counts.clamp_min(1.0))
    weights = (weights / weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(subset(train_ds), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=gen)
    val_loader = DataLoader(subset(PadDataset(args.data_root, "val", image_size=args.image_size)),
                            batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(subset(PadDataset(args.data_root, "test", image_size=args.image_size)),
                             batch_size=args.batch_size, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        [{"params": model.encoder.layer4.parameters(), "lr": args.backbone_learning_rate},
         {"params": model.head.parameters(), "lr": args.head_learning_rate}],
        weight_decay=args.weight_decay,
    )

    best_f1, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.set_train_mode()
        for b in train_loader:
            x, y = b["image"].to(device), b["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        val = evaluate(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "val_loss": val["loss"], "val_macro_f1": val["macro_f1"]})
        print(f"Epoch {epoch:03d} val_loss={val['loss']:.4f} val_macro_f1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop after {args.patience} epochs without improvement.")
                break

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, criterion)
    result = {
        "arm": ckpt["arm"], "seed": seed, "protocol": "C1 partial fine-tune (layer4 + head)",
        "best_epoch": best_epoch, "pad_val_macro_f1": best_f1,
        "pad_test_macro_f1": test["macro_f1"], "pad_test_accuracy": test["accuracy"],
        "pad_test_n": int(len(test["y"])),
        "pad_test_f1_per_class": dict(zip(PAD_CLASSES, map(float, per_class_f1(test["y"], test["p"], len(PAD_CLASSES))))),
        "history": history, "hyperparameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.run_dir / "pad_c1_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
