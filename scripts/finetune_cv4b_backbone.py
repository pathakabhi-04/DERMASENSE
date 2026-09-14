"""
CV-4b backbone fine-tune. Implements `docs/cv4b_backbone_finetune_design.md`
exactly; read that first. This script does not re-decide anything the
design pre-committed (what trains, mixing ratio, selection metric,
augmentation, decision rule).

Trains ResNet-50 layer4 + a binary referral head on ISIC (refer-vs-benign)
mixed with clinical photos (malignant-vs-benign), oversampled so clinical
images are ~50% of each batch despite being ~6% of the data.

The test split is NEVER read here. `evaluate_cv4b_finetune.py` does that
once, afterwards, against the selected checkpoint.

    PYTHONPATH=. python3 scripts/finetune_cv4b_backbone.py --fold pooled
    PYTHONPATH=. python3 scripts/finetune_cv4b_backbone.py --fold loso_atlas --resume
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.dataset import CVDataset
from src.data.transforms import (
    ImageTransformConfig,
    build_eval_transform,
    build_train_transform,
)
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.training.reproducibility import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
SPLITS = REPO_ROOT / "analysis/quality/mel_sensitivity/cv4b_finetune_splits.csv"
RUN_ROOT = REPO_ROOT / "checkpoints/cv4b_finetune"

# ISIC's referral-need label (design §"label harmonization" in the scope doc).
REFER = ("MEL", "BCC", "SCC", "AK")
BENIGN = ("NV", "BKL")

FEATURE_DIM = 2048
NON_ISIC_BATCH_FRACTION = 0.5  # design §6
FOLDS = ("pooled", "loso_atlas", "loso_stanford", "dg_atlas", "dg_stanford", "dg_pad")
DG_FOLDS = ("dg_atlas", "dg_stanford", "dg_pad")
# Auxiliary per-domain heads shape the backbone; the POOLED head is
# what ships, because a user's phone is an unseen domain with no head
# of its own (docs/cv4b_domain_generalization_design.md §3).
DOMAIN_GROUPS = ("isic", "pad", "stanford", "atlas")
AUX_LOSS_WEIGHT = 0.5

# design §7 -- targeted at the colour/white-balance hypothesis the
# composition audit left open, not a uniform turn-up.
TRAIN_TRANSFORM_CONFIG = ImageTransformConfig(
    image_size=224,
    color_jitter_brightness=0.35,
    color_jitter_contrast=0.35,
    color_jitter_saturation=0.35,
    color_jitter_hue=0.08,
    random_resized_crop_enabled=True,
    random_resized_crop_scale_min=0.7,
    random_resized_crop_scale_max=1.0,
    rotation_degrees=15.0,
)


@dataclass(frozen=True)
class Row:
    image_path: str
    label: int  # 1 = refer/malignant
    source: str  # "isic2019" | "pad_ufes" | "ddi" | "ddi2" | "fitzpatrick17k"
    is_melanoma: bool
    domain_group: str = "isic"  # isic | pad | stanford | atlas


class BinaryLesionDataset(Dataset):
    def __init__(self, rows: list[Row], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite tensor from {row.image_path}")
        return {
            "image": tensor,
            "label": torch.tensor(float(row.label)),
            "domain": torch.tensor(DOMAIN_GROUPS.index(row.domain_group)),
            "source_index": index,
        }


def isic_rows(split: str) -> list[Row]:
    dataset = CVDataset(dataset_id="isic2019", split=split, verify_images=False)
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        diagnosis = sample.native_diagnosis
        if diagnosis not in REFER + BENIGN:
            continue  # DF/VASC have no analogue; excluded, not mapped
        rows.append(Row(
            image_path=str(sample.image_path),
            label=int(diagnosis in REFER),
            source="isic2019",
            is_melanoma=diagnosis == "MEL",
        ))
    return rows


def non_isic_rows(fold: str, split: str) -> list[Row]:
    import pandas as pd

    table = pd.read_csv(SPLITS)
    if fold not in table.columns:
        raise SystemExit(f"unknown fold {fold!r}; available: pooled, loso_atlas, loso_stanford")
    selected = table[table[fold] == split]
    return [
        Row(
            image_path=row.image_path,
            label=int(row.is_malignant),
            source=row.source,
            is_melanoma=bool(row.is_melanoma),
        )
        for row in selected.itertuples()
    ]


DEFAULT_BUNDLE = REPO_ROOT / "data/processed/cv4b_bundle"


def bundle_rows(data_root: Path, fold: str, split: str) -> list[Row]:
    """Read rows from the portable bundle (scripts/build_gpu_bundle.py).

    This is the ONLY data path used by training and evaluation, locally
    and on the GPU host alike -- so the configuration validated by
    pre-flight is the one that runs on the rented card, rather than a
    second path first exercised when it costs money.

    ISIC rows carry `isic_split`; clinical rows carry a per-fold
    assignment. Both resolve relative to `data_root`, so nothing depends
    on this machine's directory layout.
    """
    import pandas as pd

    dataset_csv = Path(data_root) / "dataset.csv"
    if not dataset_csv.exists():
        raise SystemExit(
            f"no bundle at {data_root} (expected dataset.csv). "
            "Build it with scripts/build_gpu_bundle.py, or point --data-root "
            "at the network volume copy."
        )
    table = pd.read_csv(dataset_csv, keep_default_na=False)
    if fold not in table.columns:
        raise SystemExit(f"unknown fold {fold!r}; available: {', '.join(FOLDS)}")

    if fold in DG_FOLDS:
        # dg folds are authoritative for EVERY source, ISIC and PAD
        # included -- no per-source special-casing, and "unused" rows are
        # excluded by simply not matching the requested split.
        selected = table[table[fold] == split]
    else:
        is_isic = table["source"] == "isic2019"
        selected = table[(is_isic & (table["isic_split"] == split))
                         | (~is_isic & (table[fold] == split))]

    rows = []
    for record in selected.itertuples():
        path = Path(data_root) / record.relative_path
        domain = getattr(record, "domain_group", "isic") or "isic"
        rows.append(Row(str(path), int(record.label), record.source,
                        bool(record.is_melanoma), domain))
    if not rows:
        raise SystemExit(f"bundle produced no rows for fold={fold} split={split}")
    return rows


def resolve_source_checkpoint(data_root: Path | None = None) -> Path:
    """The fine-tune starts from the shipped checkpoint, which is
    gitignored -- a fresh clone on a GPU host does not have it. The bundle
    carries a copy, so prefer that and fall back to the local repo path."""
    if data_root is not None:
        bundled = Path(data_root) / "checkpoint" / SOURCE_CHECKPOINT.name
        if bundled.exists():
            return bundled
    if SOURCE_CHECKPOINT.exists():
        return SOURCE_CHECKPOINT
    raise SystemExit(
        f"source checkpoint not found in the bundle or at {SOURCE_CHECKPOINT}. "
        "Rebuild the bundle (scripts/build_gpu_bundle.py) so it ships with one."
    )


def build_model(device: torch.device, data_root: Path | None = None):
    """Load the shipped backbone, freeze everything but layer4, attach a
    fresh binary head. Freeze correctness is asserted, not printed."""
    model = DermaSenseNativeClassifier(
        NativeClassifierConfig(backbone="resnet50", pretrained=False, dropout=0.0)
    )
    checkpoint = torch.load(
        resolve_source_checkpoint(data_root), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise RuntimeError("no model state dict in the source checkpoint")
    model.load_state_dict(state_dict, strict=True)

    for parameter in model.parameters():
        parameter.requires_grad = False

    layer4 = dict(model.backbone.features.named_children()).get("7")
    if layer4 is None:
        raise RuntimeError("could not locate ResNet-50 layer4 at backbone.features[7]")
    for parameter in layer4.parameters():
        parameter.requires_grad = True

    # Pooled head (ships) + per-domain auxiliary heads (shape the
    # backbone only). A ModuleDict keeps them in one state_dict so the
    # checkpoint round-trip and resume paths need no special handling.
    head = nn.ModuleDict({
        "pooled": nn.Linear(FEATURE_DIM, 1),
        **{f"aux_{group}": nn.Linear(FEATURE_DIM, 1) for group in DOMAIN_GROUPS},
    })

    trainable_backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    layer4_params = sum(p.numel() for p in layer4.parameters())
    if trainable_backbone != layer4_params:
        raise RuntimeError(
            f"freeze mask wrong: {trainable_backbone} trainable backbone params, "
            f"expected exactly layer4's {layer4_params}"
        )
    for name, module in (("pad_ufes", model.pad_ufes_head), ("isic2019", model.isic2019_head)):
        if any(p.requires_grad for p in module.parameters()):
            raise RuntimeError(f"{name} head must stay frozen")

    return model.to(device), head.to(device), layer4


def make_sampler(rows: list[Row], generator: torch.Generator) -> WeightedRandomSampler:
    """Weight so non-ISIC is NON_ISIC_BATCH_FRACTION of each batch in
    expectation. Without this, clinical photos are ~6% of the gradient and
    the run's most likely outcome is a false negative (design §6)."""
    is_non_isic = np.array([row.source != "isic2019" for row in rows])
    n_non_isic = int(is_non_isic.sum())
    n_isic = len(rows) - n_non_isic
    if n_non_isic == 0 or n_isic == 0:
        raise RuntimeError("expected both ISIC and non-ISIC rows in the training mix")

    weights = np.where(
        is_non_isic,
        NON_ISIC_BATCH_FRACTION / n_non_isic,
        (1.0 - NON_ISIC_BATCH_FRACTION) / n_isic,
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )


@torch.no_grad()
def evaluate(model, head, rows: list[Row], device, batch_size: int, workers: int) -> dict:
    """Per-source malignant-vs-benign AUC on val. Never called on test."""
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
    scores = np.concatenate(scores)

    labels = np.array([row.label for row in rows])
    sources = np.array([row.source for row in rows])

    def auc_for(mask) -> float:
        if mask.sum() == 0 or len(set(labels[mask])) < 2:
            return float("nan")
        return float(roc_auc_score(labels[mask], scores[mask]))

    non_isic = sources != "isic2019"
    results = {
        "isic_auc": auc_for(~non_isic),
        "non_isic_auc": auc_for(non_isic),
    }
    for source in sorted(set(sources[non_isic])):
        results[f"{source}_auc"] = auc_for(sources == source)
    return results


def rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_run_checkpoint(path: Path, **payload) -> None:
    """Written every epoch, to persistent storage, with optimizer and RNG
    state -- a rented instance can die mid-run (design §9.6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)  # atomic: never leave a half-written checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="pooled", choices=FOLDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_BUNDLE,
                        help="bundle directory; on the GPU host this is the network volume copy")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT,
                        help="where checkpoints go; point at persistent storage on a rented pod")
    parser.add_argument("--max-steps", type=int, default=0, help="preflight only; 0 = full epoch")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")

    run_dir = args.run_root / args.fold
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"

    train_rows = bundle_rows(args.data_root, args.fold, "train")
    val_rows = bundle_rows(args.data_root, args.fold, "val")
    print(f"bundle: {args.data_root}")
    print(f"[{args.fold}] train {len(train_rows)} "
          f"(non-ISIC {sum(r.source != 'isic2019' for r in train_rows)}) | "
          f"val {len(val_rows)}")

    model, head, layer4 = build_model(device, args.data_root)
    optimizer = torch.optim.AdamW(
        [
            {"params": layer4.parameters(), "lr": args.backbone_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 1
    best_metric = float("-inf")
    epochs_without_improvement = 0
    history = []

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        history = checkpoint.get("history", [])
        print(f"resumed from epoch {checkpoint['epoch']}, best={best_metric:.4f}")

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        BinaryLesionDataset(train_rows, build_train_transform(TRAIN_TRANSFORM_CONFIG)),
        batch_size=args.batch_size,
        sampler=make_sampler(train_rows, sampler_generator),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # Frozen stages stay in eval mode so their BatchNorm running stats
        # are not silently updated by training-mode forward passes.
        for index in range(7):
            model.backbone.features[index].eval()
        layer4.train()
        model.pad_ufes_head.eval()
        model.isic2019_head.eval()
        head.train()

        running_loss, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            domains = batch["domain"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            features = model.extract_features(images)
            loss = criterion(head["pooled"](features).squeeze(1), labels)

            # Auxiliary per-domain heads. Each sees only its own domain's
            # samples, so it absorbs that domain's prevalence and
            # calibration and the shared backbone is pushed to carry what
            # generalises. They are never used at inference.
            for index, group in enumerate(DOMAIN_GROUPS):
                mask = domains == index
                if not bool(mask.any()):
                    continue
                aux_logits = head[f"aux_{group}"](features[mask]).squeeze(1)
                loss = loss + AUX_LOSS_WEIGHT * criterion(aux_logits, labels[mask])

            if not torch.isfinite(loss):
                raise RuntimeError("loss became non-finite")
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            seen += len(labels)
            if args.max_steps and step >= args.max_steps:
                break

        metrics = evaluate(model, head, val_rows, device, args.batch_size, args.workers)
        selection_metric = metrics["non_isic_auc"]  # design §8, NOT macro-F1
        record = {"epoch": epoch, "train_loss": running_loss / max(seen, 1), **metrics}
        history.append(record)
        print(f"epoch {epoch:03d} | loss {record['train_loss']:.4f} | "
              + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        improved = selection_metric > best_metric
        if improved:
            best_metric = selection_metric
            epochs_without_improvement = 0
            save_run_checkpoint(
                best_path, model=model.state_dict(), head=head.state_dict(),
                epoch=epoch, metrics=metrics, fold=args.fold, seed=args.seed,
                source_checkpoint=str(SOURCE_CHECKPOINT),
                selection_metric="non_isic_val_auc",
            )
            print(f"  -> new best non-ISIC val AUC {best_metric:.4f}")
        else:
            epochs_without_improvement += 1

        save_run_checkpoint(
            latest_path, model=model.state_dict(), head=head.state_dict(),
            optimizer=optimizer.state_dict(), rng=rng_state(), epoch=epoch,
            best_metric=best_metric, epochs_without_improvement=epochs_without_improvement,
            history=history, fold=args.fold,
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if epochs_without_improvement >= args.patience:
            print(f"early stop: {args.patience} epochs without improvement")
            break

    print(f"\nbest non-ISIC val AUC {best_metric:.4f} -> {best_path}")
    print("test is untouched; run scripts/evaluate_cv4b_finetune.py to use it once")


if __name__ == "__main__":
    main()
