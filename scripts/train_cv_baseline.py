from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
import yaml

from src.data.loader import (
    DataLoaderConfig,
    build_dataloader,
)
from src.data.torch_dataset import CVDatasetTorch
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.training.engine import (
    Trainer,
    TrainingConfig,
)
from src.training.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DermaSense CV native-diagnosis baseline."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML experiment configuration.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one training epoch using a small subset "
            "of the train/validation datasets."
        ),
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )

    return config


def resolve_device(config: dict) -> str:
    requested = config["runtime"].get(
        "device",
        "auto",
    )

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

        return "cpu"

    return requested


def build_dataset(
    dataset_id: str,
    split: str,
) -> CVDatasetTorch:
    return CVDatasetTorch(
        dataset_id=dataset_id,
        split=split,
        verify_images=True,
    )


def build_loader(
    dataset: CVDatasetTorch,
    config: dict,
    *,
    smoke_test: bool = False,
):
    runtime = config["runtime"]
    training = config["training"]

    loader_config = DataLoaderConfig(
        batch_size=training["batch_size"],
        num_workers=runtime.get(
            "num_workers",
            0,
        ),
        pin_memory=runtime.get(
            "pin_memory",
            False,
        ),
        drop_last=False,
    )

    return build_dataloader(
        dataset=dataset,
        config=loader_config,
    )


def limit_loader(loader, max_batches: int):
    """
    Wrap a DataLoader so only a small number of batches
    are consumed during smoke testing.
    """

    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break

        yield batch


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    seed = config["experiment"]["seed"]

    seed_everything(seed)

    dataset_id = config["dataset"]["id"]

    train_split = config["dataset"]["train_split"]
    val_split = config["dataset"]["val_split"]

    print("=" * 70)
    print("DERMASENSE CV BASELINE")
    print("=" * 70)

    print(f"Experiment:    {config['experiment']['name']}")
    print(f"Architecture:  {config['experiment']['architecture']}")
    print(f"Dataset:       {dataset_id}")
    print(f"Seed:           {seed}")

    device = resolve_device(config)

    print(f"Device:         {device}")

    if device == "cuda":
        print(
            f"GPU:            "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 70)

    train_dataset = build_dataset(
        dataset_id,
        train_split,
    )

    val_dataset = build_dataset(
        dataset_id,
        val_split,
    )

    print()
    print("DATASET")
    print(
        f"Train samples:  {len(train_dataset)}"
    )
    print(
        f"Val samples:    {len(val_dataset)}"
    )
    print(
        f"Classes:        {train_dataset.class_names}"
    )
    print(
        f"Num classes:    {train_dataset.num_classes}"
    )

    if (
        train_dataset.num_classes
        != val_dataset.num_classes
    ):
        raise RuntimeError(
            "Train/validation class counts differ."
        )

    if (
        train_dataset.class_names
        != val_dataset.class_names
    ):
        raise RuntimeError(
            "Train/validation class ordering differs."
        )

    train_loader = build_loader(
        train_dataset,
        config,
    )

    val_loader = build_loader(
        val_dataset,
        config,
    )

    model_config = NativeClassifierConfig(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"].get(
            "dropout",
            0.0,
        ),
    )

    model = DermaSenseNativeClassifier(
        model_config
    )

    print()
    print("MODEL")
    print(model)

    print()
    print("PARAMETERS")

    parameter_counts = model.parameter_counts()

    for name, count in parameter_counts.items():
        print(
            f"{name:20s}: {count:,}"
        )

    class_weighting = config["training"].get(
        "class_weighting",
        "none",
    )

    criterion = None

    if class_weighting == "sqrt_inverse_frequency":
        class_counts = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float64,
        )

        for index in range(len(train_dataset)):
            target = train_dataset.get_target(index)
            class_counts[target] += 1

        if torch.any(class_counts <= 0):
            raise RuntimeError(
                "Cannot calculate class weights because "
                "one or more classes have zero training samples."
            )

        total_samples = class_counts.sum()
        num_classes = train_dataset.num_classes

        class_weights = torch.sqrt(
            total_samples
            / (num_classes * class_counts)
        )

        class_weights = (
            class_weights
            / class_weights.mean()
        )

        class_weights = class_weights.float()

        print()
        print("CLASS WEIGHTING")
        print(
            f"Strategy:       {class_weighting}"
        )

        for index, class_name in enumerate(
            train_dataset.class_names
        ):
            print(
                f"  {class_name:6s} | "
                f"count={int(class_counts[index]):5d} | "
                f"weight={class_weights[index].item():.4f}"
            )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

    elif class_weighting != "none":
        raise ValueError(
            "Unsupported class_weighting strategy: "
            f"{class_weighting!r}. "
            "Expected 'none' or "
            "'sqrt_inverse_frequency'."
        )

    training_config = TrainingConfig(
        epochs=(
            1
            if args.smoke_test
            else config["training"]["epochs"]
        ),
        learning_rate=config["training"][
            "learning_rate"
        ],
        weight_decay=config["training"][
            "weight_decay"
        ],
        device=device,
        gradient_clip_norm=config[
            "training"
        ].get("gradient_clip_norm"),
    )

    trainer = Trainer(
        model=model,
        train_loader=(
            train_loader
        ),
        val_loader=(
            val_loader
        ),
        dataset_id=dataset_id,
        num_classes=train_dataset.num_classes,
        config=training_config,
        criterion=criterion,
    )

    if args.smoke_test:
        print()
        print("=" * 70)
        print("SMOKE TEST")
        print("=" * 70)
        print(
            "Running one epoch over a limited number "
            "of batches."
        )

        smoke_train_loader = limit_loader(
            train_loader,
            max_batches=5,
        )

        smoke_val_loader = limit_loader(
            val_loader,
            max_batches=3,
        )

        trainer.train_loader = smoke_train_loader
        trainer.val_loader = smoke_val_loader

        train_result = trainer.train_epoch()
        val_result = trainer.validate_epoch()

        print()
        print("TRAIN RESULT")
        print(train_result)

        print()
        print("VAL RESULT")
        print(val_result)

        print()
        print("SMOKE TEST PASSED")
        return

    checkpoint_dir = Path(
        config["checkpoint"]["directory"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / config["checkpoint"]["filename"]
    )

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = trainer.fit(
        checkpoint_path=str(
            checkpoint_path
        )
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch:        "
        f"{history.best_epoch}"
    )

    print(
        f"Best val Macro-F1: "
        f"{history.best_val_macro_f1:.4f}"
    )

    print(
        f"Checkpoint:        "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()