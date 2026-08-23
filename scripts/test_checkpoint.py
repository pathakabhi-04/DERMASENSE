from pathlib import Path

import torch

from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.training.checkpoint import (
    inspect_checkpoint,
)
from src.training.engine import (
    Trainer,
    TrainingConfig,
)


CHECKPOINT_PATH = Path(
    "checkpoints/test_checkpoint.pt"
)

DATASET_ID = "isic2019"
NUM_CLASSES = 8
ARCHITECTURE = "CV_MODEL_ARCHITECTURE_v1.0"


def main() -> None:
    print("=" * 70)
    print("CHECKPOINT ROUND-TRIP TEST")
    print("=" * 70)

    model_config = NativeClassifierConfig(
        backbone="resnet18",
        pretrained=False,
        dropout=0.0,
    )

    model = DermaSenseNativeClassifier(
        model_config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # Create a trainer so the exact same checkpoint
    # loading path used during training is exercised.
    trainer = Trainer(
        model=model,
        train_loader=[],
        val_loader=[],
        dataset_id=DATASET_ID,
        num_classes=NUM_CLASSES,
        config=TrainingConfig(
            epochs=1,
            device="cpu",
        ),
    )

    # Put the model into a known state.
    torch.manual_seed(42)

    for parameter in trainer.model.parameters():
        parameter.data.normal_(0.0, 0.02)

    # Save using the same utility used by Trainer.fit().
    from src.training.checkpoint import save_checkpoint

    save_checkpoint(
        CHECKPOINT_PATH,
        model=trainer.model,
        optimizer=trainer.optimizer,
        epoch=3,
        dataset_id=DATASET_ID,
        num_classes=NUM_CLASSES,
        architecture=ARCHITECTURE,
        val_macro_f1=0.7125,
        config={
            "test": True,
        },
    )

    print(
        f"Saved checkpoint: {CHECKPOINT_PATH}"
    )

    # Inspect metadata without loading into a model.
    metadata = inspect_checkpoint(
        CHECKPOINT_PATH
    )

    assert metadata.architecture == ARCHITECTURE
    assert metadata.dataset_id == DATASET_ID
    assert metadata.num_classes == NUM_CLASSES
    assert metadata.epoch == 3
    assert metadata.val_macro_f1 == 0.7125

    print("Metadata inspection: PASS")

    # Create a completely fresh model/trainer.
    fresh_model = DermaSenseNativeClassifier(
        model_config
    )

    fresh_trainer = Trainer(
        model=fresh_model,
        train_loader=[],
        val_loader=[],
        dataset_id=DATASET_ID,
        num_classes=NUM_CLASSES,
        config=TrainingConfig(
            epochs=1,
            device="cpu",
        ),
    )

    loaded = fresh_trainer.load_checkpoint(
        str(CHECKPOINT_PATH)
    )

    assert loaded["architecture"] == ARCHITECTURE
    assert loaded["dataset_id"] == DATASET_ID
    assert loaded["num_classes"] == NUM_CLASSES
    assert loaded["epoch"] == 3
    assert loaded["val_macro_f1"] == 0.7125

    print("Checkpoint loading: PASS")
    print("Metadata compatibility: PASS")

    # Verify that model parameters are identical.
    for saved, loaded_parameter in zip(
        trainer.model.parameters(),
        fresh_trainer.model.parameters(),
    ):
        if not torch.equal(
            saved.detach().cpu(),
            loaded_parameter.detach().cpu(),
        ):
            raise AssertionError(
                "Loaded model parameters do not "
                "match saved parameters."
            )

    print("Model state equality: PASS")

    # Verify that the optimizer state was restored.
    assert (
        fresh_trainer.optimizer.state_dict()
        == trainer.optimizer.state_dict()
    )

    print("Optimizer state equality: PASS")

    # Remove test artifact.
    CHECKPOINT_PATH.unlink()

    print()
    print("=" * 70)
    print("ALL CHECKPOINT ROUND-TRIP CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()