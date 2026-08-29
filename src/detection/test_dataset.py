from pathlib import Path

from .dataset import ItobosDetectionDataset


SPLIT_DIR = Path("data/splits/itobos_detection")
DATA_ROOT = Path("data/raw/itobos")


def test_split_loading() -> None:
    for split_name in ("train", "val", "test"):
        manifest = SPLIT_DIR / f"{split_name}.csv"

        if not manifest.exists():
            raise AssertionError(
                f"Missing split manifest: {manifest}"
            )

        # The official test set has no training labels, so only
        # instantiate the loader for train/val here.
        if split_name == "test":
            continue

        dataset = ItobosDetectionDataset(
            manifest_path=manifest,
            data_root=DATA_ROOT,
        )

        assert len(dataset) > 0

        print(
            f"{split_name}: {len(dataset)} samples"
        )


def test_zero_lesion_sample() -> None:
    dataset = ItobosDetectionDataset(
        manifest_path=SPLIT_DIR / "train.csv",
        data_root=DATA_ROOT,
    )

    for index in range(len(dataset)):
        sample = dataset[index]

        if len(sample.boxes) == 0:
            print(
                "Zero-lesion sample:",
                sample.image_id,
            )
            return

    raise AssertionError(
        "No zero-lesion sample found."
    )


def test_single_lesion_sample() -> None:
    dataset = ItobosDetectionDataset(
        manifest_path=SPLIT_DIR / "train.csv",
        data_root=DATA_ROOT,
    )

    for index in range(len(dataset)):
        sample = dataset[index]

        if len(sample.boxes) == 1:
            print(
                "Single-lesion sample:",
                sample.image_id,
            )
            return

    raise AssertionError(
        "No single-lesion sample found."
    )


def test_dense_lesion_sample() -> None:
    dataset = ItobosDetectionDataset(
        manifest_path=SPLIT_DIR / "train.csv",
        data_root=DATA_ROOT,
    )

    for index in range(len(dataset)):
        sample = dataset[index]

        if len(sample.boxes) >= 10:
            print(
                "Dense sample:",
                sample.image_id,
                "boxes=",
                len(sample.boxes),
            )
            return

    raise AssertionError(
        "No 10+ lesion sample found."
    )


if __name__ == "__main__":
    test_split_loading()
    test_zero_lesion_sample()
    test_single_lesion_sample()
    test_dense_lesion_sample()

    print()
    print("=" * 80)
    print("CV-2 DATASET TESTS PASSED")
    print("=" * 80)
