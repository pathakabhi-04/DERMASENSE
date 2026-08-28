from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SPLIT_DIR = Path(
    "data/splits/isic2018_task1"
)

OUTPUT_DIR = Path(
    "analysis/quality/cv2_split_audit"
)


def mask_area_fraction(path: str) -> float:
    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Could not read mask: {path}"
        )

    return float(
        np.count_nonzero(mask > 0)
        / mask.size
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for split in ("train", "val", "test"):
        df = pd.read_csv(
            SPLIT_DIR / f"{split}.csv"
        )

        for _, row in df.iterrows():
            records.append(
                {
                    "split": split,
                    "image_id": row["image_id"],
                    "area_fraction": mask_area_fraction(
                        row["mask_path"]
                    ),
                    "overlaps_isic2019": row[
                        "overlaps_isic2019"
                    ],
                }
            )

    data = pd.DataFrame(records)

    print("=" * 80)
    print("DERMASENSE CV-2 SPLIT DISTRIBUTION AUDIT")
    print("=" * 80)

    print()
    print("Images by split:")
    print(
        data["split"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Lesion area fraction:")
    print(
        data.groupby("split")["area_fraction"]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(4)
        .to_string()
    )

    print()
    print("Small-lesion representation:")
    for threshold in (0.01, 0.02, 0.05):
        counts = (
            data.assign(
                small=data["area_fraction"]
                < threshold
            )
            .groupby("split")["small"]
            .agg(["sum", "count"])
        )

        counts["percentage"] = (
            counts["sum"]
            / counts["count"]
            * 100
        )

        print(
            f"\narea < {threshold:.2f}:"
        )
        print(
            counts.to_string()
        )

    print()
    print("Large-lesion representation:")
    for threshold in (0.50, 0.75, 0.90):
        counts = (
            data.assign(
                large=data["area_fraction"]
                > threshold
            )
            .groupby("split")["large"]
            .agg(["sum", "count"])
        )

        counts["percentage"] = (
            counts["sum"]
            / counts["count"]
            * 100
        )

        print(
            f"\narea > {threshold:.2f}:"
        )
        print(
            counts.to_string()
        )

    print()
    print("ISIC-2019 training overlap:")
    print(
        data.groupby("split")[
            "overlaps_isic2019"
        ]
        .sum()
        .to_string()
    )

    data.to_csv(
        OUTPUT_DIR / "cv2_split_cases.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(
        "Saved:",
        OUTPUT_DIR / "cv2_split_cases.csv",
    )


if __name__ == "__main__":
    main()
