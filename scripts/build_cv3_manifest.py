from __future__ import annotations

from pathlib import Path

import pandas as pd


IMAGE_DIR = Path(
    "data/raw/isic2018/images/"
    "ISIC2018_Task1-2_Training_Input"
)

MASK_DIR = Path(
    "data/raw/isic2018/masks/"
    "ISIC2018_Task1_Training_GroundTruth"
)

ISIC2019_MANIFEST = Path(
    "data/manifests/isic2019_manifest.csv"
)

ISIC2019_SPLITS = Path(
    "data/splits/isic2019"
)

OUTPUT = Path(
    "data/manifests/isic2018_task1_manifest.csv"
)


def load_isic2019_split_membership() -> pd.DataFrame:
    frames = []

    for split in ("train", "val", "test"):
        path = ISIC2019_SPLITS / f"{split}.csv"

        df = pd.read_csv(path)

        df = df.copy()
        df["isic2019_split"] = split

        frames.append(
            df[
                [
                    "image",
                    "lesion_id",
                    "lesion_id_status",
                    "isic2019_split",
                ]
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def main() -> None:
    images = {
        p.stem: p
        for p in IMAGE_DIR.glob("*.jpg")
    }

    masks = {
        p.name.replace(
            "_segmentation.png",
            "",
        ): p
        for p in MASK_DIR.glob(
            "*_segmentation.png"
        )
    }

    assert len(images) == 2594
    assert len(masks) == 2594

    assert set(images) == set(masks)

    isic19 = load_isic2019_split_membership()

    rows = []

    for image_id in sorted(images):
        match = isic19[
            isic19["image"].astype(str) == image_id
        ]

        if len(match) > 1:
            raise RuntimeError(
                f"Duplicate ISIC 2019 membership: {image_id}"
            )

        if len(match) == 1:
            record = match.iloc[0]

            overlaps = True
            isic19_split = record["isic2019_split"]
            lesion_id = record["lesion_id"]
            lesion_status = record[
                "lesion_id_status"
            ]
        else:
            overlaps = False
            isic19_split = ""
            lesion_id = ""
            lesion_status = ""

        rows.append(
            {
                "dataset": "isic2018_task1",
                "image_id": image_id,
                "image_path": str(images[image_id]),
                "mask_path": str(masks[image_id]),
                "image_domain": "dermoscopic",
                "overlaps_isic2019": overlaps,
                "isic2019_split": isic19_split,
                "isic2019_lesion_id": lesion_id,
                "isic2019_lesion_status": lesion_status,
            }
        )

    manifest = pd.DataFrame(rows)

    # The single CV-2 image that is also part of the
    # frozen ISIC 2019 test set is explicitly excluded.
    manifest["cv2_eligible"] = (
        manifest["isic2019_split"] != "test"
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 80)
    print("DERMASENSE CV-2 MANIFEST")
    print("=" * 80)

    print(f"Total images: {len(manifest)}")
    print(
        f"CV-2 eligible: "
        f"{manifest['cv2_eligible'].sum()}"
    )
    print(
        f"Excluded: "
        f"{(~manifest['cv2_eligible']).sum()}"
    )

    print()
    print("ISIC 2019 overlap:")
    print(
        manifest[
            manifest["overlaps_isic2019"]
        ]["isic2019_split"]
        .value_counts()
        .sort_index()
    )

    print()
    print("ISIC 2019 lesion status among overlaps:")
    print(
        manifest[
            manifest["overlaps_isic2019"]
        ]["isic2019_lesion_status"]
        .value_counts()
    )

    print()
    print("Excluded images:")
    print(
        manifest[
            ~manifest["cv2_eligible"]
        ][
            [
                "image_id",
                "isic2019_split",
                "isic2019_lesion_status",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
