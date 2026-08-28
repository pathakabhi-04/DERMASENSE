from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_DIR = Path(
    "data/raw/isic2018/images/"
    "ISIC2018_Task1-2_Training_Input"
)

MASK_DIR = Path(
    "data/raw/isic2018/masks/"
    "ISIC2018_Task1_Training_GroundTruth"
)

OUTPUT_DIR = Path(
    "analysis/quality/cv2_mask_audit"
)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    rows = []

    for image_id in sorted(images):
        image_path = images[image_id]
        mask_path = masks[image_id]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        height, width = mask.shape

        foreground = mask > 0

        area_pixels = int(
            np.count_nonzero(foreground)
        )

        area_fraction = (
            area_pixels / float(mask.size)
        )

        ys, xs = np.where(foreground)

        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1

        bbox_area_fraction = (
            bbox_width * bbox_height
            / float(width * height)
        )

        bbox_aspect_ratio = (
            bbox_width / float(bbox_height)
        )

        touches_left = bool(x_min == 0)
        touches_right = bool(
            x_max == width - 1
        )
        touches_top = bool(y_min == 0)
        touches_bottom = bool(
            y_max == height - 1
        )

        touches_border = (
            touches_left
            or touches_right
            or touches_top
            or touches_bottom
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                foreground.astype(np.uint8),
                connectivity=8,
            )
        )

        component_areas = (
            stats[1:, cv2.CC_STAT_AREA]
        )

        component_count = int(
            len(component_areas)
        )

        largest_component_fraction = (
            float(component_areas.max())
            / area_pixels
            if area_pixels > 0
            else 0.0
        )

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "height": height,
                "width": width,
                "area_pixels": area_pixels,
                "area_fraction": area_fraction,
                "bbox_x_min": x_min,
                "bbox_y_min": y_min,
                "bbox_x_max": x_max,
                "bbox_y_max": y_max,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_area_fraction": (
                    bbox_area_fraction
                ),
                "bbox_aspect_ratio": (
                    bbox_aspect_ratio
                ),
                "touches_border": touches_border,
                "touches_left": touches_left,
                "touches_right": touches_right,
                "touches_top": touches_top,
                "touches_bottom": touches_bottom,
                "component_count": component_count,
                "largest_component_fraction": (
                    largest_component_fraction
                ),
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_DIR / "cv2_mask_cases.csv",
        index=False,
    )

    print("=" * 80)
    print("DERMASENSE CV-2 MASK AUDIT")
    print("=" * 80)

    print(f"Images: {len(images)}")
    print(f"Masks:  {len(masks)}")

    print("\n" + "=" * 80)
    print("LESION AREA FRACTION")
    print("=" * 80)

    print(
        df["area_fraction"].describe(
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
        ).to_string()
    )

    print("\n" + "=" * 80)
    print("BOUNDING BOX AREA FRACTION")
    print("=" * 80)

    print(
        df["bbox_area_fraction"].describe(
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
        ).to_string()
    )

    print("\n" + "=" * 80)
    print("BOUNDING BOX ASPECT RATIO")
    print("=" * 80)

    print(
        df["bbox_aspect_ratio"].describe(
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
        ).to_string()
    )

    print("\n" + "=" * 80)
    print("BORDER CONTACT")
    print("=" * 80)

    print(
        f"Touching any border: "
        f"{df['touches_border'].sum()} "
        f"({df['touches_border'].mean() * 100:.2f}%)"
    )

    for column in [
        "touches_left",
        "touches_right",
        "touches_top",
        "touches_bottom",
    ]:
        print(
            f"{column}: "
            f"{df[column].sum()}"
        )

    print("\n" + "=" * 80)
    print("CONNECTED COMPONENTS")
    print("=" * 80)

    print(
        df["component_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMasks with >1 connected component:")
    print(
        int(
            (df["component_count"] > 1).sum()
        )
    )

    print("\n" + "=" * 80)
    print("EXTREME CASES")
    print("=" * 80)

    print("\nSmallest lesion fractions:")
    print(
        df.nsmallest(
            10,
            "area_fraction",
        )[
            [
                "image_id",
                "area_fraction",
                "bbox_area_fraction",
                "component_count",
            ]
        ].to_string(index=False)
    )

    print("\nLargest lesion fractions:")
    print(
        df.nlargest(
            10,
            "area_fraction",
        )[
            [
                "image_id",
                "area_fraction",
                "bbox_area_fraction",
                "component_count",
            ]
        ].to_string(index=False)
    )

    print("\nMost fragmented masks:")
    print(
        df.nlargest(
            10,
            "component_count",
        )[
            [
                "image_id",
                "component_count",
                "largest_component_fraction",
                "area_fraction",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"Saved: "
        f"{OUTPUT_DIR / 'cv2_mask_cases.csv'}"
    )


if __name__ == "__main__":
    main()
