from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/workspace/dermasense")
SPLIT_DIR = ROOT / "data/splits/itobos_detection"


def create_list(split_name: str) -> None:
    manifest = SPLIT_DIR / f"{split_name}.csv"
    output = SPLIT_DIR / f"{split_name}.txt"

    df = pd.read_csv(manifest)

    if "image_path" not in df.columns:
        raise RuntimeError(
            f"{manifest} does not contain image_path"
        )

    paths = []

    for value in df["image_path"]:
        path = Path(str(value))

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {path}"
            )

        paths.append(str(path))

    output.write_text(
        "\n".join(paths) + "\n",
        encoding="utf-8",
    )

    print(f"{split_name}: {len(paths)} images")
    print(f"Saved: {output}")


def main() -> None:
    print("=" * 80)
    print("CV-2 YOLO IMAGE LIST CREATION")
    print("=" * 80)

    create_list("train")
    create_list("val")

    print()
    print("YOLO image lists created successfully.")


if __name__ == "__main__":
    main()
