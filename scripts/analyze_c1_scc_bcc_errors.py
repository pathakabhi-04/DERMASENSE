from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from src.data.torch_dataset import CVDatasetTorch
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)


PAD_CLASSES = (
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
)

BCC_INDEX = PAD_CLASSES.index("BCC")
SCC_INDEX = PAD_CLASSES.index("SCC")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc analysis of C1 SCC -> BCC "
            "test errors."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="C1 checkpoint.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output-dir",
        default="analysis/scc_bcc/c1_seed42_errors",
    )

    return parser.parse_args()


def resolve_device(requested: str):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
):
    config = NativeClassifierConfig(
        backbone="resnet50",
        pretrained=False,
        dropout=0.0,
    )

    model = DermaSenseNativeClassifier(
        config
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )

    if state_dict is None:
        raise RuntimeError(
            "Checkpoint does not contain "
            "model_state_dict/state_dict."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def make_contact_sheet(
    errors,
    output_path: Path,
):
    if not errors:
        raise RuntimeError(
            "No errors available for contact sheet."
        )

    # 3 columns x 5 rows for the expected
    # 15 SCC -> BCC errors.
    columns = 3
    rows = (len(errors) + columns - 1) // columns

    cell_width = 360
    image_height = 280
    text_height = 100
    cell_height = image_height + text_height

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            18,
        )
        small_font = ImageFont.truetype(
            "DejaVuSans.ttf",
            14,
        )
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for position, error in enumerate(errors):
        row = position // columns
        column = position % columns

        x0 = column * cell_width
        y0 = row * cell_height

        image_path = Path(
            error["image_path"]
        )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                (
                    cell_width - 20,
                    image_height - 20,
                )
            )

            image_x = (
                x0
                + (cell_width - image.width) // 2
            )

            image_y = (
                y0
                + (image_height - image.height) // 2
            )

            sheet.paste(
                image,
                (
                    image_x,
                    image_y,
                ),
            )

        except Exception as exc:
            draw.text(
                (
                    x0 + 10,
                    y0 + 20,
                ),
                f"Image error:\n{exc}",
                fill="black",
                font=small_font,
            )

        text_y = y0 + image_height + 5

        image_id = error["image_id"]

        bcc_prob = float(
            error["bcc_probability"]
        )

        scc_prob = float(
            error["scc_probability"]
        )

        margin = float(
            error["bcc_minus_scc"]
        )

        draw.text(
            (
                x0 + 8,
                text_y,
            ),
            f"{position + 1}. {image_id}",
            fill="black",
            font=font,
        )

        draw.text(
            (
                x0 + 8,
                text_y + 25,
            ),
            (
                f"True: SCC | Pred: BCC\n"
                f"BCC={bcc_prob:.3f}  "
                f"SCC={scc_prob:.3f}\n"
                f"BCC-SCC={margin:.3f}"
            ),
            fill="black",
            font=small_font,
        )

    sheet.save(
        output_path,
        quality=95,
    )


def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    print("=" * 70)
    print("DERMASENSE C1 SCC -> BCC ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Device:     {device}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    model, checkpoint = load_model(
        checkpoint_path,
        device,
    )

    print(
        f"Experiment: "
        f"{checkpoint.get('experiment')}"
    )

    print(
        f"Architecture: "
        f"{checkpoint.get('architecture')}"
    )

    print(
        f"Seed: "
        f"{checkpoint.get('seed')}"
    )

    if checkpoint.get("seed") != 42:
        print()
        print(
            "WARNING: checkpoint seed is not 42."
        )

    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    print()
    print(
        f"Test samples: {len(dataset)}"
    )

    print(
        f"BCC index: {BCC_INDEX}"
    )

    print(
        f"SCC index: {SCC_INDEX}"
    )

    all_errors = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(dataset),
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                len(dataset),
            )

            indices = list(
                range(start, end)
            )

            batch_images = []
            batch_targets = []

            for index in indices:
                item = dataset[index]

                batch_images.append(
                    item["image"]
                )

                batch_targets.append(
                    int(item["target"])
                )

            images = torch.stack(
                batch_images
            ).to(
                device,
                non_blocking=True,
            )

            targets = torch.tensor(
                batch_targets,
                dtype=torch.long,
            )

            logits = model(
                images,
                dataset_id="pad_ufes",
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            ).cpu()

            probabilities = (
                probabilities.cpu()
            )

            for local_index, dataset_index in enumerate(
                indices
            ):
                true_label = int(
                    targets[local_index]
                )

                predicted_label = int(
                    predictions[local_index]
                )

                # We specifically want:
                #
                # true = SCC
                # predicted = BCC
                if (
                    true_label != SCC_INDEX
                    or predicted_label != BCC_INDEX
                ):
                    continue

                image_id = dataset.get_image_id(
                    dataset_index
                )

                sample = dataset.base_dataset[
                    dataset_index
                ]

                image_path = sample.image_path

                bcc_probability = float(
                    probabilities[
                        local_index,
                        BCC_INDEX,
                    ]
                )

                scc_probability = float(
                    probabilities[
                        local_index,
                        SCC_INDEX,
                    ]
                )

                error = {
                    "dataset_index": dataset_index,
                    "image_id": image_id,
                    "image_path": str(
                        image_path
                    ),
                    "true_label": "SCC",
                    "predicted_label": "BCC",
                    "bcc_probability": (
                        bcc_probability
                    ),
                    "scc_probability": (
                        scc_probability
                    ),
                    "bcc_minus_scc": (
                        bcc_probability
                        - scc_probability
                    ),
                    "prediction_confidence": float(
                        probabilities[
                            local_index
                        ].max()
                    ),
                }

                all_errors.append(
                    error
                )

    # Sort strongest BCC predictions first.
    all_errors.sort(
        key=lambda item: item[
            "bcc_minus_scc"
        ],
        reverse=True,
    )

    print()
    print("=" * 70)
    print("SCC -> BCC ERRORS")
    print("=" * 70)

    print(
        f"Total SCC -> BCC errors: "
        f"{len(all_errors)}"
    )

    for rank, error in enumerate(
        all_errors,
        start=1,
    ):
        print()
        print(
            f"{rank:02d}. "
            f"{error['image_id']}"
        )

        print(
            f"    Index:       "
            f"{error['dataset_index']}"
        )

        print(
            f"    BCC prob:    "
            f"{error['bcc_probability']:.6f}"
        )

        print(
            f"    SCC prob:    "
            f"{error['scc_probability']:.6f}"
        )

        print(
            f"    BCC-SCC:     "
            f"{error['bcc_minus_scc']:.6f}"
        )

        print(
            f"    Confidence:  "
            f"{error['prediction_confidence']:.6f}"
        )

        print(
            f"    Image:       "
            f"{error['image_path']}"
        )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    csv_path = (
        output_dir
        / "scc_to_bcc_errors.csv"
    )

    fieldnames = [
        "dataset_index",
        "image_id",
        "image_path",
        "true_label",
        "predicted_label",
        "bcc_probability",
        "scc_probability",
        "bcc_minus_scc",
        "prediction_confidence",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_errors
        )

    print()
    print(
        f"Saved CSV: {csv_path}"
    )

    # ------------------------------------------------------------
    # Copy the actual images
    # ------------------------------------------------------------

    image_dir = (
        output_dir
        / "images"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for rank, error in enumerate(
        all_errors,
        start=1,
    ):
        source = Path(
            error["image_path"]
        )

        destination = (
            image_dir
            / (
                f"{rank:02d}_"
                f"{error['image_id']}"
                f"{source.suffix}"
            )
        )

        try:
            image = Image.open(
                source
            ).convert("RGB")

            image.save(
                destination,
            )

        except Exception as exc:
            print(
                f"WARNING: could not copy "
                f"{source}: {exc}"
            )

    # ------------------------------------------------------------
    # Contact sheet
    # ------------------------------------------------------------

    contact_sheet_path = (
        output_dir
        / "scc_to_bcc_contact_sheet.jpg"
    )

    make_contact_sheet(
        all_errors,
        contact_sheet_path,
    )

    print(
        f"Saved contact sheet: "
        f"{contact_sheet_path}"
    )

    # ------------------------------------------------------------
    # Sanity check against known C1 confusion matrix
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SANITY CHECK")
    print("=" * 70)

    if len(all_errors) == 15:
        print(
            "PASS: Found exactly 15 SCC -> BCC "
            "errors, matching the C1 seed-42 "
            "test confusion matrix."
        )
    else:
        print(
            "WARNING: Expected 15 SCC -> BCC "
            f"errors but found {len(all_errors)}."
        )

    print()
    print("=" * 70)
    print("C1 SCC -> BCC ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
