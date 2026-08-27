from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, wilcoxon
from torch.utils.data import DataLoader

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
            "Compare C1 vs F1 SCC/BCC classifier logits "
            "on the exact PAD-UFES test set."
        )
    )

    parser.add_argument(
        "--c1-checkpoint",
        default=(
            "checkpoints/archive/"
            "pad_ufes_c1_partial_finetune_seed42_best.pt"
        ),
    )

    parser.add_argument(
        "--f1-checkpoint",
        default=(
            "artifacts/f1_supcon_seed42/"
            "pad_ufes_f1_supcon_best.pt"
        ),
    )

    parser.add_argument(
        "--geometry",
        default=(
            "analysis/scc_bcc/f1/"
            "scc_lesion_geometry.csv"
        ),
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
        default="analysis/scc_bcc/logit_analysis",
    )

    return parser.parse_args()


def resolve_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_model(checkpoint_path, device):
    config = NativeClassifierConfig(
        backbone="resnet50",
        pretrained=False,
        dropout=0.0,
    )

    model = DermaSenseNativeClassifier(config)

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
            f"No model state dict found in {checkpoint_path}"
        )

    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    return model, checkpoint


def collate(batch):
    return {
        "image": torch.stack(
            [item["image"] for item in batch]
        ),
        "target": torch.tensor(
            [item["target"] for item in batch],
            dtype=torch.long,
        ),
    }


def extract_outputs(model, dataset, device, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    logits = []
    features = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            out = model(
                images,
                "pad_ufes",
            )

            feat = model.forward_features(images)

            logits.append(out.cpu())
            features.append(feat.cpu())
            targets.append(batch["target"])

    return (
        torch.cat(logits).numpy(),
        torch.cat(features).numpy(),
        torch.cat(targets).numpy(),
    )


def safe_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0 or np.allclose(values, 0):
        return np.nan

    try:
        return float(
            wilcoxon(
                values,
                alternative="two-sided",
            ).pvalue
        )
    except ValueError:
        return np.nan


def safe_mannwhitney(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    return float(
        mannwhitneyu(
            a,
            b,
            alternative="two-sided",
        ).pvalue
    )


def main():
    args = parse_args()

    device = resolve_device(args.device)

    c1_path = Path(args.c1_checkpoint)
    f1_path = Path(args.f1_checkpoint)
    geometry_path = Path(args.geometry)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (c1_path, f1_path, geometry_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 80)
    print("DERMASENSE C1 → F1 SCC LOGIT ANALYSIS")
    print("=" * 80)

    print(f"Device: {device}")
    print(f"C1:     {c1_path}")
    print(f"F1:     {f1_path}")

    c1_model, c1_checkpoint = load_model(
        c1_path,
        device,
    )

    f1_model, f1_checkpoint = load_model(
        f1_path,
        device,
    )

    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    print(f"Test images: {len(dataset)}")

    c1_logits, c1_features, targets = extract_outputs(
        c1_model,
        dataset,
        device,
        args.batch_size,
    )

    f1_logits, f1_features, f1_targets = extract_outputs(
        f1_model,
        dataset,
        device,
        args.batch_size,
    )

    if not np.array_equal(targets, f1_targets):
        raise RuntimeError(
            "C1/F1 target ordering differs."
        )

    if not np.array_equal(
        targets,
        np.asarray(
            [item["target"] for item in dataset],
            dtype=int,
        ),
    ):
        raise RuntimeError(
            "Feature/logit output ordering does not match dataset."
        )

    geometry = pd.read_csv(
        geometry_path,
    )

    required_geometry = {
        "patient_id",
        "lesion_uid",
        "image_ids",
        "error_status",
    }

    missing = required_geometry - set(
        geometry.columns
    )

    if missing:
        raise RuntimeError(
            f"Geometry missing columns: {sorted(missing)}"
        )

    # ------------------------------------------------------------
    # Build image-level table.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Recover image IDs from the authoritative PAD-UFES test split.
    # ------------------------------------------------------------
    #
    # The dataset items do not reliably expose image_id, so use the
    # exact test split CSV that defines the 352-image evaluation set.
    # This also avoids accidental blank/duplicate IDs.
    # ------------------------------------------------------------

    test_split_path = Path(
        "data/splits/pad_ufes/test.csv"
    )

    if not test_split_path.exists():
        raise FileNotFoundError(
            f"PAD-UFES test split not found: {test_split_path}"
        )

    test_split = pd.read_csv(
        test_split_path
    )

    if "image_id" not in test_split.columns:
        raise RuntimeError(
            "PAD-UFES test split is missing "
            "'image_id' column."
        )

    image_ids = (
        test_split["image_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(image_ids) != len(dataset):
        raise RuntimeError(
            "PAD-UFES test split length does not match "
            f"dataset length: {len(image_ids)} vs "
            f"{len(dataset)}."
        )

    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError(
            "PAD-UFES test split contains duplicate image IDs."
        )

    if any(x == "" for x in image_ids):
        raise RuntimeError(
            "PAD-UFES test split contains blank image IDs."
        )

    # Verify that the split ordering agrees with the dataset targets.
    split_targets = (
        test_split["native_diagnosis"]
        .map(
            {
                name: index
                for index, name in enumerate(PAD_CLASSES)
            }
        )
        .to_numpy()
    )

    if not np.array_equal(
        split_targets,
        targets,
    ):
        raise RuntimeError(
            "PAD-UFES test split ordering does not match "
            "the dataset target ordering."
        )

    print(
        "Image-ID recovery: PASS "
        f"({len(image_ids)} unique IDs)"
    )

    c1_scc_bcc_logit = (
        c1_logits[:, SCC_INDEX]
        - c1_logits[:, BCC_INDEX]
    )

    f1_scc_bcc_logit = (
        f1_logits[:, SCC_INDEX]
        - f1_logits[:, BCC_INDEX]
    )

    c1_probs = torch.softmax(
        torch.from_numpy(c1_logits),
        dim=1,
    ).numpy()

    f1_probs = torch.softmax(
        torch.from_numpy(f1_logits),
        dim=1,
    ).numpy()

    table = pd.DataFrame({
        "image_id": image_ids,
        "target_index": targets,
        "target_class": [
            PAD_CLASSES[int(x)]
            for x in targets
        ],
        "c1_pred": [
            PAD_CLASSES[int(x)]
            for x in c1_logits.argmax(axis=1)
        ],
        "f1_pred": [
            PAD_CLASSES[int(x)]
            for x in f1_logits.argmax(axis=1)
        ],
        "c1_scc_logit": c1_logits[:, SCC_INDEX],
        "c1_bcc_logit": c1_logits[:, BCC_INDEX],
        "c1_scc_bcc_logit_margin": c1_scc_bcc_logit,
        "f1_scc_logit": f1_logits[:, SCC_INDEX],
        "f1_bcc_logit": f1_logits[:, BCC_INDEX],
        "f1_scc_bcc_logit_margin": f1_scc_bcc_logit,
        "delta_logit_margin": (
            f1_scc_bcc_logit
            - c1_scc_bcc_logit
        ),
        "c1_scc_probability": c1_probs[:, SCC_INDEX],
        "c1_bcc_probability": c1_probs[:, BCC_INDEX],
        "f1_scc_probability": f1_probs[:, SCC_INDEX],
        "f1_bcc_probability": f1_probs[:, BCC_INDEX],
    })

    # Only SCC test images are relevant here.
    table = table[
        table["target_class"] == "SCC"
    ].copy()

    # ------------------------------------------------------------
    # Match SCC images to lesion groups.
    # ------------------------------------------------------------

    geometry_rows = []

    for _, row in geometry.iterrows():
        ids = str(
            row.get("image_ids", "")
        ).split(";")

        for image_id in ids:
            image_id = image_id.strip()

            if not image_id:
                continue

            error_status = str(
                row["error_status"]
            ).strip()

            if error_status == "SCC_to_BCC_error":
                group = "problematic"
            elif error_status == "clean_SCC":
                group = "clean"
            else:
                raise RuntimeError(
                    f"Unknown SCC error status "
                    f"{error_status!r} for lesion "
                    f"{row['lesion_uid']}"
                )

            geometry_rows.append({
                "image_id": image_id,
                "patient_id": row["patient_id"],
                "lesion_uid": row["lesion_uid"],
                "group": group,
            })

    geometry_image = pd.DataFrame(
        geometry_rows
    )

    table = table.merge(
        geometry_image,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if table["lesion_uid"].isna().any():
        missing_ids = table.loc[
            table["lesion_uid"].isna(),
            "image_id",
        ].tolist()

        raise RuntimeError(
            "Some SCC images could not be matched "
            f"to lesion geometry: {missing_ids}"
        )

    if table["lesion_uid"].nunique() != 22:
        raise RuntimeError(
            "Expected 22 SCC lesions after matching; "
            f"got {table['lesion_uid'].nunique()}"
        )

    # ------------------------------------------------------------
    # Aggregate image-level logits to lesion level.
    # ------------------------------------------------------------

    lesion = (
        table.groupby(
            [
                "patient_id",
                "lesion_uid",
                "group",
            ],
            as_index=False,
        )
        .agg(
            image_count=("image_id", "count"),
            c1_scc_bcc_logit_margin=(
                "c1_scc_bcc_logit_margin",
                "mean",
            ),
            f1_scc_bcc_logit_margin=(
                "f1_scc_bcc_logit_margin",
                "mean",
            ),
            delta_logit_margin=(
                "delta_logit_margin",
                "mean",
            ),
            c1_scc_probability=(
                "c1_scc_probability",
                "mean",
            ),
            c1_bcc_probability=(
                "c1_bcc_probability",
                "mean",
            ),
            f1_scc_probability=(
                "f1_scc_probability",
                "mean",
            ),
            f1_bcc_probability=(
                "f1_bcc_probability",
                "mean",
            ),
        )
    )

    lesion["c1_scc_bcc_prediction"] = np.where(
        lesion["c1_scc_bcc_logit_margin"] >= 0,
        "SCC",
        "BCC",
    )

    lesion["f1_scc_bcc_prediction"] = np.where(
        lesion["f1_scc_bcc_logit_margin"] >= 0,
        "SCC",
        "BCC",
    )

    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("LESION-LEVEL SCC/BCC LOGIT SUMMARY")
    print("=" * 80)

    for group in ("problematic", "clean"):
        g = lesion[
            lesion["group"] == group
        ]

        print()
        print(group.upper())
        print("-" * 80)

        print(f"Lesions: {len(g)}")

        print(
            f"C1 mean SCC-BCC logit margin: "
            f"{g['c1_scc_bcc_logit_margin'].mean():.6f}"
        )

        print(
            f"F1 mean SCC-BCC logit margin: "
            f"{g['f1_scc_bcc_logit_margin'].mean():.6f}"
        )

        print(
            f"Mean Δ logit margin: "
            f"{g['delta_logit_margin'].mean():.6f}"
        )

        print(
            f"Median Δ logit margin: "
            f"{g['delta_logit_margin'].median():.6f}"
        )

        print(
            f"C1 SCC-side: "
            f"{(g['c1_scc_bcc_logit_margin'] >= 0).sum()}/"
            f"{len(g)}"
        )

        print(
            f"F1 SCC-side: "
            f"{(g['f1_scc_bcc_logit_margin'] >= 0).sum()}/"
            f"{len(g)}"
        )

    problematic_delta = lesion.loc[
        lesion["group"] == "problematic",
        "delta_logit_margin",
    ]

    clean_delta = lesion.loc[
        lesion["group"] == "clean",
        "delta_logit_margin",
    ]

    print()
    print("=" * 80)
    print("PAIRED LOGIT CHANGE")
    print("=" * 80)

    print(
        f"Overall mean Δ logit margin: "
        f"{lesion['delta_logit_margin'].mean():.6f}"
    )

    print(
        f"Overall median Δ logit margin: "
        f"{lesion['delta_logit_margin'].median():.6f}"
    )

    print(
        f"Problematic mean Δ: "
        f"{problematic_delta.mean():.6f}"
    )

    print(
        f"Clean mean Δ: "
        f"{clean_delta.mean():.6f}"
    )

    problematic_vs_clean_p = safe_mannwhitney(
        problematic_delta,
        clean_delta,
    )

    print(
        f"Problematic vs clean Δ p: "
        f"{problematic_vs_clean_p:.6f}"
    )

    print(
        f"Paired Wilcoxon p "
        f"(all lesions): "
        f"{safe_wilcoxon(lesion['delta_logit_margin']):.6f}"
    )

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------

    image_path = output_dir / "scc_image_logits.csv"
    lesion_path = output_dir / "scc_lesion_logits.csv"

    table.to_csv(
        image_path,
        index=False,
    )

    lesion.to_csv(
        lesion_path,
        index=False,
    )

    summary_path = (
        output_dir / "summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "DERMASENSE C1 → F1 SCC LOGIT ANALYSIS\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            f"C1 checkpoint: {c1_path}\n"
        )
        f.write(
            f"F1 checkpoint: {f1_path}\n"
        )
        f.write(
            f"Matched SCC lesions: {len(lesion)}\n"
        )
        f.write(
            "SCC logit margin = SCC logit - BCC logit\n\n"
        )

        for group in ("problematic", "clean"):
            g = lesion[
                lesion["group"] == group
            ]

            f.write(
                f"{group.upper()}\n"
            )
            f.write("-" * 50 + "\n")

            f.write(
                f"n: {len(g)}\n"
            )

            f.write(
                f"C1 mean margin: "
                f"{g['c1_scc_bcc_logit_margin'].mean():.6f}\n"
            )

            f.write(
                f"F1 mean margin: "
                f"{g['f1_scc_bcc_logit_margin'].mean():.6f}\n"
            )

            f.write(
                f"Mean delta: "
                f"{g['delta_logit_margin'].mean():.6f}\n"
            )

            f.write(
                f"Median delta: "
                f"{g['delta_logit_margin'].median():.6f}\n"
            )

            f.write(
                f"C1 SCC-side: "
                f"{(g['c1_scc_bcc_logit_margin'] >= 0).sum()}/"
                f"{len(g)}\n"
            )

            f.write(
                f"F1 SCC-side: "
                f"{(g['f1_scc_bcc_logit_margin'] >= 0).sum()}/"
                f"{len(g)}\n\n"
            )

        f.write(
            "OVERALL\n"
        )
        f.write("-" * 50 + "\n")

        f.write(
            f"Mean delta: "
            f"{lesion['delta_logit_margin'].mean():.6f}\n"
        )

        f.write(
            f"Median delta: "
            f"{lesion['delta_logit_margin'].median():.6f}\n"
        )

        f.write(
            f"Problematic vs clean delta p: "
            f"{safe_mannwhitney(problematic_delta, clean_delta):.6f}\n"
        )

        f.write(
            f"Paired Wilcoxon p: "
            f"{safe_wilcoxon(lesion['delta_logit_margin']):.6f}\n"
        )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"Image table:   {image_path}")
    print(f"Lesion table:  {lesion_path}")
    print(f"Summary:       {summary_path}")

    print()
    print("=" * 80)
    print("C1 → F1 SCC LOGIT ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
