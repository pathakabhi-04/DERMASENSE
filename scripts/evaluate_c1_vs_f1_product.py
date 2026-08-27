from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import CVDatasetTorch
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)


CLASSES = (
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
)

HIGH_RISK = {"BCC", "SCC", "MEL"}

CLASS_TO_INDEX = {
    name: i for i, name in enumerate(CLASSES)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DermaSense C1 vs F1 six-class product evaluation "
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
        "--test-csv",
        default="data/splits/pad_ufes/test.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="analysis/product_eval",
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

    return parser.parse_args()


def resolve_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but CUDA is unavailable."
            )
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

    # Handle checkpoints saved from DataParallel.
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def extract_logits(
    model,
    dataset,
    device,
    batch_size,
):
    """
    Extract model logits from CVDatasetTorch.

    CVDatasetTorch returns dictionaries containing:
        image  -> torch.Tensor
        target -> int
        sample -> CVSample

    Samples are manually batched to preserve the exact dataset
    ordering while avoiding the default collate function.
    """

    logits_all = []
    targets_all = []

    images = []
    targets = []

    def flush_batch():
        if not images:
            return

        batch_images = torch.stack(images).to(
            device,
            non_blocking=True,
        )

        with torch.no_grad():
            output = model(
                batch_images,
                dataset_id="pad_ufes",
            )

            if isinstance(output, dict):
                logits = output.get("logits")

                if logits is None:
                    raise RuntimeError(
                        "Model returned a dict without 'logits'."
                    )

            elif isinstance(output, (tuple, list)):
                logits = output[0]

            else:
                logits = output

        logits_all.append(
            logits.detach().cpu()
        )

        targets_all.append(
            torch.tensor(
                targets,
                dtype=torch.long,
            )
        )

        images.clear()
        targets.clear()

    for i in range(len(dataset)):
        item = dataset[i]

        if not isinstance(item, dict):
            raise RuntimeError(
                "Unexpected dataset item type: "
                f"{type(item)}"
            )

        if "image" not in item:
            raise RuntimeError(
                "Dataset item is missing 'image'."
            )

        if "target" not in item:
            raise RuntimeError(
                "Dataset item is missing 'target'."
            )

        image = item["image"]
        target = item["target"]

        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)

        target = int(target)

        images.append(image)
        targets.append(target)

        if len(images) >= batch_size:
            flush_batch()

    flush_batch()

    if not logits_all:
        raise RuntimeError(
            "No logits were produced."
        )

    return (
        torch.cat(logits_all).numpy(),
        torch.cat(targets_all).numpy(),
    )

def softmax(logits):
    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    shifted = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    exp = np.exp(shifted)

    return exp / exp.sum(
        axis=1,
        keepdims=True,
    )


def wilson_interval(
    successes,
    total,
    z=1.959963984540054,
):
    if total == 0:
        return np.nan, np.nan

    p = successes / total
    denom = 1.0 + z ** 2 / total

    centre = (
        p
        + z ** 2 / (2.0 * total)
    ) / denom

    half = (
        z
        * np.sqrt(
            (
                p * (1.0 - p)
                + z ** 2 / (4.0 * total)
            )
            / total
        )
        / denom
    )

    return (
        max(0.0, centre - half),
        min(1.0, centre + half),
    )


def classification_metrics(
    y_true,
    y_pred,
):
    rows = []

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for class_name in CLASSES:
        idx = CLASS_TO_INDEX[class_name]

        true_positive = int(
            np.sum(
                (y_true == idx)
                & (y_pred == idx)
            )
        )

        actual_positive = int(
            np.sum(y_true == idx)
        )

        predicted_positive = int(
            np.sum(y_pred == idx)
        )

        recall = (
            true_positive / actual_positive
            if actual_positive
            else np.nan
        )

        precision = (
            true_positive / predicted_positive
            if predicted_positive
            else np.nan
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (
                np.isfinite(precision)
                and np.isfinite(recall)
                and precision + recall > 0
            )
            else np.nan
        )

        recall_lo, recall_hi = wilson_interval(
            true_positive,
            actual_positive,
        )

        precision_lo, precision_hi = wilson_interval(
            true_positive,
            predicted_positive,
        )

        rows.append({
            "class": class_name,
            "support": actual_positive,
            "predicted_count": predicted_positive,
            "true_positive": true_positive,
            "precision": precision,
            "precision_wilson_low": precision_lo,
            "precision_wilson_high": precision_hi,
            "recall": recall,
            "recall_wilson_low": recall_lo,
            "recall_wilson_high": recall_hi,
            "f1": f1,
        })

    valid_f1 = [
        x["f1"]
        for x in rows
        if np.isfinite(x["f1"])
    ]

    macro_f1 = (
        float(np.mean(valid_f1))
        if valid_f1
        else np.nan
    )

    accuracy = float(
        np.mean(y_true == y_pred)
    )

    return rows, accuracy, macro_f1


def confusion_matrix_df(
    y_true,
    y_pred,
):
    matrix = np.zeros(
        (len(CLASSES), len(CLASSES)),
        dtype=int,
    )

    for true_value, pred_value in zip(
        y_true,
        y_pred,
    ):
        matrix[
            int(true_value),
            int(pred_value),
        ] += 1

    return pd.DataFrame(
        matrix,
        index=[
            f"true_{x}"
            for x in CLASSES
        ],
        columns=[
            f"pred_{x}"
            for x in CLASSES
        ],
    )


def high_risk_metrics(
    y_true,
    y_pred,
):
    true_names = np.array(
        [CLASSES[int(x)] for x in y_true]
    )

    pred_names = np.array(
        [CLASSES[int(x)] for x in y_pred]
    )

    true_high = np.isin(
        true_names,
        list(HIGH_RISK),
    )

    pred_high = np.isin(
        pred_names,
        list(HIGH_RISK),
    )

    high_risk_tp = int(
        np.sum(true_high & pred_high)
    )

    high_risk_total = int(
        np.sum(true_high)
    )

    high_risk_recall = (
        high_risk_tp / high_risk_total
        if high_risk_total
        else np.nan
    )

    lo, hi = wilson_interval(
        high_risk_tp,
        high_risk_total,
    )

    per_class = {}

    for class_name in (
        "MEL",
        "BCC",
        "SCC",
    ):
        idx = CLASS_TO_INDEX[class_name]

        tp = int(
            np.sum(
                (y_true == idx)
                & (y_pred == idx)
            )
        )

        total = int(
            np.sum(y_true == idx)
        )

        recall = (
            tp / total
            if total
            else np.nan
        )

        per_class[class_name] = {
            "recall": recall,
            "support": total,
            "wilson_low": (
                wilson_interval(tp, total)[0]
            ),
            "wilson_high": (
                wilson_interval(tp, total)[1]
            ),
        }

    recalls = [
        per_class[x]["recall"]
        for x in ("MEL", "BCC", "SCC")
        if np.isfinite(per_class[x]["recall"])
    ]

    minimum_malignant_recall = (
        float(min(recalls))
        if recalls
        else np.nan
    )

    return {
        "high_risk_support": high_risk_total,
        "high_risk_true_positive": high_risk_tp,
        "high_risk_recall": high_risk_recall,
        "high_risk_recall_wilson_low": lo,
        "high_risk_recall_wilson_high": hi,
        "mel_recall": per_class["MEL"]["recall"],
        "mel_recall_low": per_class["MEL"]["wilson_low"],
        "mel_recall_high": per_class["MEL"]["wilson_high"],
        "bcc_recall": per_class["BCC"]["recall"],
        "bcc_recall_low": per_class["BCC"]["wilson_low"],
        "bcc_recall_high": per_class["BCC"]["wilson_high"],
        "scc_recall": per_class["SCC"]["recall"],
        "scc_recall_low": per_class["SCC"]["wilson_low"],
        "scc_recall_high": per_class["SCC"]["wilson_high"],
        "minimum_malignant_recall": minimum_malignant_recall,
    }


def compute_ece(
    probabilities,
    y_true,
    n_bins=10,
):
    confidence = np.max(
        probabilities,
        axis=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    correctness = (
        predictions == y_true
    ).astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []
    ece = 0.0

    total = len(y_true)

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(np.sum(mask))

        if count == 0:
            continue

        mean_confidence = float(
            np.mean(confidence[mask])
        )

        accuracy = float(
            np.mean(correctness[mask])
        )

        gap = abs(
            accuracy
            - mean_confidence
        )

        ece += (
            count / total
        ) * gap

        rows.append({
            "bin": i,
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "absolute_gap": gap,
        })

    return float(ece), pd.DataFrame(rows)


def build_prediction_table(
    test_df,
    y_true,
    c1_probs,
    f1_probs,
):
    c1_pred = np.argmax(
        c1_probs,
        axis=1,
    )

    f1_pred = np.argmax(
        f1_probs,
        axis=1,
    )

    table = test_df[
        [
            "image_id",
            "patient_id",
            "lesion_uid",
        ]
    ].copy()

    table["true_class"] = [
        CLASSES[int(x)]
        for x in y_true
    ]

    table["c1_pred"] = [
        CLASSES[int(x)]
        for x in c1_pred
    ]

    table["f1_pred"] = [
        CLASSES[int(x)]
        for x in f1_pred
    ]

    table["c1_confidence"] = np.max(
        c1_probs,
        axis=1,
    )

    table["f1_confidence"] = np.max(
        f1_probs,
        axis=1,
    )

    table["c1_correct"] = (
        c1_pred == y_true
    )

    table["f1_correct"] = (
        f1_pred == y_true
    )

    for i, class_name in enumerate(CLASSES):
        table[
            f"c1_{class_name.lower()}_probability"
        ] = c1_probs[:, i]

        table[
            f"f1_{class_name.lower()}_probability"
        ] = f1_probs[:, i]

    return table


def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    c1_path = Path(
        args.c1_checkpoint
    )

    f1_path = Path(
        args.f1_checkpoint
    )

    test_csv = Path(
        args.test_csv
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        c1_path,
        f1_path,
        test_csv,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 80)
    print(
        "DERMASENSE C1 vs F1 PRODUCT EVALUATION"
    )
    print("=" * 80)

    print(f"Device: {device}")
    print(f"C1:     {c1_path}")
    print(f"F1:     {f1_path}")
    print(f"Test:   {test_csv}")

    test_df = pd.read_csv(
        test_csv
    )

    required = {
        "image_id",
        "patient_id",
        "lesion_uid",
        "native_diagnosis",
    }

    missing = (
        required
        - set(test_df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Test CSV missing columns: "
            f"{sorted(missing)}"
        )

    if len(test_df) != 352:
        raise RuntimeError(
            "Expected exactly 352 PAD-UFES "
            f"test images, got {len(test_df)}."
        )

    if not test_df["image_id"].is_unique:
        raise RuntimeError(
            "Test image IDs are not unique."
        )

    csv_targets = np.array(
        [
            CLASS_TO_INDEX[str(x)]
            for x in test_df["native_diagnosis"]
        ],
        dtype=int,
    )

    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    if len(dataset) != len(test_df):
        raise RuntimeError(
            "Dataset/test CSV length mismatch."
        )

    c1_model = load_model(
        c1_path,
        device,
    )

    f1_model = load_model(
        f1_path,
        device,
    )

    print(
        f"Test images: {len(dataset)}"
    )

    c1_logits, c1_targets = extract_logits(
        c1_model,
        dataset,
        device,
        args.batch_size,
    )

    f1_logits, f1_targets = extract_logits(
        f1_model,
        dataset,
        device,
        args.batch_size,
    )

    if not np.array_equal(
        c1_targets,
        f1_targets,
    ):
        raise RuntimeError(
            "C1/F1 target ordering differs."
        )

    if not np.array_equal(
        c1_targets,
        csv_targets,
    ):
        raise RuntimeError(
            "Dataset target ordering does not "
            "match data/splits/pad_ufes/test.csv."
        )

    c1_probs = softmax(
        c1_logits
    )

    f1_probs = softmax(
        f1_logits
    )

    print(
        "Prediction/probability recovery: PASS"
    )

    # ------------------------------------------------------------
    # Prediction table
    # ------------------------------------------------------------

    prediction_table = build_prediction_table(
        test_df,
        c1_targets,
        c1_probs,
        f1_probs,
    )

    prediction_path = (
        output_dir
        / "c1_f1_test_predictions.csv"
    )

    prediction_table.to_csv(
        prediction_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------

    c1_pred = np.argmax(
        c1_probs,
        axis=1,
    )

    f1_pred = np.argmax(
        f1_probs,
        axis=1,
    )

    c1_rows, c1_accuracy, c1_macro_f1 = (
        classification_metrics(
            c1_targets,
            c1_pred,
        )
    )

    f1_rows, f1_accuracy, f1_macro_f1 = (
        classification_metrics(
            f1_targets,
            f1_pred,
        )
    )

    metrics = []

    for c1_row, f1_row in zip(
        c1_rows,
        f1_rows,
    ):
        metrics.append({
            "class": c1_row["class"],
            "support": c1_row["support"],
            "c1_precision": c1_row["precision"],
            "c1_precision_wilson_low": (
                c1_row["precision_wilson_low"]
            ),
            "c1_precision_wilson_high": (
                c1_row["precision_wilson_high"]
            ),
            "c1_recall": c1_row["recall"],
            "c1_recall_wilson_low": (
                c1_row["recall_wilson_low"]
            ),
            "c1_recall_wilson_high": (
                c1_row["recall_wilson_high"]
            ),
            "c1_f1": c1_row["f1"],
            "f1_precision": f1_row["precision"],
            "f1_precision_wilson_low": (
                f1_row["precision_wilson_low"]
            ),
            "f1_precision_wilson_high": (
                f1_row["precision_wilson_high"]
            ),
            "f1_recall": f1_row["recall"],
            "f1_recall_wilson_low": (
                f1_row["recall_wilson_low"]
            ),
            "f1_recall_wilson_high": (
                f1_row["recall_wilson_high"]
            ),
            "f1_f1": f1_row["f1"],
        })

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_df.to_csv(
        output_dir
        / "classification_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------

    confusion_matrix_df(
        c1_targets,
        c1_pred,
    ).to_csv(
        output_dir
        / "confusion_matrix_c1.csv"
    )

    confusion_matrix_df(
        f1_targets,
        f1_pred,
    ).to_csv(
        output_dir
        / "confusion_matrix_f1.csv"
    )

    # ------------------------------------------------------------
    # High-risk evaluation
    # ------------------------------------------------------------

    c1_high = high_risk_metrics(
        c1_targets,
        c1_pred,
    )

    f1_high = high_risk_metrics(
        f1_targets,
        f1_pred,
    )

    high_risk_df = pd.DataFrame([
        {
            "metric": key,
            "c1": c1_high[key],
            "f1": f1_high[key],
        }
        for key in c1_high
    ])

    high_risk_df.to_csv(
        output_dir
        / "high_risk_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------

    c1_ece, c1_calibration = compute_ece(
        c1_probs,
        c1_targets,
    )

    f1_ece, f1_calibration = compute_ece(
        f1_probs,
        f1_targets,
    )

    c1_calibration["model"] = "C1"
    f1_calibration["model"] = "F1"

    calibration_df = pd.concat(
        [
            c1_calibration,
            f1_calibration,
        ],
        ignore_index=True,
    )

    calibration_df.to_csv(
        output_dir
        / "calibration.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary_path = (
        output_dir
        / "product_evaluation_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "DERMASENSE C1 vs F1 "
            "PRODUCT EVALUATION\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            "DATASET\n"
        )
        f.write("-" * 80 + "\n")
        f.write(
            "PAD-UFES exact test set: 352 images\n"
        )
        f.write(
            "Classes: ACK, BCC, MEL, NEV, SCC, SEK\n"
        )
        f.write(
            "High-risk definition: BCC + SCC + MEL\n"
        )
        f.write(
            "ACK is outside the primary high-risk bucket.\n"
        )
        f.write(
            "Malignant tie-break: "
            "minimum of MEL/BCC/SCC recall.\n\n"
        )

        f.write(
            "OVERALL\n"
        )
        f.write("-" * 80 + "\n")
        f.write(
            f"C1 accuracy:  {c1_accuracy:.6f}\n"
        )
        f.write(
            f"F1 accuracy:  {f1_accuracy:.6f}\n"
        )
        f.write(
            f"C1 macro-F1:  {c1_macro_f1:.6f}\n"
        )
        f.write(
            f"F1 macro-F1:  {f1_macro_f1:.6f}\n\n"
        )

        f.write(
            "HIGH-RISK\n"
        )
        f.write("-" * 80 + "\n")

        f.write(
            f"C1 pooled high-risk recall: "
            f"{c1_high['high_risk_recall']:.6f} "
            f"95% CI "
            f"[{c1_high['high_risk_recall_wilson_low']:.6f}, "
            f"{c1_high['high_risk_recall_wilson_high']:.6f}]\n"
        )

        f.write(
            f"F1 pooled high-risk recall: "
            f"{f1_high['high_risk_recall']:.6f} "
            f"95% CI "
            f"[{f1_high['high_risk_recall_wilson_low']:.6f}, "
            f"{f1_high['high_risk_recall_wilson_high']:.6f}]\n\n"
        )

        f.write(
            "Minimum malignant recall "
            "(MEL/BCC/SCC)\n"
        )

        f.write(
            f"C1: {c1_high['minimum_malignant_recall']:.6f}\n"
        )

        f.write(
            f"F1: {f1_high['minimum_malignant_recall']:.6f}\n\n"
        )

        f.write(
            "PER-CLASS RECALL\n"
        )
        f.write("-" * 80 + "\n")

        for class_name in CLASSES:
            c1_row = next(
                x for x in c1_rows
                if x["class"] == class_name
            )

            f1_row = next(
                x for x in f1_rows
                if x["class"] == class_name
            )

            f.write(
                f"{class_name} "
                f"(n={c1_row['support']}):\n"
            )

            f.write(
                f"  C1 recall = "
                f"{c1_row['recall']:.6f} "
                f"95% CI "
                f"[{c1_row['recall_wilson_low']:.6f}, "
                f"{c1_row['recall_wilson_high']:.6f}]\n"
            )

            f.write(
                f"  F1 recall = "
                f"{f1_row['recall']:.6f} "
                f"95% CI "
                f"[{f1_row['recall_wilson_low']:.6f}, "
                f"{f1_row['recall_wilson_high']:.6f}]\n"
            )

        f.write("\n")

        f.write(
            "CALIBRATION\n"
        )
        f.write("-" * 80 + "\n")
        f.write(
            f"C1 ECE (10 bins): {c1_ece:.6f}\n"
        )
        f.write(
            f"F1 ECE (10 bins): {f1_ece:.6f}\n\n"
        )

        f.write(
            "ACK FLAGGED ERRORS\n"
        )
        f.write("-" * 80 + "\n")

        ack_errors = prediction_table[
            (
                prediction_table["true_class"]
                == "ACK"
            )
            & (
                prediction_table["c1_pred"]
                != "ACK"
            )
            | (
                (
                    prediction_table["true_class"]
                    == "ACK"
                )
                & (
                    prediction_table["f1_pred"]
                    != "ACK"
                )
            )
        ].copy()

        if len(ack_errors) == 0:
            f.write(
                "No ACK misclassifications.\n"
            )
        else:
            for _, row in ack_errors.iterrows():
                f.write(
                    f"{row['image_id']} | "
                    f"C1={row['c1_pred']} | "
                    f"F1={row['f1_pred']}\n"
                )

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL")
    print("=" * 80)

    print(
        f"C1 accuracy: {c1_accuracy:.4f}"
    )
    print(
        f"F1 accuracy: {f1_accuracy:.4f}"
    )
    print(
        f"C1 macro-F1: {c1_macro_f1:.4f}"
    )
    print(
        f"F1 macro-F1: {f1_macro_f1:.4f}"
    )

    print()
    print("=" * 80)
    print("PER-CLASS RECALL")
    print("=" * 80)

    for class_name in CLASSES:
        c1_row = next(
            x for x in c1_rows
            if x["class"] == class_name
        )

        f1_row = next(
            x for x in f1_rows
            if x["class"] == class_name
        )

        print(
            f"{class_name:>3} | "
            f"C1={c1_row['recall']:.4f} "
            f"[{c1_row['recall_wilson_low']:.4f}, "
            f"{c1_row['recall_wilson_high']:.4f}] | "
            f"F1={f1_row['recall']:.4f} "
            f"[{f1_row['recall_wilson_low']:.4f}, "
            f"{f1_row['recall_wilson_high']:.4f}]"
        )

    print()
    print("=" * 80)
    print("HIGH-RISK")
    print("=" * 80)

    print(
        f"C1 pooled high-risk recall: "
        f"{c1_high['high_risk_recall']:.4f}"
    )

    print(
        f"F1 pooled high-risk recall: "
        f"{f1_high['high_risk_recall']:.4f}"
    )

    print(
        f"C1 minimum malignant recall: "
        f"{c1_high['minimum_malignant_recall']:.4f}"
    )

    print(
        f"F1 minimum malignant recall: "
        f"{f1_high['minimum_malignant_recall']:.4f}"
    )

    print()
    print("=" * 80)
    print("CALIBRATION")
    print("=" * 80)

    print(
        f"C1 ECE: {c1_ece:.6f}"
    )

    print(
        f"F1 ECE: {f1_ece:.6f}"
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"Predictions: {prediction_path}"
    )

    print(
        "Metrics:     "
        f"{output_dir / 'classification_metrics.csv'}"
    )

    print(
        "High-risk:  "
        f"{output_dir / 'high_risk_metrics.csv'}"
    )

    print(
        "Calibration: "
        f"{output_dir / 'calibration.csv'}"
    )

    print(
        f"Summary:     {summary_path}"
    )

    print()
    print("=" * 80)
    print(
        "PRODUCT EVALUATION COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
