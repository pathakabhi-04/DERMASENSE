from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint UMAP analysis of SCC/BCC features."
    )

    parser.add_argument(
        "--input-dir",
        default="analysis/scc_bcc",
    )

    parser.add_argument(
        "--output-dir",
        default="analysis/scc_bcc",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_features(path: Path, domain: str):
    data = np.load(path)

    features = data["features"]
    targets = data["targets"]

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D feature matrix in {path}, "
            f"got {features.shape}."
        )

    # PAD-UFES:
    # BCC = 1
    # SCC = 4
    #
    # ISIC2019:
    # BCC = 1
    # SCC = 6
    if domain == "pad_ufes":
        bcc_index = 1
        scc_index = 4
    elif domain == "isic2019":
        bcc_index = 1
        scc_index = 6
    else:
        raise ValueError(
            f"Unknown domain: {domain}"
        )

    mask = np.logical_or(
        targets == bcc_index,
        targets == scc_index,
    )

    features = features[mask]
    targets = targets[mask]

    diagnosis = np.where(
        targets == bcc_index,
        "BCC",
        "SCC",
    )

    domains = np.full(
        len(features),
        domain,
        dtype=object,
    )

    return features, diagnosis, domains


def save_plot(
    embedding,
    diagnosis,
    domains,
    output_path,
    title,
    color_by,
):
    plt.figure(figsize=(10, 8))

    if color_by == "diagnosis":
        labels = ("BCC", "SCC")
    else:
        labels = ("pad_ufes", "isic2019")

    for label in labels:
        mask = (
            diagnosis == label
            if color_by == "diagnosis"
            else domains == label
        )

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.55,
            label=label,
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pad_path = (
        input_dir
        / "pad_ufes_test_scc_bcc_features.npz"
    )

    isic_path = (
        input_dir
        / "isic2019_test_scc_bcc_features.npz"
    )

    if not pad_path.exists():
        raise FileNotFoundError(
            f"Missing feature file: {pad_path}"
        )

    if not isic_path.exists():
        raise FileNotFoundError(
            f"Missing feature file: {isic_path}"
        )

    print("=" * 70)
    print("DERMASENSE SCC/BCC JOINT UMAP ANALYSIS")
    print("=" * 70)

    pad_features, pad_diagnosis, pad_domains = (
        load_features(
            pad_path,
            "pad_ufes",
        )
    )

    isic_features, isic_diagnosis, isic_domains = (
        load_features(
            isic_path,
            "isic2019",
        )
    )

    print()
    print("INPUT DATA")
    print(
        f"PAD-UFES SCC/BCC: "
        f"{len(pad_features)}"
    )
    print(
        f"ISIC2019 SCC/BCC: "
        f"{len(isic_features)}"
    )

    features = np.concatenate(
        [
            pad_features,
            isic_features,
        ],
        axis=0,
    )

    diagnosis = np.concatenate(
        [
            pad_diagnosis,
            isic_diagnosis,
        ]
    )

    domains = np.concatenate(
        [
            pad_domains,
            isic_domains,
        ]
    )

    print(
        f"Combined features: "
        f"{features.shape}"
    )

    # Standardize before UMAP so dimensions with
    # unusually large scale do not dominate.
    print()
    print("STANDARDIZING FEATURES")

    scaler = StandardScaler()

    features_scaled = scaler.fit_transform(
        features
    )

    print("Standardization complete.")

    print()
    print("RUNNING JOINT UMAP")

    reducer = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=args.seed,
    )

    embedding = reducer.fit_transform(
        features_scaled
    )

    print(
        f"Embedding shape: "
        f"{embedding.shape}"
    )

    # ------------------------------------------------------------
    # Plot 1: diagnosis
    # ------------------------------------------------------------

    diagnosis_path = (
        output_dir
        / "scc_bcc_umap_by_diagnosis.png"
    )

    save_plot(
        embedding,
        diagnosis,
        domains,
        diagnosis_path,
        "Joint UMAP: SCC vs BCC",
        "diagnosis",
    )

    print(
        f"Saved: {diagnosis_path}"
    )

    # ------------------------------------------------------------
    # Plot 2: domain
    # ------------------------------------------------------------

    domain_path = (
        output_dir
        / "scc_bcc_umap_by_domain.png"
    )

    save_plot(
        embedding,
        diagnosis,
        domains,
        domain_path,
        "Joint UMAP: PAD-UFES vs ISIC2019",
        "domain",
    )

    print(
        f"Saved: {domain_path}"
    )

    # ------------------------------------------------------------
    # Save numerical embedding
    # ------------------------------------------------------------

    embedding_path = (
        output_dir
        / "scc_bcc_umap_embedding.npz"
    )

    np.savez_compressed(
        embedding_path,
        embedding=embedding,
        diagnosis=diagnosis,
        domains=domains,
    )

    print(
        f"Saved: {embedding_path}"
    )

    print()
    print("=" * 70)
    print("UMAP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
