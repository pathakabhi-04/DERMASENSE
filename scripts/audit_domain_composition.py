"""
Compositional audit: why does the native classifier collapse severely on
DDI/DDI-2 (melanoma routing 0.72 -> 0.25-0.33) but only mildly on
Fitzpatrick17k (0.69)? (domain_shift_second_source.md)

Ruled out already: raw image resolution/file size is not it -- Fitzpatrick
images are systematically SMALLER/more compressed than DDI's, which would
predict worse transfer, not better, if quality alone were the driver.

The remaining, unconfirmed hypothesis from that doc: Fitzpatrick17k's atlas
images are curated teaching examples (edited/cropped to show a lesion
clearly); DDI/DDI-2 are less-curated real clinical patient photography.
If atlas curation happens to produce framing closer to how ISIC/PAD-UFES
training images were captured (lesion-centered, one lesion filling a
consistent fraction of the frame) than genuine patient photos do, that
would explain the differential transfer without the backbone being
"more broken" on real photos -- it would mean Fitzpatrick is a LESS
representative stand-in for deployment, not a better one.

This audit tests that directly using CV-3 -- ALREADY VALIDATED,
ALREADY-DEPLOYED segmentation -- rather than inventing new tooling:

  - mask_area_fraction (src/segmentation/inference.py::mask_evidence):
    what fraction of the frame the lesion occupies. If DDI's distribution
    sits far lower than Fitzpatrick's and PAD-UFES's, that is direct
    evidence DDI images are wider/less lesion-centered.
  - mask_degenerate / mask_touches_border (same function): does CV-3
    even find a single clear lesion in these images at all. A source
    with many degenerate masks is a source where nothing downstream
    (native classifier OR referral head) can reasonably be expected to
    perform on, because CV-3 could not agree on what "the lesion" is.
  - crop_blur / crop_contrast (src/quality/signals.py, identical
    functions and identical role to what CV-8's quality_flags already
    disclose for the deployed product): whether DDI images are
    systematically blurrier/lower-contrast in the pipeline's own
    existing terms, not a new metric invented for this audit.

Treats each image as its own "candidate" the way an already-detected,
single-candidate PAD-UFES/DDI/Fitzpatrick image is treated everywhere
else in this project's domain-shift work: crop_and_normalize() is called
with the full-frame box (0.5, 0.5, 1.0, 1.0) and zero margin, i.e. "the
whole photo is the crop" -- identical preprocessing to what CV-3 would
see for these images inside the real orchestrator if a detector had
already produced a single full-frame box.

PAD-UFES's own test split is included as the reference: it is the
domain the deployed checkpoint's own product convention ("whole photo
is the crop") was built around, so it is the right baseline to compare
DDI/DDI-2/Fitzpatrick against -- not an assumption of what "good"
composition looks like, a measurement of what the shipped model already
works reasonably on.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.inference.crop_normalize import crop_and_normalize
from src.quality.signals import blur_signal, contrast_signal
from src.segmentation.inference import load_segmentation_model, mask_evidence, predict_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
FULL_FRAME_BOX = (0.5, 0.5, 1.0, 1.0)  # (x_center, y_center, w, h), normalized


def audit_one_image(model: torch.nn.Module, device: torch.device, path: Path) -> dict | None:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        return None

    cv3_tensor, _ = crop_and_normalize(image_bgr, FULL_FRAME_BOX, margin=0.0)
    mask = predict_mask(model, cv3_tensor, device)
    evidence = mask_evidence(mask)

    crop_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    blur = blur_signal(crop_rgb).score
    contrast = contrast_signal(crop_rgb).score

    height, width = image_bgr.shape[:2]
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "mask_area_fraction": evidence["mask_area_fraction"],
        "mask_degenerate": evidence["mask_degenerate"],
        "mask_touches_border": evidence["mask_touches_border"],
        "crop_blur": blur,
        "crop_contrast": contrast,
    }


def audit_source(model: torch.nn.Module, device: torch.device, paths: list[Path], label: str) -> pd.DataFrame:
    rows = []
    for i, path in enumerate(paths, 1):
        result = audit_one_image(model, device, path)
        if result is not None:
            result["source"] = label
            rows.append(result)
        if i % 100 == 0 or i == len(paths):
            print(f"  [{label}] {i}/{len(paths)}", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    print(f"\n{'='*78}")
    print(f"{'source':<14}{'n':>6}{'degenerate':>12}{'touches_edge':>14}{'area_frac(med)':>16}"
          f"{'blur(med)':>12}{'contrast(med)':>14}")
    print("-" * 78)
    for source, sub in df.groupby("source"):
        print(f"{source:<14}{len(sub):>6}"
              f"{sub['mask_degenerate'].mean():>12.3f}"
              f"{sub['mask_touches_border'].mean():>14.3f}"
              f"{sub['mask_area_fraction'].median():>16.4f}"
              f"{sub['crop_blur'].median():>12.4f}"
              f"{sub['crop_contrast'].median():>14.4f}")


def main() -> None:
    device = torch.device("cpu")
    print("Loading CV-3 segmentation model...")
    model = load_segmentation_model(SEGMENTATION_CHECKPOINT, device)

    sources: dict[str, list[Path]] = {}

    ddi_dir = REPO_ROOT / "data/raw/ddi/images"
    if ddi_dir.exists():
        sources["DDI"] = sorted(ddi_dir.glob("*.png"))

    ddi2_dir = REPO_ROOT / "data/raw/ddi2/images"
    if ddi2_dir.exists():
        sources["DDI-2"] = sorted(ddi2_dir.glob("*"))

    fitz_dir = REPO_ROOT / "data/raw/fitzpatrick17k/images"
    if fitz_dir.exists():
        sources["Fitzpatrick17k"] = sorted(fitz_dir.glob("*.jpg"))

    pad_test = REPO_ROOT / "data/splits/pad_ufes/test.csv"
    if pad_test.exists():
        pad_df = pd.read_csv(pad_test)
        sources["PAD-UFES (reference)"] = [Path(p) for p in pad_df["image_path"]]

    all_frames = []
    for label, paths in sources.items():
        print(f"\n[{label}] {len(paths)} images")
        df = audit_source(model, device, paths, label)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    summarize(combined)

    out = REPO_ROOT / "analysis/quality/mel_sensitivity/domain_composition_audit.csv"
    combined.drop(columns=["path"]).to_csv(out, index=False)
    print(f"\nper-image results -> {out}")


if __name__ == "__main__":
    main()
