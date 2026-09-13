"""
Extract frozen 2048-d backbone features for DDI, DDI-2, and Fitzpatrick17k
(atlas subset), caching them the same way ISIC's already-cached
`analysis/scc_bcc/isic2019_*_backbone_features.npz` files are shaped
(`features`, `targets` arrays) -- so the CV-4b refit
(`cv4b_retrain_scope.md`) can load non-ISIC sources exactly like ISIC.

Reuses `load_backbone_for_features()` from `evaluate_domain_shift.py`
(construction already verified against `NativePredictor.from_checkpoint`)
rather than reconstructing the model again.

    PYTHONPATH=. python3 scripts/extract_domain_shift_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_domain_shift import _read_tensor, load_backbone_for_features
from scripts.run_domain_shift_check import load_ddi, load_ddi2, load_fitzpatrick

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "analysis/quality/mel_sensitivity/features"


def extract(model: torch.nn.Module, device: str, paths, label: str) -> tuple[np.ndarray, np.ndarray]:
    features, ok_mask = [], []
    for i, path in enumerate(paths, 1):
        tensor = _read_tensor(Path(path))
        if tensor is None:
            ok_mask.append(False)
            continue
        with torch.no_grad():
            feat = model.extract_features(tensor.unsqueeze(0).to(device))
        features.append(feat.cpu().numpy().reshape(-1))
        ok_mask.append(True)
        if i % 100 == 0 or i == len(paths):
            print(f"  [{label}] {i}/{len(paths)}", flush=True)
    return np.stack(features), np.array(ok_mask)


def main() -> None:
    device = "cpu"
    print("Loading backbone...")
    model = load_backbone_for_features(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, loader in (("ddi", load_ddi), ("ddi2", load_ddi2), ("fitzpatrick17k", load_fitzpatrick)):
        df = loader()
        print(f"\n[{name}] {len(df)} rows")
        features, ok_mask = extract(model, device, df["image_path"].tolist(), name)
        kept = df[ok_mask].reset_index(drop=True)
        assert len(kept) == len(features), f"{name}: feature/label count mismatch"

        out = OUT_DIR / f"{name}_features.npz"
        np.savez(
            out,
            features=features,
            is_malignant=kept["is_malignant"].to_numpy(),
            is_melanoma=kept["is_melanoma"].to_numpy(),
            image_path=kept["image_path"].astype(str).to_numpy(),
        )
        print(f"  {len(features)}/{len(df)} images -> {out}")


if __name__ == "__main__":
    main()
