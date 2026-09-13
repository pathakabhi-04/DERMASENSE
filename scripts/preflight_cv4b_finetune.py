"""
Pre-flight gate for the CV-4b backbone fine-tune (design §9).

Everything here runs on CPU, locally, for free. **The GPU is not rented
until all checks pass.** The point is that the rented card never spends
paid time discovering something a laptop could have found.

Each check is a question that has burned real ML runs:

  1 split integrity     -- does the data say what we think it says?
  2 freeze mask         -- are we training what we meant to train?
  3 overfit-one-batch   -- can this loop learn AT ALL? (highest value)
  4 determinism         -- will the result be reproducible?
  5 checkpoint trip     -- does saving/loading preserve the model exactly?
  6 resume fidelity     -- can we survive a dead instance mid-run?
  7 pre-resize fidelity -- is the shrunk dataset faithful? (design §11)
  8 metric sanity       -- do the metrics do the obvious right thing?
  9 throughput          -- is the time/cost estimate roughly real?

    PYTHONPATH=. python3 scripts/preflight_cv4b_finetune.py
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from scripts.finetune_cv4b_backbone import (
    DEFAULT_BUNDLE,
    FOLDS,
    SPLITS,
    BinaryLesionDataset,
    Row,
    build_model,
    bundle_rows,
    isic_rows,
    make_sampler,
    non_isic_rows,
    save_run_checkpoint,
)
from src.data.transforms import build_eval_transform, build_train_transform
from src.risk.referral_head import ReferralHead
from src.training.reproducibility import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
PRERESIZED_ROOT = REPO_ROOT / "data/processed/cv4b_finetune_256"
SCRATCH = REPO_ROOT / "analysis/quality/mel_sensitivity/_preflight"

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(function):
        def wrapped():
            print(f"\n--- {name} ---")
            try:
                detail = function()
                results.append((name, True, detail or ""))
                print(f"PASS  {detail or ''}")
            except Exception as exc:  # noqa: BLE001 -- a failed check must not abort the gate
                results.append((name, False, str(exc)))
                print(f"FAIL  {exc}")
        return wrapped
    return decorator


def fixed_batches(count: int = 4, batch_size: int = 8) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Real images from the BUNDLE -- the same path the GPU host uses --
    with deterministic preprocessing, held in memory so the
    determinism/resume checks test the optimizer and RNG, not the loader."""
    rows = [row for row in bundle_rows(DEFAULT_BUNDLE, "pooled", "train")
            if row.source != "isic2019"][: count * batch_size]
    transform = build_eval_transform()
    images, labels = [], []
    for row in rows:
        with Image.open(row.image_path) as image:
            images.append(transform(image.convert("RGB")))
        labels.append(float(row.label))
    stacked = torch.stack(images)
    targets = torch.tensor(labels)
    return [
        (stacked[i * batch_size:(i + 1) * batch_size], targets[i * batch_size:(i + 1) * batch_size])
        for i in range(count)
    ]


def train_steps(model, head, optimizer, criterion, batches, steps: int) -> list[float]:
    model.train()
    for index in range(7):
        model.backbone.features[index].eval()
    losses = []
    for step in range(steps):
        images, labels = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(model.extract_features(images)).squeeze(1), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


@check("1 split integrity")
def check_splits() -> str:
    table = pd.read_csv(SPLITS)
    details = []
    for fold in FOLDS:
        spans = table.groupby("group")[fold].nunique()
        if (spans > 1).any():
            raise AssertionError(f"{fold}: {(spans > 1).sum()} groups span splits")
        if (table.groupby("image_path")[fold].nunique() > 1).any():
            raise AssertionError(f"{fold}: an image appears in two splits")
        if set(table[fold]) != {"train", "val", "test"}:
            raise AssertionError(f"{fold}: unexpected split values {set(table[fold])}")
        for split in ("train", "val"):
            subset = table[table[fold] == split]
            if subset["is_malignant"].nunique() < 2:
                raise AssertionError(f"{fold}/{split} has only one class")
        details.append(f"{fold} ok")

    missing = [path for path in table["image_path"] if not Path(path).exists()]
    if missing:
        raise AssertionError(f"{len(missing)} image files missing, e.g. {missing[0]}")

    sample = table["image_path"].sample(min(40, len(table)), random_state=0)
    for path in sample:
        with Image.open(path) as image:
            image.convert("RGB")
    return f"{len(table)} images, {table['group'].nunique()} groups; " + ", ".join(details)


@check("2 freeze mask")
def check_freeze() -> str:
    model, head, layer4 = build_model(torch.device("cpu"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = sum(p.numel() for p in layer4.parameters())
    if trainable != expected:
        raise AssertionError(f"{trainable} trainable model params, expected layer4's {expected}")
    frozen_total = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    head_params = sum(p.numel() for p in head.parameters())
    if head_params != 2048 + 1:
        raise AssertionError(f"binary head has {head_params} params, expected 2049")
    return f"trainable layer4 {trainable:,} + head {head_params:,}; frozen {frozen_total:,}"


@check("3 overfit one batch")
def check_overfit() -> str:
    """If the loop cannot memorise 16 images, something is wrong with the
    LR, the labels or the graph -- and it is far cheaper to learn that
    here than on a rented card."""
    seed_everything(0)
    model, head, layer4 = build_model(torch.device("cpu"))
    optimizer = torch.optim.AdamW(
        [{"params": layer4.parameters(), "lr": 1e-3},
         {"params": head.parameters(), "lr": 1e-3}]
    )
    batches = fixed_batches(count=1, batch_size=16)
    if len(set(batches[0][1].tolist())) < 2:
        raise AssertionError("overfit batch is single-class; check cannot detect learning")
    losses = train_steps(model, head, optimizer, nn.BCEWithLogitsLoss(), batches, steps=60)
    if losses[-1] >= 0.05:
        raise AssertionError(
            f"loss only reached {losses[-1]:.4f} after 60 steps (start {losses[0]:.4f}); "
            "the training loop cannot memorise 16 images"
        )
    return f"loss {losses[0]:.4f} -> {losses[-1]:.6f} in 60 steps"


@check("4 determinism")
def check_determinism() -> str:
    runs = []
    for _ in range(2):
        seed_everything(123)
        model, head, layer4 = build_model(torch.device("cpu"))
        optimizer = torch.optim.AdamW(
            [{"params": layer4.parameters(), "lr": 1e-5},
             {"params": head.parameters(), "lr": 1e-4}]
        )
        runs.append(train_steps(model, head, optimizer, nn.BCEWithLogitsLoss(),
                                fixed_batches(count=2, batch_size=8), steps=5))
    difference = max(abs(a - b) for a, b in zip(*runs))
    if difference > 1e-6:
        raise AssertionError(f"two identically-seeded runs diverged by {difference:.2e}")
    return f"max loss difference over 5 steps: {difference:.2e}"


@check("5 checkpoint round-trip")
def check_checkpoint() -> str:
    seed_everything(7)
    model, head, _ = build_model(torch.device("cpu"))
    batch = fixed_batches(count=1, batch_size=8)[0][0]

    model.eval(); head.eval()
    with torch.no_grad():
        before = head(model.extract_features(batch))

    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "roundtrip.pt"
    save_run_checkpoint(path, model=model.state_dict(), head=head.state_dict(), epoch=1)

    reloaded_model, reloaded_head, _ = build_model(torch.device("cpu"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    reloaded_model.load_state_dict(payload["model"])
    reloaded_head.load_state_dict(payload["head"])
    reloaded_model.eval(); reloaded_head.eval()
    with torch.no_grad():
        after = reloaded_head(reloaded_model.extract_features(batch))

    if not torch.equal(before, after):
        raise AssertionError(f"predictions changed across save/load by "
                             f"{(before - after).abs().max().item():.2e}")
    return "predictions bit-identical after save/load"


@check("6 resume fidelity")
def check_resume() -> str:
    """4 steps straight through must equal 2 steps + checkpoint + 2 steps,
    including optimizer and RNG state. A resume that silently restarts is
    worse than no resume."""
    batches = fixed_batches(count=2, batch_size=8)
    criterion = nn.BCEWithLogitsLoss()

    seed_everything(99)
    model_a, head_a, layer4_a = build_model(torch.device("cpu"))
    optimizer_a = torch.optim.AdamW(
        [{"params": layer4_a.parameters(), "lr": 1e-5}, {"params": head_a.parameters(), "lr": 1e-4}]
    )
    straight = train_steps(model_a, head_a, optimizer_a, criterion, batches, steps=4)

    seed_everything(99)
    model_b, head_b, layer4_b = build_model(torch.device("cpu"))
    optimizer_b = torch.optim.AdamW(
        [{"params": layer4_b.parameters(), "lr": 1e-5}, {"params": head_b.parameters(), "lr": 1e-4}]
    )
    train_steps(model_b, head_b, optimizer_b, criterion, batches, steps=2)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "resume.pt"
    save_run_checkpoint(
        path, model=model_b.state_dict(), head=head_b.state_dict(),
        optimizer=optimizer_b.state_dict(), rng={"torch": torch.get_rng_state(),
                                                 "numpy": np.random.get_state(),
                                                 "python": __import__("random").getstate(),
                                                 "cuda": None},
        epoch=1,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    model_c, head_c, layer4_c = build_model(torch.device("cpu"))
    model_c.load_state_dict(payload["model"])
    head_c.load_state_dict(payload["head"])
    optimizer_c = torch.optim.AdamW(
        [{"params": layer4_c.parameters(), "lr": 1e-5}, {"params": head_c.parameters(), "lr": 1e-4}]
    )
    optimizer_c.load_state_dict(payload["optimizer"])
    resumed = train_steps(model_c, head_c, optimizer_c, criterion,
                          batches[2 % len(batches):] + batches[:2 % len(batches)], steps=2)

    difference = max(abs(a - b) for a, b in zip(straight[2:], resumed))
    if difference > 1e-6:
        raise AssertionError(f"resumed losses diverge from straight-through by {difference:.2e}")

    weight_gap = max(
        (pa - pc).abs().max().item()
        for pa, pc in zip(model_a.parameters(), model_c.parameters())
    )
    if weight_gap > 1e-6:
        raise AssertionError(f"resumed weights differ by {weight_gap:.2e}")
    return f"loss diff {difference:.2e}, weight diff {weight_gap:.2e}"


@check("7 pre-resize fidelity")
def check_preresize() -> str:
    """Does pre-resizing preserve what the model actually sees?

    Gated on PAIRED statistics -- feature cosine similarity, Spearman
    correlation of referral scores, decision-flip rate -- comparing each
    image against itself.

    An earlier version of this check gated on the gap between two
    independent AUC estimates with a 0.01 tolerance. That was
    unmeasurable: at n=240 the standard error of AUC alone is ~0.04, so
    the tolerance sat far inside the noise, and the check both failed
    faithful configurations and ranked them non-monotonically (PNG, the
    most faithful variant by every paired measure, scored a *worse* AUC
    gap than a lossier one). AUC gap is still reported, as context only.
    """
    manifest_path = PRERESIZED_ROOT / "manifest.csv"
    if not manifest_path.exists():
        raise AssertionError(
            f"no pre-resized manifest at {manifest_path}. Either build it "
            "(scripts/preresize_cv4b_dataset.py) and re-run, or upload "
            "full-resolution data instead. This check must not be skipped."
        )
    manifest = pd.read_csv(manifest_path)
    resized_by_original = dict(zip(manifest["image_path"], manifest["resized_path"]))

    model, _head, _ = build_model(torch.device("cpu"))
    model.eval()
    shipped_head = ReferralHead.load(REPO_ROOT / "checkpoints/referral_head/referral_head.json")
    rows = non_isic_rows("pooled", "val")[:240]
    missing = [row.image_path for row in rows if row.image_path not in resized_by_original]
    if missing:
        raise AssertionError(f"{len(missing)} images absent from the manifest, e.g. {missing[0]}")

    transform = build_eval_transform()

    def features_for(paths) -> np.ndarray:
        collected = []
        for start in range(0, len(paths), 24):
            tensors = []
            for path in paths[start:start + 24]:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            with torch.no_grad():
                collected.append(model.extract_features(torch.stack(tensors)).numpy())
        return np.vstack(collected)

    original = features_for([row.image_path for row in rows])
    resized = features_for([resized_by_original[row.image_path] for row in rows])

    cosine = float(np.mean(
        np.sum(original * resized, axis=1)
        / (np.linalg.norm(original, axis=1) * np.linalg.norm(resized, axis=1))
    ))
    original_scores = np.array([shipped_head.decide(v).probability for v in original])
    resized_scores = np.array([shipped_head.decide(v).probability for v in resized])
    spearman = float(spearmanr(original_scores, resized_scores).statistic)
    flipped = int(((original_scores >= shipped_head.threshold)
                   != (resized_scores >= shipped_head.threshold)).sum())
    flip_rate = flipped / len(rows)

    labels = np.array([row.label for row in rows])
    auc_gap = (abs(roc_auc_score(labels, original_scores) - roc_auc_score(labels, resized_scores))
               if len(set(labels)) > 1 else float("nan"))

    failures = []
    if cosine < 0.99:
        failures.append(f"feature cosine {cosine:.4f} < 0.99")
    if spearman < 0.98:
        failures.append(f"score spearman {spearman:.4f} < 0.98")
    if flip_rate > 0.05:
        failures.append(f"{flip_rate:.1%} of referral decisions flip > 5%")
    if failures:
        raise AssertionError("; ".join(failures) + " -- pre-resizing is not faithful enough")

    return (f"cosine {cosine:.4f}, spearman {spearman:.4f}, flips {flipped}/{len(rows)} "
            f"({flip_rate:.1%}); AUC gap {auc_gap:.4f} (context only, SE~0.04 at this n)")


@check("8 metric sanity")
def check_metrics() -> str:
    labels = np.array([0, 0, 0, 1, 1, 1])
    if roc_auc_score(labels, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) != 1.0:
        raise AssertionError("perfect separation did not score AUC 1.0")
    if roc_auc_score(labels, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) != 0.0:
        raise AssertionError("perfectly inverted scores did not give AUC 0.0")

    rows = [Row("x", 1, "ddi", False)] * 10 + [Row("x", 0, "isic2019", False)] * 90
    generator = torch.Generator(); generator.manual_seed(0)
    sampler = make_sampler(rows, generator)
    drawn = np.array(list(sampler))
    non_isic_share = float((drawn < 10).mean())
    if not 0.40 <= non_isic_share <= 0.60:
        raise AssertionError(
            f"sampler drew {non_isic_share:.2%} non-ISIC, expected ~50% -- "
            "the mixing ratio is the most likely cause of a false-negative run"
        )
    return f"AUC endpoints correct; sampler drew {non_isic_share:.1%} non-ISIC"


@check("9 throughput extrapolation")
def check_throughput() -> str:
    """Times the JPEG-decode path, measured to be the bottleneck (a
    page-cache re-read runs at the same rate, so this is CPU-bound, not
    disk-bound). The rate transfers to the GPU box -- decode is CPU work
    there too -- scaled by its core count.

    Timing starts AFTER the first batch: DataLoader worker spawn is a
    fixed ~1-2s cost that swamped an earlier 128-image measurement and
    produced a 3x-pessimistic number."""
    original_rows = non_isic_rows("pooled", "train")[:512]
    bundle_clinical = [row for row in bundle_rows(DEFAULT_BUNDLE, "pooled", "train")
                       if row.source != "isic2019"][:512]
    rows = original_rows
    measurements = {}
    for label, paths in (
        ("original", [row.image_path for row in original_rows]),
        ("bundle", [row.image_path for row in bundle_clinical]),
    ):
        loader = DataLoader(
            BinaryLesionDataset(
                [Row(path, row.label, row.source, row.is_melanoma)
                 for path, row in zip(paths, rows)],
                build_train_transform(),
            ),
            batch_size=32, num_workers=4, shuffle=False,
        )
        iterator = iter(loader)
        seen = len(next(iterator)["label"])  # warm-up batch, excluded from timing
        start = time.perf_counter()
        for batch in iterator:
            seen += len(batch["label"])
        measurements[label] = (seen - 32) / (time.perf_counter() - start)

    train_size = len(bundle_rows(DEFAULT_BUNDLE, "pooled", "train"))
    rate = measurements["bundle"]
    if rate < 40:
        raise AssertionError(
            f"only {rate:.0f} img/s decode throughput on this machine; an epoch "
            f"would spend {(train_size / rate) / 60:.0f} min on data loading alone"
        )
    return "; ".join(
        f"{label} {value:.0f} img/s -> {(train_size / value) / 60:.1f} min/epoch here"
        for label, value in measurements.items()
    )


@check("10 bundle matches splits")
def check_bundle() -> str:
    """The bundle is what ships. Verify it agrees with the split CSV it was
    built from, so a stale bundle cannot silently train on old assignments."""
    dataset_csv = DEFAULT_BUNDLE / "dataset.csv"
    if not dataset_csv.exists():
        raise AssertionError(
            f"no bundle at {DEFAULT_BUNDLE}; run scripts/build_gpu_bundle.py"
        )
    bundle = pd.read_csv(dataset_csv, keep_default_na=False)
    splits = pd.read_csv(SPLITS)

    clinical = bundle[bundle["source"] != "isic2019"]
    if len(clinical) != len(splits):
        raise AssertionError(
            f"bundle has {len(clinical)} clinical images, split CSV has {len(splits)} "
            "-- rebuild the bundle"
        )
    for fold in FOLDS:
        bundle_counts = clinical[fold].value_counts().to_dict()
        split_counts = splits[fold].value_counts().to_dict()
        if bundle_counts != split_counts:
            raise AssertionError(f"fold {fold} differs: bundle {bundle_counts} vs {split_counts}")

    isic = bundle[bundle["source"] == "isic2019"]
    for split in ("train", "val", "test"):
        expected = len(isic_rows(split))
        actual = int((isic["isic_split"] == split).sum())
        if actual != expected:
            raise AssertionError(f"ISIC {split}: bundle {actual}, CVDataset {expected}")

    missing = [p for p in bundle["relative_path"].sample(200, random_state=0)
               if not (DEFAULT_BUNDLE / p).exists()]
    if missing:
        raise AssertionError(f"{len(missing)} sampled bundle files absent, e.g. {missing[0]}")
    return (f"{len(bundle)} rows ({len(isic)} ISIC + {len(clinical)} clinical) "
            f"agree with {SPLITS.name}")


def main() -> None:
    print("=" * 72)
    print("CV-4b FINE-TUNE PRE-FLIGHT  (all checks must pass before renting a GPU)")
    print("=" * 72)

    for function in (check_splits, check_freeze, check_overfit, check_determinism,
                     check_checkpoint, check_resume, check_preresize, check_metrics,
                     check_throughput, check_bundle):
        function()

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<26} {detail}")
    print("=" * 72)
    print(f"{passed}/{len(results)} checks passed")
    if passed != len(results):
        print("\nDO NOT RENT THE GPU until every check above passes.")
        raise SystemExit(1)
    print("\nCleared for the GPU run.")


if __name__ == "__main__":
    main()
