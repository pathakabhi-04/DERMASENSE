"""
Build an EXTERNAL 6-class evaluation set for CV-4 from DDI, DDI-2 and
Fitzpatrick17k, mapped onto the PAD-UFES label space.

## Why

PAD-UFES's test split has **9 melanomas**. Its MEL recall of 0.6667 is
6/9, with a ~30-93% confidence interval — we do not currently have a
usable measurement of CV-4's melanoma recall
(`docs/cv_metrics_improvement_plan.md` Step 1). Nothing can be shown to
improve it until that changes.

The three clinical-photo sources already on disk carry real diagnosis
strings. Mapped onto PAD's six classes they yield ~100 melanomas instead
of 9.

## The mapping is PRE-COMMITTED here, before any evaluation is run

Ordered rules, first match wins, EXCLUDE rules first so known traps
cannot be silently absorbed by a looser rule below them. The mapping is
written and its class counts inspected BEFORE any model is scored, so it
cannot be tuned to produce a desired result.

Judgment calls, stated rather than buried:

  - **`porokeratosis actinic` is NOT actinic keratosis.** 101 Fitzpatrick
    images say "porokeratosis actinic"; porokeratosis is a distinct
    disorder of keratinization. Mapping it to ACK would nearly 10x that
    class with the wrong disease. Excluded explicitly.
  - **`squamous cell carcinoma in situ` (Bowen's) is EXCLUDED by
    default.** Whether PAD-UFES's SCC class subsumes Bowen's is not
    documented anywhere in this repo, and asserting it either way would
    be a guess. `--include-scc-in-situ` runs the sensitivity variant;
    both numbers get reported rather than one being chosen.
  - **Only MELANOCYTIC nevi map to NEV.** `epidermal nevus`,
    `naevus comedonicus` and `nevus lipomatosus` are keratinocytic or
    adnexal lesions, not melanocytic, and are excluded. Dysplastic and
    blue nevi are included as melanocytic.
  - **Melanoma in situ, acral lentiginous and nodular melanoma map to
    MEL.** They are melanoma subtypes. `acral melanotic macule` is not
    melanoma and is excluded.
  - Ambiguous keratoses (`benign keratosis`, `lichenoid keratosis`,
    `inverted follicular keratosis`, `solar lentigo`) are excluded
    rather than forced into SEK.

Anything with no PAD analogue is **excluded, not mapped** — the same
convention `train_referral_head.py` already uses for DF/VASC.

    PYTHONPATH=. python3 scripts/build_external_6class_eval.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "analysis/quality/mel_sensitivity/external_6class_eval_set.csv"

PAD_CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
EXCLUDE = "__EXCLUDE__"

# (regex, label). First match wins. EXCLUDE entries come first on purpose.
RULES: list[tuple[str, str]] = [
    # --- traps: must be caught before the looser rules below ---
    (r"porokeratosis", EXCLUDE),                    # NOT actinic keratosis
    (r"melanotic macule", EXCLUDE),                 # NOT melanoma
    (r"metastatic", EXCLUDE),
    (r"sebaceous carcinoma", EXCLUDE),
    (r"epidermal nevus|naevus comedonicus|nevus lipomatosus", EXCLUDE),  # not melanocytic
    (r"benign keratosis|lichenoid keratosis|inverted follicular keratosis", EXCLUDE),
    (r"solar lentigo", EXCLUDE),

    # --- the six PAD classes ---
    (r"melanoma", "MEL"),                           # incl. in situ, acral lentiginous, nodular
    (r"basal cell carcinoma", "BCC"),
    (r"squamous cell carcinoma", "SCC"),            # in-situ handled before this, see main()
    (r"seborrheic keratosis", "SEK"),
    (r"actinic keratosis", "ACK"),
    (r"nevus|nevi", "NEV"),                         # melanocytic only; others excluded above
]

SCC_IN_SITU = r"squamous cell carcinoma[, ]+in situ|squamous cell carcinoma in situ"


def normalise(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text)


def map_label(raw: str, include_scc_in_situ: bool) -> str:
    text = normalise(raw)
    if re.search(SCC_IN_SITU, text):
        return "SCC" if include_scc_in_situ else EXCLUDE
    for pattern, label in RULES:
        if re.search(pattern, text):
            return label
    return EXCLUDE


def load_sources() -> pd.DataFrame:
    frames = []

    ddi = pd.read_csv(REPO_ROOT / "data/raw/ddi/manifest.csv")
    ddi_dir = REPO_ROOT / "data/raw/ddi/images"
    frames.append(pd.DataFrame({
        "image_path": ddi["DDI_file"].apply(lambda f: str(ddi_dir / f)),
        "raw_label": ddi["disease"],
        "source": "ddi",
    }))

    ddi2 = pd.read_csv(REPO_ROOT / "data/raw/ddi2/manifest.csv")
    by_stem = {p.stem: p for p in (REPO_ROOT / "data/raw/ddi2/images").glob("*")}
    ddi2 = ddi2.assign(path=ddi2["photo_id"].astype(str).map(
        lambda s: str(by_stem[s]) if s in by_stem else None))
    ddi2 = ddi2[ddi2["path"].notna()]
    frames.append(pd.DataFrame({
        "image_path": ddi2["path"],
        "raw_label": ddi2["diagnosis_detailed"],
        "source": "ddi2",
    }))

    fitz = pd.read_csv(REPO_ROOT / "data/raw/fitzpatrick17k/manifest.csv")
    fitz = fitz[fitz["downloaded"] == True]  # noqa: E712
    fitz_dir = REPO_ROOT / "data/raw/fitzpatrick17k/images"
    frames.append(pd.DataFrame({
        "image_path": fitz["filename"].apply(lambda f: str(fitz_dir / f)),
        "raw_label": fitz["label"],
        "source": "fitzpatrick17k",
    }))

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-scc-in-situ", action="store_true",
                        help="sensitivity variant: treat Bowen's disease as SCC")
    args = parser.parse_args()

    data = load_sources()
    data["label"] = data["raw_label"].apply(lambda r: map_label(r, args.include_scc_in_situ))

    mapped = data[data["label"] != EXCLUDE].copy()
    print(f"{len(data)} images across 3 sources -> {len(mapped)} mapped, "
          f"{len(data) - len(mapped)} excluded (no PAD analogue)\n")

    table = pd.crosstab(mapped["label"], mapped["source"], margins=True, margins_name="TOTAL")
    print(table.to_string())

    print("\nwhat each class was built from (top raw strings):")
    for label in PAD_CLASSES:
        subset = mapped[mapped["label"] == label]
        if subset.empty:
            print(f"  {label}: EMPTY")
            continue
        top = subset["raw_label"].value_counts().head(4)
        print(f"  {label:4} n={len(subset):4}  " + "; ".join(f"{k} ({v})" for k, v in top.items()))

    thin = [c for c in PAD_CLASSES if (mapped["label"] == c).sum() < 30]
    if thin:
        print(f"\nTHIN CLASSES (n<30), interpret their recall with care: {', '.join(thin)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    mapped[["image_path", "label", "raw_label", "source"]].to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
