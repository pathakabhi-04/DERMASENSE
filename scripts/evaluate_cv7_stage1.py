"""
CV-7 Stage 1 evaluation: clinical signal validity.

This is question 2 from docs/cv7_temporal_technical_spec.md's
evaluation plan, run now that the malignant-enrichment pull makes it
answerable (99 malignant-outcome lesions with visit pairs staged).
Question 1 (pipeline correctness) was already answered by the
per-module validation done while building calibration.py/measurement.py/
delta.py.

PRE-COMMITTED QUESTION (written before running, not adjusted after
seeing results): among the staged sample, do lesion pairs with a
malignant outcome diagnosis (melanoma/BCC/SCC) show a higher rate of
non-STABLE verdicts, and/or higher delta magnitude, than pairs with a
benign outcome?

PRE-COMMITTED SAMPLE: ALL malignant-outcome lesions with >=2 visits in
the staged data (the full enrichment pull, not a further sub-sample --
there are few enough that no further bounding is needed), vs. a
bounded random sample of up to 300 benign-outcome lesions (seed=17,
same bound used for the delta-threshold calibration run) -- a fair
comparison group, not cherry-picked.

PRE-COMMITTED DECISION RULE:
  1. NO_PRIOR_DATA pairs (no mask found in >=1 visit) are excluded from
     both the verdict-rate and magnitude comparisons -- that reflects
     CV-3 mask availability, not a temporal signal, and would confound
     the comparison if left in.
  2. Non-STABLE rate (GROWING | SHRINKING | CHANGED_COLOR, out of all
     non-NO_PRIOR_DATA pairs) compared via Fisher's exact test.
  3. Delta magnitude distributions (non-NO_PRIOR_DATA pairs only)
     compared via Mann-Whitney U (does not assume normality).
  4. Significance threshold: p < 0.05, two-sided, standard and set
     before looking at any result.
  5. Interpretation is fixed in advance:
     - Either test significant at p<0.05, in the expected direction
       (malignant higher) -> Stage 1 shows a real, detectable signal;
       sufficient evidence to proceed to CV-8 integration as designed,
       no Stage 2 needed on this basis.
     - Neither test significant -> Stage 1's classical measurements do
       not discriminate on this data. This does NOT automatically
       trigger Stage 2 (per the technical spec's own anti-rabbit-hole
       boundary) -- it must be weighed against known instrument
       limitations already documented (calibration's 4.0% coverage,
       color's lighting-noise confound) before concluding the
       underlying idea is insufficient vs. the instrument being noisy.
     - Significant in the WRONG direction (malignant lower) is reported
       as-is, not explained away.

This script does not modify src/temporal/. It reports; interpretation
and the resulting decision are documented separately in
analysis/quality/cv7_temporal_data/stage1_evaluation_result.md.
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from scripts.calibrate_cv7_thresholds import (
    ARCHIVE_ROOT,
    CHECKPOINT,
    MALIGNANT_DIAGNOSES,
    ZIP_PATH,
    find_visit_pairs,
    load_diagnosis_lookup,
    load_staged_participants,
)
from src.segmentation.inference import load_segmentation_model
from src.temporal.delta import TemporalVerdict
from src.temporal.pipeline import TemporalPipeline

SEED = 17
MAX_BENIGN_PAIRS = 300

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "analysis/quality/cv7_temporal_data"

NON_STABLE = {TemporalVerdict.GROWING, TemporalVerdict.SHRINKING, TemporalVerdict.CHANGED_COLOR}


def _read(zf: zipfile.ZipFile, path: str) -> np.ndarray:
    import cv2

    with zf.open(path) as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    staged_participants = load_staged_participants()
    device = torch.device("cpu")
    pipeline = TemporalPipeline(segmenter=load_segmentation_model(CHECKPOINT, device), device=device)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        diagnosis_lookup = load_diagnosis_lookup(zf)
        all_pairs = find_visit_pairs(zf, staged_participants)

        malignant_pairs = [p for p in all_pairs if diagnosis_lookup.get(p[0], "unknown") in MALIGNANT_DIAGNOSES]
        benign_pairs = [
            p for p in all_pairs
            if diagnosis_lookup.get(p[0], "unknown") not in MALIGNANT_DIAGNOSES
            and diagnosis_lookup.get(p[0], "unknown") != "unknown"
        ]
        random.seed(SEED)
        random.shuffle(benign_pairs)
        benign_pairs = benign_pairs[:MAX_BENIGN_PAIRS]

        def run_group(pairs, label):
            rows = []
            for key, earlier_path, later_path in pairs:
                earlier_img = _read(zf, earlier_path)
                later_img = _read(zf, later_path)
                result = pipeline.assess_pair(earlier_img, later_img)
                rows.append({"lesion": key, "verdict": result.verdict, "magnitude": result.magnitude})
            print(f"{label}: {len(rows)} pairs processed")
            return rows

        malignant_rows = run_group(malignant_pairs, "malignant")
        benign_rows = run_group(benign_pairs, "benign")

    def summarize(rows):
        scored = [r for r in rows if r["verdict"] != TemporalVerdict.NO_PRIOR_DATA]
        non_stable = [r for r in scored if r["verdict"] in NON_STABLE]
        magnitudes = [r["magnitude"] for r in scored]
        return {
            "n_total": len(rows),
            "n_no_prior_data": len(rows) - len(scored),
            "n_scored": len(scored),
            "n_non_stable": len(non_stable),
            "non_stable_rate": len(non_stable) / len(scored) if scored else None,
            "magnitudes": magnitudes,
        }

    malignant_summary = summarize(malignant_rows)
    benign_summary = summarize(benign_rows)

    # Fisher's exact on non-STABLE vs STABLE counts.
    table = [
        [malignant_summary["n_non_stable"], malignant_summary["n_scored"] - malignant_summary["n_non_stable"]],
        [benign_summary["n_non_stable"], benign_summary["n_scored"] - benign_summary["n_non_stable"]],
    ]
    fisher_odds, fisher_p = stats.fisher_exact(table)

    # Mann-Whitney U on magnitude distributions.
    mw_stat, mw_p = (None, None)
    if malignant_summary["magnitudes"] and benign_summary["magnitudes"]:
        mw_stat, mw_p = stats.mannwhitneyu(
            malignant_summary["magnitudes"], benign_summary["magnitudes"], alternative="two-sided"
        )

    report = []
    report.append(f"seed={SEED}, max_benign_pairs={MAX_BENIGN_PAIRS}")
    report.append(f"malignant: {malignant_summary}")
    report.append(f"benign: {benign_summary}")
    report.append(f"fisher exact (non-stable vs stable, malignant vs benign): odds={fisher_odds:.4f} p={fisher_p:.4f}")
    report.append(f"mann-whitney U on magnitude: stat={mw_stat} p={mw_p}")
    text = "\n".join(report)
    print(text)
    (OUTPUT_DIR / "stage1_evaluation_raw.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
