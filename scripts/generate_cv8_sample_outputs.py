"""
Generate real sample CV-8 JSON outputs for the RAG collaborator.

This is a delivery artifact, not an evaluation: five real pipeline runs
(real checkpoints, real images -- 4 from the staged UQ Longitudinal
sample, 1 from PAD-UFES), picked to show the RANGE of shapes the locked
contract (docs/cv7_temporal_rag_integration_spec.md) can actually
produce, not to tell a particular story. None are hand-edited after
the run -- what's here is exactly what `RiskAssessment.to_dict()`
returned.

Run: python -m scripts.generate_cv8_sample_outputs
Output: docs/cv8_sample_outputs/sample_outputs.json
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.orchestrator import DermaSensePipeline, PipelineOutcome

REPO_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = REPO_ROOT / "data/raw/UQ_zip/866990d01449152d_NIMARE-A11453_A11453.zip"
ARCHIVE_PREFIX = "866990d01449152d_NIMARE-A11453_A11453/data/Dermoscopic Images/"
OUTPUT_DIR = REPO_ROOT / "docs/cv8_sample_outputs"

ROUTER_CHECKPOINT = REPO_ROOT / "checkpoints/cv1_5_router/best.pt"
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)
PAD_UFES_TEST = REPO_ROOT / "data/splits/pad_ufes/test.csv"

# (lesion_id, description, earlier_visit_path, later_visit_path) -- all
# real, hand-identified pairs from the staged sample, not randomly
# re-drawn each run, so this script is reproducible.
UQ_EXAMPLES = [
    (
        "General18_Lesion7",
        "returning visit, stable temporal verdict",
        ARCHIVE_PREFIX + "General/General18_Lesion7_visit1.jpg",
        ARCHIVE_PREFIX + "General/General18_Lesion7_visit2.jpg",
    ),
    (
        "HighRisk37_Lesion6",
        "returning visit, CV-7 CHANGED_COLOR escalates risk_category "
        "one step (MONITOR base -> MEDIUM) and forces requires_review",
        ARCHIVE_PREFIX + "HighRisk/HighRisk37_Lesion6_visit1.jpg",
        ARCHIVE_PREFIX + "HighRisk/HighRisk37_Lesion6_visit2.jpg",
    ),
    (
        "General151_Lesion3",
        "prior image WAS supplied but CV-3 found no lesion mask in the "
        "current image -- NO_PRIOR_DATA despite a real comparison "
        "attempt, distinct from never having a prior image at all",
        ARCHIVE_PREFIX + "General/General151_Lesion3_visit1.jpg",
        ARCHIVE_PREFIX + "General/General151_Lesion3_visit2.jpg",
    ),
    (
        "HighRisk274_Lesion16",
        "returning visit with a disclosed quality flag (low crop blur) "
        "alongside a normal STABLE verdict -- flags never change the risk",
        ARCHIVE_PREFIX + "HighRisk/HighRisk274_Lesion16_visit2.jpg",
        ARCHIVE_PREFIX + "HighRisk/HighRisk274_Lesion16_visit3.jpg",
    ),
]


def _read_zip_image(zf: zipfile.ZipFile, path: str) -> np.ndarray:
    with zf.open(path) as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        detector_weights=None,
        device="cpu",
    )

    examples = []

    # First-visit example: no prior image at all (PAD-UFES, matching
    # the checkpoint's own training domain).
    row = pd.read_csv(PAD_UFES_TEST).iloc[5]
    image_bgr = cv2.imread(str(REPO_ROOT / row["image_path"]))
    result = pipeline.predict(image_bgr, lesion_id="first-visit-example")
    assert result.outcome is PipelineOutcome.ASSESSED
    examples.append(
        {
            "description": "first visit -- no prior image supplied at all",
            "source": str(row["image_path"]),
            "payload": result.candidates[0].risk_assessment.to_dict(),
        }
    )

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for lesion_id, description, earlier_path, later_path in UQ_EXAMPLES:
            earlier_img = _read_zip_image(zf, earlier_path)
            later_img = _read_zip_image(zf, later_path)
            result = pipeline.predict(
                later_img,
                lesion_id=lesion_id,
                prior_image_bgr=earlier_img,
                prior_timestamp=earlier_path.rsplit("/", 1)[-1],
                current_timestamp=later_path.rsplit("/", 1)[-1],
            )
            assert result.outcome is PipelineOutcome.ASSESSED, (lesion_id, result.outcome)
            examples.append(
                {
                    "description": description,
                    "source": {"earlier": earlier_path, "later": later_path},
                    "payload": result.candidates[0].risk_assessment.to_dict(),
                }
            )

    output_path = OUTPUT_DIR / "sample_outputs.json"
    output_path.write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
