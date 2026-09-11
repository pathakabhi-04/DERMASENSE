"""
`/assess` endpoint tests.

The centrepiece is `test_endpoint_reproduces_the_delivered_fixtures`:
docs/build_on_baseline_1.md Section A's own acceptance criterion is that
the endpoint's response match the five delivered examples byte-for-byte.
That is what proves the API is a transport layer and not a second,
subtly different implementation of the contract.

Uses FastAPI's TestClient, so no server process and no network.
Checkpoint-backed tests are skipped when the weights or the dataset
volume are unavailable.
"""

from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "docs/cv8_sample_outputs/sample_outputs.json"
UQ_ZIP = REPO_ROOT / "data/raw/UQ_zip/866990d01449152d_NIMARE-A11453_A11453.zip"
PAD_TEST = REPO_ROOT / "data/splits/pad_ufes/test.csv"
CHECKPOINTS = [
    REPO_ROOT / "checkpoints/cv1_5_router/best.pt",
    REPO_ROOT / "checkpoints/cv3_512/best.pt",
    REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt",
]


def _png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def _checkpoints_available() -> bool:
    return all(p.exists() for p in CHECKPOINTS)


def _data_available() -> bool:
    try:
        return UQ_ZIP.exists() and PAD_TEST.exists()
    except OSError:
        # The dataset volume is an external disk; an I/O error means
        # "not attached", not "test failed".
        return False


class RequestValidationTests(unittest.TestCase):
    """No checkpoints needed: these never reach the pipeline."""

    def setUp(self) -> None:
        from fastapi.testclient import TestClient
        from src.serving import assess_api

        # Stand in for a loaded pipeline so validation runs without
        # 9 seconds of checkpoint loading.
        assess_api._state["pipeline"] = object()
        self.client = TestClient(assess_api.app)
        self.api = assess_api

    def tearDown(self) -> None:
        self.api._state["pipeline"] = None

    def test_undecodable_upload_is_a_client_error_not_a_server_error(self):
        response = self.client.post(
            "/assess", files={"image": ("note.txt", b"this is not an image", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("could not be decoded", response.json()["detail"])

    def test_empty_upload_is_rejected(self):
        response = self.client.post(
            "/assess", files={"image": ("empty.png", b"", "image/png")}
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_image_is_rejected(self):
        self.assertEqual(self.client.post("/assess").status_code, 422)

    def test_health_reports_contract_version(self):
        from src.risk.convergence import CONTRACT_VERSION

        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["contract_version"], CONTRACT_VERSION)

    def test_requests_before_the_pipeline_loads_get_503(self):
        self.api._state["pipeline"] = None
        response = self.client.post(
            "/assess", files={"image": ("x.png", _png(np.zeros((8, 8, 3), np.uint8)), "image/png")}
        )
        self.assertEqual(response.status_code, 503)


@unittest.skipUnless(_checkpoints_available(), "CV checkpoints not available")
@unittest.skipUnless(_data_available(), "dataset volume not attached")
class FixtureReproductionTests(unittest.TestCase):
    """
    Section A, item 3: the endpoint must reproduce the delivered
    examples. Loads checkpoints once for the whole class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient
        from src.serving import assess_api
        import pandas as pd

        cls.api = assess_api
        assess_api._state["pipeline"] = assess_api.load_pipeline()
        cls.client = TestClient(assess_api.app)
        cls.fixtures = json.loads(FIXTURES.read_text())
        cls.pad_row = pd.read_csv(PAD_TEST).iloc[5]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api._state["pipeline"] = None

    def test_first_visit_matches_the_delivered_payload(self):
        expected = self.fixtures[0]["payload"]
        image = cv2.imread(str(REPO_ROOT / self.pad_row["image_path"]))
        self.assertIsNotNone(image, "PAD-UFES image unreadable")

        response = self.client.post(
            "/assess",
            files={"image": ("current.png", _png(image), "image/png")},
            data={"lesion_id": "first-visit-example"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["assessed"])
        self.assertEqual(body["num_assessments"], 1)
        self.assertEqual(body["assessments"][0], expected)
        # The token is envelope-level metadata, never inside the contract.
        self.assertNotIn("measurement", body["assessments"][0])
        self.assertIsNotNone(body["measurement"])

    def test_returning_visit_with_prior_image_matches(self):
        expected = self.fixtures[1]["payload"]
        source = self.fixtures[1]["source"]

        with zipfile.ZipFile(UQ_ZIP) as zf:
            earlier = cv2.imdecode(
                np.frombuffer(zf.read(source["earlier"]), np.uint8), cv2.IMREAD_COLOR)
            later = cv2.imdecode(
                np.frombuffer(zf.read(source["later"]), np.uint8), cv2.IMREAD_COLOR)

        response = self.client.post(
            "/assess",
            files={
                "image": ("later.png", _png(later), "image/png"),
                "prior_image": ("earlier.png", _png(earlier), "image/png"),
            },
            data={
                "lesion_id": expected["lesion_id"],
                "prior_timestamp": source["earlier"].rsplit("/", 1)[-1],
                "current_timestamp": source["later"].rsplit("/", 1)[-1],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assessments"][0], expected)

    def test_cached_measurement_reproduces_the_same_assessment(self):
        """
        Reusing the prior visit's measurement must be bit-identical to
        re-measuring its image. If it ever diverges, the optimisation has
        silently become an approximation of a temporal verdict.
        """
        expected = self.fixtures[1]["payload"]
        source = self.fixtures[1]["source"]

        with zipfile.ZipFile(UQ_ZIP) as zf:
            earlier = cv2.imdecode(
                np.frombuffer(zf.read(source["earlier"]), np.uint8), cv2.IMREAD_COLOR)
            later = cv2.imdecode(
                np.frombuffer(zf.read(source["later"]), np.uint8), cv2.IMREAD_COLOR)

        # Visit 1: measure the earlier image, keep the token.
        first = self.client.post(
            "/assess", files={"image": ("earlier.png", _png(earlier), "image/png")})
        token = first.json()["measurement"]
        self.assertIsNotNone(token, "first visit must hand back a token")

        # Visit 2: send the token instead of the prior image.
        second = self.client.post(
            "/assess",
            files={"image": ("later.png", _png(later), "image/png")},
            data={
                "lesion_id": expected["lesion_id"],
                "prior_timestamp": source["earlier"].rsplit("/", 1)[-1],
                "current_timestamp": source["later"].rsplit("/", 1)[-1],
                "prior_measurement": json.dumps(token),
            },
        )
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["assessments"][0], expected)

    def test_malformed_prior_measurement_is_rejected_loudly(self):
        image = cv2.imread(str(REPO_ROOT / self.pad_row["image_path"]))
        for bad in ('{"measurement": {"valid": true}}', "not json", '["a"]'):
            with self.subTest(bad=bad):
                response = self.client.post(
                    "/assess",
                    files={"image": ("c.png", _png(image), "image/png")},
                    data={"prior_measurement": bad},
                )
                self.assertEqual(response.status_code, 400)

    def test_response_parses_with_the_rag_side_parser(self):
        """The endpoint's output must satisfy the consumer that exists."""
        from src.rag.cv_context.parser import parse_cv_assessment

        image = cv2.imread(str(REPO_ROOT / self.pad_row["image_path"]))
        response = self.client.post(
            "/assess",
            files={"image": ("current.png", _png(image), "image/png")},
            data={"lesion_id": "parser-check"},
        )
        context = parse_cv_assessment(response.json()["assessments"][0])
        self.assertEqual(context.lesion_id, "parser-check")
        self.assertEqual(context.contract_version, "1.1")

    def test_blank_image_is_not_assessed_and_says_why(self):
        """
        'We never looked at this' must not read as 'we looked and it was
        fine': a non-ASSESSED outcome returns 200 with assessed=False,
        an empty list, and the reason named.
        """
        blank = np.zeros((512, 512, 3), np.uint8)
        response = self.client.post(
            "/assess", files={"image": ("blank.png", _png(blank), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["assessed"])
        self.assertEqual(body["assessments"], [])
        self.assertIn(body["outcome"], {"QUALITY_REJECTED", "NO_CANDIDATES"})


if __name__ == "__main__":
    unittest.main()
