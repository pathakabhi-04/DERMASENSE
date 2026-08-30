from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen



USER_AGENT = (
    "DermaSense-RAG/0.1 "
    "(medical-research-hackathon; contact: project-team)"
)


class SourceAcquisitionError(RuntimeError):
    """Raised when a medical source cannot be acquired safely."""


def load_manifest(
    manifest_path: str | Path,
) -> dict:
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def download_source(
    document: dict,
    output_dir: str | Path,
    timeout: int = 30,
) -> dict:
    """
    Download one source and save a reproducible local snapshot.

    Returns acquisition metadata.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_id = document["document_id"]
    url = document["source_url"]

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.8",
        },
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:

            status = response.status
            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            raw_bytes = response.read()

    except Exception as exc:
        raise SourceAcquisitionError(
            f"Failed to acquire {document_id}: {exc}"
        ) from exc

    if status != 200:
        raise SourceAcquisitionError(
            f"{document_id}: HTTP status {status}"
        )

    if not raw_bytes:
        raise SourceAcquisitionError(
            f"{document_id}: empty response"
        )

    sha256 = hashlib.sha256(
        raw_bytes
    ).hexdigest()

    extension = ".html"

    if "text/plain" in content_type.lower():
        extension = ".txt"

    output_path = (
        output_dir
        / f"{document_id}{extension}"
    )

    output_path.write_bytes(
        raw_bytes
    )

    return {
        "document_id": document_id,
        "title": document["title"],
        "source": document["source"],
        "source_url": url,
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "http_status": status,
        "content_type": content_type,
        "sha256": sha256,
        "local_path": str(output_path),
        "byte_size": len(raw_bytes),
    }


def acquire_corpus(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> list[dict]:
    """
    Acquire every available document listed in the manifest.

    Documents explicitly marked as unavailable are skipped.
    Acquisition failures are recorded rather than aborting the
    entire corpus acquisition process.
    """

    manifest = load_manifest(
        manifest_path
    )

    documents = manifest["documents"]

    results = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        document_id = document["document_id"]

        print(
            f"[{index}/{len(documents)}] "
            f"{document_id}"
        )

        status = document.get(
            "status",
            "available",
        )

        if status != "available":
            print(
                f"    skipped: status={status}"
            )

            results.append(
                {
                    "document_id": document_id,
                    "title": document["title"],
                    "source": document["source"],
                    "source_url": document["source_url"],
                    "status": status,
                    "acquired": False,
                }
            )

            continue

        try:
            metadata = download_source(
                document,
                output_dir,
            )

            metadata["status"] = "acquired"
            metadata["acquired"] = True

            results.append(
                metadata
            )

            print(
                f"    saved: "
                f"{metadata['local_path']}"
            )

            print(
                f"    sha256: "
                f"{metadata['sha256']}"
            )

        except SourceAcquisitionError as exc:
            print(
                f"    FAILED: {exc}"
            )

            results.append(
                {
                    "document_id": document_id,
                    "title": document["title"],
                    "source": document["source"],
                    "source_url": document["source_url"],
                    "status": "acquisition_failed",
                    "acquired": False,
                    "error": str(exc),
                }
            )

    return results


def save_acquisition_manifest(
    records: list[dict],
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )