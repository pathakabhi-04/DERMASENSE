import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def main():
    with open(
        "data/rag/corpus_manifest.json",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    for document in manifest["documents"]:
        document_id = document["document_id"]
        url = document["source_url"]

        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                },
            )

            with urlopen(
                request,
                timeout=15,
            ) as response:
                print(
                    f"{document_id}: "
                    f"{response.status} "
                    f"{response.geturl()}"
                )

        except HTTPError as exc:
            print(
                f"{document_id}: "
                f"HTTP {exc.code}"
            )

        except URLError as exc:
            print(
                f"{document_id}: "
                f"URL ERROR {exc.reason}"
            )

        except Exception as exc:
            print(
                f"{document_id}: "
                f"ERROR {exc}"
            )


if __name__ == "__main__":
    main()