# DermaSense RAG Development State

**Status:** Active checkpoint

**Updated:** 2026-09-06

**Primary specification:** `docs/rag/rag_evolution_and_cv_integration_strategy_v2.md`

**Purpose:** This is the persistent engineering checkpoint for the RAG pipeline. Update it after every material implementation, experiment, validation run, decision, or blocker. It records what is true in the repository, rather than replacing the architecture and safety decisions in the primary specification.

---

## 1. Governing architecture and current phase

The primary design rule remains unchanged:

> **The CV system computes the clinical signals. RAG supplies medical knowledge. The LLM explains both without inventing or overriding either.**

The intended separation of responsibilities is:

| Component | Responsibility |
|---|---|
| CV-3 / CV-4 / CV-6 / CV-7 / CV-8 | Compute lesion, classification, uncertainty, temporal, and risk signals |
| RAG Retriever | Retrieve attributable medical evidence |
| RAG context layer | Normalize and retain CV, retrieval, and future patient context separately |
| Prompt builder | Assemble constrained, grounded inputs |
| LLM | Constrained paraphrase/explanation only |
| Safety layer | Reject or fall back from unsupported or unsafe output |

### Current phase position

The project is still in **Phase 1 — Baseline RAG**. The current work completed the per-class corpus coverage prerequisite identified in Section 1.1 of the primary specification. It has **not** completed the Phase 1 end-to-end answer-generation gate.

The Phase 1 completion gate remains:

- retrieval + evidence formatting + hosted LLM + safety/grounding check are implemented;
- 100% of the fixed evaluation set passes the defined criteria;
- every answer cites a real retrieved source;
- no answer contains a banned diagnostic claim; and
- low-evidence retrieval results explicitly state uncertainty.

No CV fixture parser, CV-context prompt integration, patient memory, reranking, or learned groundedness checker has been implemented as part of this checkpoint.

---

## 2. Coverage experiment: original finding

The latest experiment was `src/rag/retrieval/check_class_coverage.py`. Its purpose is to answer the specification's immediate safety question:

> Does the corpus contain class-specific evidence for every native CV label that may be emitted?

The current CV-native taxonomy is:

```text
ACK   BCC   MEL   NEV   SCC   SEK
```

### Result before corpus expansion

The persisted `medical_v0.1` index initially contained 45 chunks across eight acquired documents. The original term scan found:

| Native class | Initial result |
|---|---|
| ACK — actinic keratosis | Not covered |
| BCC — basal cell carcinoma | Not covered |
| MEL — melanoma | Covered |
| NEV — nevus / nevi | Covered |
| SCC — squamous cell carcinoma | Not covered |
| SEK — seborrheic keratosis | Not covered |

This meant four of six possible CV classifications had no dedicated medical source for grounded explanation. Continuing directly to answer generation would have created a predictable failure: the retriever could return the nearest unrelated lesion evidence and the LLM could narrate it convincingly despite its irrelevance.

---

## 3. Coverage-tool defect and correction

### Defect found

The original checker grouped chunks by `metadata.corpus_id`. All sources share the corpus ID `dermasense-medical`, so the script reported one misleading "document" even though the index actually held eight source documents.

The word-match result was valid for whole-corpus coverage, but the script did not meet the specification's document-level requirement: list the individual source documents supporting each class.

### Change made

`src/rag/retrieval/check_class_coverage.py` now:

- groups chunks by `document_id`;
- preserves the source title for each document;
- exposes `load_documents()` and `find_class_coverage()` for testability; and
- prints matching document IDs and titles for every covered class.

### Regression coverage

`src/rag/retrieval/test_class_coverage.py` was added with two standard-library unit tests:

1. Verifies two documents sharing one `corpus_id` remain separate in the report.
2. Verifies all six native classes are detected when their terms occur in a source document.

Latest result:

```text
Ran 2 tests
OK
```

---

## 4. Corpus expansion completed

Four class-specific sources were added to `data/rag/corpus_manifest.json`. All are from the American Academy of Dermatology (AAD), which is an approved primary source family under `src/rag/MEDICAL_CORPUS_v0.1.md`.

| CV class | New document ID | Source title | Topic |
|---|---|---|---|
| ACK | `AAD_ACTINIC_KERATOSIS_SYMPTOMS_001` | *Actinic keratosis: Signs and symptoms* | `skin_lesions` |
| BCC | `AAD_BASAL_CELL_CARCINOMA_001` | *Basal cell carcinoma: From symptoms to treatments* | `skin_cancer` |
| SCC | `AAD_SQUAMOUS_CELL_CARCINOMA_001` | *Squamous cell carcinoma: From symptoms to treatments* | `skin_cancer` |
| SEK | `AAD_SEBORRHEIC_KERATOSES_SYMPTOMS_001` | *Seborrheic keratoses: Signs and symptoms* | `skin_lesions` |

The sources were downloaded through the existing reproducible acquisition flow:

```powershell
.\.venv\Scripts\python.exe -m src.rag.ingestion.acquire_corpus
```

The resulting snapshot locations, timestamps, HTTP statuses, byte counts, and SHA-256 values are recorded in `data/rag/acquisition_manifest.json`.

### Current acquired-corpus state

| Measure | Value |
|---|---:|
| Manifest entries | 14 |
| Explicitly unavailable entries | 2 |
| Successfully acquired documents | 12 |
| Chunks produced from acquired documents | 156 |

The corpus-loader expectation was updated from eight to twelve documents in `src/rag/ingestion/test_corpus_loader.py` and passed against the acquired snapshots.

### Class coverage after source acquisition and chunking

The persisted FAISS metadata has not yet been rebuilt, so the corrected checker cannot truthfully claim the old 45-chunk index is current. Coverage was therefore verified against the freshly loaded and chunked acquisition output:

```text
ACK: covered
BCC: covered
MEL: covered
NEV: covered
SCC: covered
SEK: covered
```

Each class has at least one direct, class-specific AAD or pre-existing melanoma/nevus source. Some sources mention additional class names incidentally; coverage should be interpreted as an existence gate, not as a ranking-quality result.

---

## 5. Retrieval-evaluation changes

`src/rag/retrieval/retrieval_cases.json` was expanded from 12 to 16 cases.

The four new cases are:

| Query | Expected document |
|---|---|
| What are common signs of actinic keratosis? | `AAD_ACTINIC_KERATOSIS_SYMPTOMS_001` |
| What can basal cell carcinoma look like? | `AAD_BASAL_CELL_CARCINOMA_001` |
| What are possible signs of squamous cell carcinoma? | `AAD_SQUAMOUS_CELL_CARCINOMA_001` |
| What are the signs of seborrheic keratosis? | `AAD_SEBORRHEIC_KERATOSES_SYMPTOMS_001` |

This evaluates the full current CV-native taxonomy rather than relying on aggregate retrieval accuracy that was mostly driven by melanoma, wounds, and burns.

### Important scope note

The wounds and burns documents remain in the corpus because the approved medical-corpus scope includes first-aid guidance. They should not be deleted merely because they do not support lesion classification. If future retrieval results show that they crowd out lesion evidence for lesion/CV queries, address that observed failure with metadata-aware filtering or routing in the later retrieval-optimization phase; do not preemptively redesign retrieval.

---

## 6. Environment and reproducibility status — RESOLVED (2026-09-06)

### Original problem

The root `.venv` pointed at a Python interpreter path that appeared removed, and `python`/`py -0p` did not resolve one in the shell at the time. A temporary `.venv` was created on an embedded Python 3.13 runtime, which was sufficient for corpus acquisition (`beautifulsoup4`) but not for embedding/index work (`faiss`, `sentence_transformers` were undeclared, and the full `requirements.txt` install failed regardless of Python version).

### Actual root cause (corrected)

Two independent issues, not one:

1. **Python 3.12 was in fact available**, at `C:\Users\ishub\AppData\Local\Programs\Python\Python312\python.exe` (`py -0p` resolves it; it runs and reports `Python 3.12.5`). The "removed interpreter" was a stale shell-level `.venv` pointer, not a missing system interpreter.
2. **The real install blocker was platform, not Python version.** `requirements.txt` is a `pip freeze` of a **Linux** CUDA training environment. The `nvidia-*` packages (`nvidia-cublas`, `nvidia-cufile`, etc.) and `triton==3.7.1` have **no Windows distribution at all** — confirmed directly (`pip download` returns "No matching distribution found" for these on Windows regardless of Python version). `nvidia-cufile` in particular backs GPUDirect Storage, a Linux-only CUDA feature. Torch itself has a normal `win_amd64` wheel and does not need these packages on Windows — Windows torch wheels bundle their own CUDA runtime.

### Fix applied

- Recreated `.venv` from the real Python 3.12.5 interpreter.
- Added `; sys_platform == "linux"` environment markers to all 15 `nvidia-*` lines and to `triton` in `requirements.txt`, so pip skips them on Windows while leaving Linux installs (e.g. the CV training environment) unaffected.
- Added `faiss-cpu==1.9.0` and `sentence-transformers==3.3.1` to `requirements.txt` — both are imported directly by `src/rag/embeddings/embedder.py` and `src/rag/vectorstore/faiss_store.py` but were never declared.
- `pip install -r requirements.txt` now completes successfully (exit code 0) on Python 3.12.5 / Windows.

### Verified working versions

| Package | Version |
|---|---|
| Python | 3.12.5 |
| faiss | 1.9.0 (`faiss-cpu`) |
| sentence-transformers | 3.3.1 |
| torch | 2.13.0+cpu |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |

Confirmed via direct import (`faiss`, `sentence_transformers`, `torch`, `sklearn`, `pandas`, `bs4` all import cleanly) and by re-running the existing test suite on the new environment:

- `python -m unittest src.rag.retrieval.test_class_coverage` — 2/2 passed.
- `python -m src.rag.ingestion.test_corpus_loader` — passed (12 documents).

### Environment is no longer a blocker

The `.venv` at the repo root, built from Python 3.12.5 with the corrected `requirements.txt`, is now the supported, reproducible project environment. `build_index.py` and `evaluate_retrieval.py` are unblocked and ready to run against the expanded 12-document/156-chunk corpus (§7, §8).

---

## 7. Persisted-index status — REBUILT (2026-09-06)

The index at `data/rag/indexes/medical_v0.1` has been rebuilt against the expanded corpus:

| Item | Old persisted index | Rebuilt index |
|---|---:|---:|
| Documents | 8 | 12 |
| Chunks | 45 | 156 |
| Embedding dimension | — | 384 |
| ACK/BCC/SCC/SEK evidence | Absent | Present |

Rebuild command used:

```powershell
.\.venv\Scripts\python.exe -m src.rag.vectorstore.build_index
```

### Bug found and fixed during rebuild

`src/rag/embeddings/embedder.py`'s `dimension` property called `self.model.get_embedding_dimension()`, which does not exist on `sentence_transformers.SentenceTransformer` (the real method is `get_sentence_embedding_dimension()`). This was a latent defect — the embedding/index-build code path had never been executed end-to-end before this checkpoint, so the error was never surfaced. Fixed by correcting the method name. No other code path depends on the old (incorrect) name.

### Post-rebuild validation

Class coverage against the rebuilt index (`check_class_coverage.py`): all six classes covered, matching §4's pre-rebuild projection exactly — no surprises.

16-case retrieval evaluation (`evaluate_retrieval.py`):

| Metric | Result |
|---|---:|
| Top-1 document accuracy | 15/16 (93.8%) |
| Top-3 document accuracy | 16/16 (100.0%) |
| Top-3 topic accuracy | 16/16 (100.0%) |

The single Top-1 miss is case 11 ("How should I clean an abrasion?"), which ranks `MEDLINEPLUS_MINOR_BURNS_001` above the correct `MEDLINEPLUS_SCRAPE_001` at rank 1 (score 0.3415 vs. eventual rank-3 score 0.3297 — a low-confidence, closely-scored case), but recovers the correct document by rank 3. This is not one of the four new class-specific cases — all four (ACK/BCC/SCC/SEK) pass at Top-1 with clear score margins.

**This result confirms, rather than requires re-deciding, the evidence-selection policy already specified in the primary specification (§3.1): feed the top-3 chunks deduplicated by source document, not Top-1 alone.** Case 11 is a concrete instance of exactly the failure mode that policy exists to prevent.

Do not manually overwrite `chunks.json` without rebuilding `medical.faiss`; the vector store enforces an equal count between vector rows and serialized chunks, and a mismatch would make retrieval unsafe/unusable. (Not an issue in this rebuild — both were regenerated together by `build_index.py`.)

**Re-verification (2026-09-06, same checkpoint):** `check_class_coverage.py` was re-run independently against the persisted index a second time to confirm the result was not a fluke of the immediate post-rebuild state. Identical outcome: all six classes (ACK, BCC, MEL, NEV, SCC, SEK) covered, each backed by at least one dedicated document. No drift between runs. This closes the original ACK/BCC/SCC/SEK non-coverage gap (§2) as verified against the live, queryable index — not just against freshly-chunked text.

---

## 8. Next actions, in order

Steps 1-6 below are complete as of this checkpoint (2026-09-06). See §6 and §7 for details.

1. ~~Restore a supported Python 3.12 interpreter and recreate `.venv` from it.~~ Done.
2. ~~Install the full declared dependency set successfully.~~ Done — see §6 for the platform-marker fix and verified versions.
3. ~~Rebuild the FAISS index.~~ Done — 156 chunks, 384-dim, 12 documents.
4. ~~Run the corrected coverage checker against the rebuilt index.~~ Done — all six classes covered.
5. ~~Run the 16-case retrieval evaluation.~~ Done — Top-1 93.8% (15/16), Top-3 document 100%, Top-3 topic 100%. See §7 for the single Top-1 miss (case 11) and why it does not block progress.
6. ~~Inspect failures.~~ Done — the one miss is a wounds/burns query, not one of the four new class-specific cases, and it recovers by Top-3. No newly added class-specific case failed. The expanded retrieval gate is understood; no corpus or retrieval rework is warranted before proceeding.

**Next concrete gate (not yet started):** build Phase 1's remaining pipeline components per the primary specification —

7. Evidence formatter (`src/rag/retrieval/evidence.py`, per spec §3.1): top-3 chunks, deduplicated by source document.
8. Prompt builder (per spec §3.2/§4.1): constrained-paraphrase system prompt, uncertainty/adversarial-decline instructions.
9. Hosted LLM adapter (per spec §4.2).
10. Deterministic safety/grounding check (per spec §5): banned-phrase scan, source-presence check, generation-timeout fallback.
11. Run the Phase 1 answer-evaluation gate (spec §6): 100% pass on the fixed test set across all three criteria (cites a real source, no banned diagnostic phrase, states uncertainty on low-similarity retrieval — define the similarity threshold from the actual score distribution observed in §7 above, e.g. case 11's ~0.33 range, rather than inventing one).
12. Update this file with commands run, results, failures, decisions, and the next concrete gate.

---

## 9. Commands and validation record

Completed successfully during this checkpoint:

```powershell
# Coverage-tool unit tests
.\.venv\Scripts\python.exe -m unittest src.rag.retrieval.test_class_coverage

# Corpus acquisition
.\.venv\Scripts\python.exe -m src.rag.ingestion.acquire_corpus

# Corpus-loader validation
.\.venv\Scripts\python.exe -m src.rag.ingestion.test_corpus_loader

# Fresh acquisition/chunk coverage validation
# (performed through MedicalCorpusChunker plus the checker's reusable helpers)
```

Validated outcomes:

- class-coverage tests: passed (2/2);
- corpus loader: passed (12 documents);
- fresh chunk coverage: all six native CV classes covered;
- retrieval-case JSON: valid, 16 cases;
- persisted FAISS rebuild and retrieval evaluation: blocked pending supported Python 3.12 environment.

---

## 10. Change log

| Date | Change | Result |
|---|---|---|
| 2026-09-06 | Analysed `rag_evolution_and_cv_integration_strategy_v2.md` against the coverage experiment. | Identified missing ACK, BCC, SCC, and SEK evidence as the immediate Phase 1 blocker. |
| 2026-09-06 | Corrected document grouping in the class-coverage checker and added regression tests. | Source-level reporting is now accurate; 2 tests pass. |
| 2026-09-06 | Added four authoritative AAD class-specific sources and acquired snapshots. | Fresh corpus expanded from 8 to 12 usable documents. |
| 2026-09-06 | Expanded retrieval evaluation from 12 to 16 cases. | Each native CV class now has at least one targeted retrieval query. |
| 2026-09-06 | Recreated stale root `.venv` using available temporary Python 3.13; declared/installled Beautiful Soup. | Acquisition and loading work; full RAG dependency installation and index rebuild remain blocked by Python-version incompatibility. |
| 2026-09-06 | Diagnosed the real environment blocker: Python 3.12.5 was actually available; the true install failure was Linux-only `nvidia-*`/`triton` packages with no Windows distribution. Recreated `.venv` on Python 3.12.5, added platform markers, added missing `faiss-cpu`/`sentence-transformers` declarations. | `pip install -r requirements.txt` succeeds; all RAG dependencies import cleanly; existing tests pass on the new environment. |
| 2026-09-06 | Rebuilt the FAISS index against the 12-document/156-chunk corpus. | Fixed a latent bug in `embedder.py` (`get_embedding_dimension` → `get_sentence_embedding_dimension`, a method that never existed in `sentence_transformers` and had never been exercised before). Index now holds 156 chunks, 384-dim embeddings. |
| 2026-09-06 | Re-ran class coverage and the 16-case retrieval evaluation against the rebuilt index. | All six classes covered; Top-1 93.8% (15/16), Top-3 document 100%, Top-3 topic 100%. Single Top-1 miss (case 11) is a wounds/burns query, not a class-specific case, and recovers by Top-3 — confirms rather than changes the spec's existing top-3-dedup evidence policy. |

