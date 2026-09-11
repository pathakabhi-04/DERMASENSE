# CHANGES — divergence from the RAG baseline we received

**Purpose:** every change made on this side to the RAG architecture the
collaborator delivered, so the two trees can be reconciled without
archaeology. **Append to this file with every further change.**

**Baseline received:** `rag_baseline.zip`, 2026-09-11, 155 files,
equivalent to `origin/rag-development` @ `2dcbc8d`.
**Working branch:** `rag-integration` (off `cv8-contract-v1.1`, off `main`).

**Section references** ("§4.3") are to
`RAG_EVOLUTION_AND_CV_INTEGRATION_STRATEGY_v2.md` unless stated otherwise.

---

## Legend

| Marker | Meaning |
|---|---|
| ADDED | new file/behaviour, nothing of theirs altered |
| EXTENDED | their code, additive and backward-compatible |
| CHANGED | their behaviour is different now — needs their sign-off |
| NOT CHANGED | considered and deliberately left alone |

---

## 1. Import scope (2026-09-11, `4e01036`)

**ADDED** — `src/rag/**` and `data/rag/**`, imported verbatim.

**NOT IMPORTED, deliberately.** The archive also carried `src/data/`,
`src/models/`, `src/training/`, `configs/cv_*.yaml` and `docs/CV_*.md` —
stale copies of this repo's own CV code from an older merge-base.
Importing them would have clobbered current CV work. If they ever need
CV code it should come from this repo, not round-trip through theirs.

**EXTENDED** — `requirements.txt`: union of both. Their `faiss-cpu`,
`sentence-transformers`, `beautifulsoup4`, `streamlit`, and their
`; sys_platform == "linux"` markers on `nvidia-*`/`triton`/`cuda-*` (no
Windows wheels exist for those; a no-op on Linux), plus this repo's
`matplotlib`, `opencv-python`, `umap-learn`, `numba`, `pytest`. No
version disagreed between the two files.

**EXTENDED** — `.gitignore`: added `data/rag/{raw,processed,indexes}/`,
matching their convention so the branches merge cleanly.

**Not a change, but recorded:** `rag_baseline.zip` included a live
`.env`. It was excluded from extraction and never read; the Groq key in
it should be rotated.

---

## 2. CV-8 contract v1.1 — changed on the CV side, not theirs (`3a49231`)

Listed because their parser depends on it. Detail in
`../cv8_contract_delta_v1.1.md`.

- `risk_reason` now quotes the **calibrated** confidence, not the raw
  softmax (they differed by 1.8–7.7 points).
- `risk_reason` no longer states `magnitude` numerically.
- `contract_version` added; fixtures regenerated.

---

## 3. `CVAssessmentContext` and parser (2026-09-11, `5d8f580`)

**ADDED** — `src/rag/cv_context/{schema,parser}.py`. No equivalent
existed; `memory/` and `patient_context/` were empty placeholders and
remain untouched.

Implements §19's context model and §16.1's fail-loud rule:

- Fails loudly on a missing/renamed top-level key, an unrecognised
  `risk_category` or `temporal.verdict`, a missing `contract_version`,
  or an unsupported MAJOR version.
- Preserves unknown `quality_flags` (§16) — never an error.
- All three `per_feature_deltas` independently nullable (§14.1); `null`
  is never coerced to `0.0`.
- `order_by_severity` implements §20's most-severe-first rule.

`render_cv_context` is the single place a CV assessment becomes text,
and it enforces the CV-side contract rules: calibrated confidence only;
never `magnitude` or per-feature deltas numerically; never
`compared_timestamps` as dates; null-vs-zero preserved in words.

---

## 4. Blocker A — grounding accepts CV context ⚠️ CHANGED

**CHANGED** — `check_source_presence` / `run_safety_check` accept
`cv_context` and treat it as a grounding source alongside retrieved
evidence.

**Why this was safe to change without waiting:** §4.3 already defines CV
context as supplied evidence "on exactly the same footing as retrieved
medical passages." Measuring grounding only against corpus text
contradicted the architecture the check exists to enforce.

**Measured.** Before: 0/5 real CV-grounded answers passed (corpus
overlap 0.0617–0.1005 against a 0.12 threshold; a CV narration scored
**0.0000** against the chunks actually retrieved for a matching query).
After: **4/5**. Off-topic text still scores 0.0000 against CV context and
a banned-phrase claim still fails regardless of grounding — the check
did not get weaker.

**Open, raised with them** (`../rag_safety_findings.md`): passing margins
are thin (0.1203–0.1489 against 0.12), because Jaccard divides by the
union and so penalises thorough answers for their length. Containment
(`|A∩B|/|B|`) would fit better. **Their metric, their call — not
changed.**

---

## 5. Blocker B — fallback carries the CV signal ⚠️ CHANGED

**CHANGED** — `build_fallback_answer(evidence, cv_context=None)`.

Previously it took only `EvidenceBundle`, so an LLM failure on a lesion
assessed HIGH risk and flagged for review produced corpus text that
never mentioned the assessment. §5.4 forbids exactly this: "that
evidence is safety-relevant and must reach them even if narration
fails." It is also the common path, not a rare one.

The assessment is now rendered first, most-severe-first across multiple
lesions, and never dropped.

---

## 6. Threading `cv_context` through the pipeline ⚠️ EXTENDED

All additive, `cv_context=None` by default, so every Phase 1 caller is
unchanged (verified: their 53 tests pass untouched).

| File | Change |
|---|---|
| `pipeline.py` | `answer(query, cv_context=None)`, threaded to prompt, safety check and fallback |
| `prompts/prompt_builder.py` | `build(query, evidence, cv_context=None)`; adds a CV block plus an instruction to keep patient-specific and general information distinct |
| `safety/grounding_check.py` | see §4 and §5 above |

**SYSTEM_PROMPT is NOT CHANGED.** Per §4.3 the generation approach does
not change when CV context arrives — and their system prompt already
said "the evidence or the structured CV context you are given." It was
written for this.

**Accepts a single context or a list.** CV-8 emits one assessment per
detected lesion, so multi-lesion is a real code path, not hypothetical
(§20, §26 Test 7).

**Refactor note:** `render_cv_context` lives in `cv_context/schema.py`,
not in the safety module, so the prompt builder need not import from the
safety layer (wrong dependency direction). All three consumers share one
renderer, so they agree exactly on what the CV evidence says — if they
diverged, an answer could be judged against different text than the user
is shown.

---

## 7. Phase 2 tooling (2026-09-11)

**EXTENDED** — `src/rag/cli.py`. Their Phase 1 modes are untouched;
added `--cv-fixture N`, `--cv-json PATH`, `--cv-all`. CV context is
resolved *before* the embedding model loads, so a bad fixture index or
malformed payload fails in ~4s instead of after a model load. The
interactive REPL keeps the CV context across turns.

**ADDED** — `src/rag/evaluate_cv_integration.py`, the Phase 2 gate
(§28 Step 5's analogue of their §6 Phase 1 gate). Eight binary criteria
per answer against the real pipeline and real Groq:

| # | Criterion |
|---|---|
| 1 | no fallback |
| 2 | cites a real retrieved source |
| 3 | no banned-phrase diagnostic claim |
| 4 | states the CALIBRATED confidence, never the raw softmax |
| 5 | never states `magnitude` or a per-feature delta numerically |
| 6 | never prints a `compared_timestamps` value |
| 7 | claims no comparison when verdict is `NO_PRIOR_DATA` |
| 8 | never narrates a null delta as "unchanged" |

Criteria 4–8 are the CV-specific ones a generic RAG eval would miss:
they check that a particular patient's numbers are narrated correctly,
not that the prose is fluent. Answers are written to
`evaluation/rag/cv_integration_answers.json` so a failure is
diagnosable without re-running five LLM calls.

**Current result: 4/5, GATE FAIL.** Criteria 2, 4, 5, 6, 7, 8 pass 5/5.
The single failing answer trips criterion 3 (banned phrase) and
therefore 1 (fallback) — the known, unfixed issue in §8 below, not a CV
contract problem.

**A gate bug found and fixed during bring-up, worth recording:**
criterion 7 initially matched the bare phrase "since the previous
photo", which is the *label* of a line in `render_cv_context`'s own
output — and the fallback embeds that render verbatim. The gate was
failing its own text. It now strips the supplied CV block and matches
only comparison *outcome* assertions. Re-checked against the same saved
answers: 7 and 8 went from 3/5 and 4/5 to 5/5 with no new LLM calls,
confirming those were harness bugs rather than model errors.

---

## 8. Tests ADDED

- `src/rag/cv_context/test_parser.py` — 21 tests including §26 Tests
  1–7, run against the **real** v1.1 fixtures rather than hand-written
  dicts, so a CV-side contract change breaks them loudly.
- `src/rag/safety/test_cv_grounding.py` — 16 blocker regressions,
  including negative controls proving the grounding check did not become
  permissive.

Suite: **90 pass**, up from their 53. None of their 53 were modified.

---

## 9. NOT CHANGED — considered and left alone

| Thing | Why |
|---|---|
| `SYSTEM_PROMPT` | §4.3; already written for CV context |
| Banned-phrase check | Fires on 1–2 of 5 good CV answers, and **39 of 156 corpus chunks (25%) would fail it themselves** — including "A dermatologist can tell you if you have basal cell carcinoma" and NCI lifetime-risk statistics. So a faithful paraphrase of the evidence inherits the phrasing and is rejected for it, and the fallback can contain what the check rejected. §5 deliberately chose over-flagging, so narrowing it is their call — raised in `../rag_safety_findings.md`, which now recommends the syntactic-proximity fix outright |
| Jaccard similarity metric | Thin margins argue for containment, but swapping the metric is a semantic decision — raised, not taken |
| `_SENTENCE_SPLIT_RE` | Hypothesised as the banned-phrase cause; **tested and disproved** — a newline-aware splitter flags the identical cases. Recorded so nobody retries it |
| `EvidenceFormatter`, retriever, index, chunking, embeddings | Correct as delivered; §7 rightly defers optimization |
| `top_score` → uncertainty | Defined and tested but never consumed, so their Phase 1 gate criterion 3 has no implementation. **Theirs, and Phase 1 — not ours to close** |
| Corpus, `retrieval_cases.json`, Phase 1 gate | Theirs |
| 6-class vs 8-class taxonomy | Jointly open (§18); still not guessed at |

---

## Change log

| Date | Commit | Change |
|---|---|---|
| 2026-09-11 | `4e01036` | Import RAG baseline; verify it runs here |
| 2026-09-11 | `5d8f580` | CV context schema + parser; fix Blockers A and B; thread `cv_context` through the pipeline |
| 2026-09-11 | `5fb3aa0` | Safety-layer findings note for the collaborator; this file |
| 2026-09-11 | (this commit) | Phase 2: CV modes in the CLI, CV-integration gate, corpus false-positive measurement |
