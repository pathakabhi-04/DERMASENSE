# CHANGES — divergence from the RAG baseline we received

**Purpose:** every change made on this side to the RAG architecture the
collaborator delivered, so the two trees can be reconciled without
archaeology. **Append to this file with every further change.**

**Baseline received:** `rag_baseline.zip`, 2026-09-11, 155 files,
verified content-identical to `origin/rag-development` @ `2dcbc8d`
(`git diff --ignore-cr-at-eol` was empty; the zip differed only in line
endings).
**Working branch:** `integration`.
**Integration point:** `84d8b5a`, tagged `integration-v1` — a two-parent
merge of the CV baseline (`d55cff7`, carrying contract v1.1) and the RAG
baseline (`2dcbc8d`). Everything before it belongs to one baseline;
everything after is integration work.

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

## 1. How the RAG side arrived (`84d8b5a`)

**MERGED** — the RAG baseline enters as a genuine two-parent merge of
`origin/rag-development`, not as a copy of the delivered zip.

An earlier attempt did import the zip, adding 64 files as though they
were written here. That was wrong about provenance (`git blame` on
`src/rag` credited the wrong author) and would have made every future
merge with `rag-development` an add/add conflict, for want of a common
ancestor. It also buried the real changes: reviewing the zip-import
branch against `rag-development` showed 44 files and 4967 insertions,
almost all line-ending noise. Against the merge it is **11 files and
1874 insertions** — exactly the work described below.

Because untouched files keep her exact bytes, `git diff
origin/rag-development..integration -- src/rag/` is a clean review diff.

**NOT taken from the zip, deliberately.** The archive also carried
`src/data/`, `src/models/`, `src/training/`, `configs/cv_*.yaml` and
`docs/CV_*.md` — stale copies of this repo's own CV code from an older
merge-base. The merge does not bring them either, since `rag-development`
predates CV-8 entirely (it has no `src/risk/convergence.py` and no
`docs/cv8_sample_outputs/`). That is also why the integration could not
live on `rag-development`: the parser tests read the v1.1 fixtures.

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

## 3. `CVAssessmentContext` and parser (2026-09-11, `bc7d70b`)

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

**RESOLVED (2026-09-11) — metric changed for the CV source only.**
Jaccard margins were thin because it divides by the union, penalising a
thorough answer for its own length; real CV answers scored 0.1049–0.1776
against a 0.12 bar, the worst of them *below* it despite being correct.

The CV source is now scored by **containment** (`|A∩B|/|B|`, "how much
of the supplied source did the answer use?"), which is length-insensitive.
**Corpus chunks keep Jaccard at 0.12, untouched** — that threshold was
calibrated against corpus text (§13) and Phase 1 behaviour must not shift.

Threshold calibrated, not chosen: positives 0.3659–1.0000; 780 negative
pairs (all 156 corpus chunks × 5 CV contexts) peak at 0.1622.
`CV_SOURCE_PRESENCE_THRESHOLD = 0.25` is the geometric midpoint — 1.5×
above the worst negative, 1.46× below the worst positive.

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
| ~~Banned-phrase check~~ | **Now CHANGED — see §10 below** |
| Jaccard similarity metric | Thin margins argue for containment, but swapping the metric is a semantic decision — raised, not taken |
| `_SENTENCE_SPLIT_RE` | Hypothesised as the banned-phrase cause; **tested and disproved** — a newline-aware splitter flags the identical cases. Recorded so nobody retries it |
| `EvidenceFormatter`, retriever, index, chunking, embeddings | Correct as delivered; §7 rightly defers optimization |
| `top_score` → uncertainty | Defined and tested but never consumed, so their Phase 1 gate criterion 3 has no implementation. **Theirs, and Phase 1 — not ours to close** |
| Corpus, `retrieval_cases.json`, Phase 1 gate | Theirs |
| 6-class vs 8-class taxonomy | Jointly open (§18); still not guessed at |

---

## 10. Banned-phrase check narrowed ⚠️ CHANGED (2026-09-11)

**CHANGED** — `check_banned_phrases`. Bare co-occurrence of a certainty
phrase and a condition name in a sentence now additionally requires:

1. the condition name within `MAX_CERTAINTY_CONDITION_GAP` (3) tokens
   after the certainty phrase, and
2. no conditional/hedge governing **that clause**.

**Why it was safe to change:** §5 chose over-flagging and said to narrow
"later only if real failures justify it". Those failures arrived —
**39 of 156 corpus chunks (25%) failed the check themselves**, including
"A dermatologist can tell you if you have basal cell carcinoma" and NCI
lifetime-risk statistics. A faithful constrained paraphrase inherited
that phrasing and was rejected for being faithful.

**The bar we held:** 100% recall on a true-positive set containing cases
designed to defeat the new rule — not merely "the FP rate improved".

**A near-miss worth recording.** The first version scoped the hedge to
the whole **sentence**. It measured beautifully (100% recall on 15 TPs,
FPs 44→2) and was **dangerously wrong**: any hedge earlier in the
sentence suppressed the flag, so "If you were wondering, you have
melanoma" passed. It missed **8 of 8** adversarial claims — strictly
worse than the original rule. Scoping the hedge to its own clause fixed
it. Those 8 cases now live in `test_banned_phrase_precision.py` so the
scope cannot be re-widened without a red test.

| Rule | Recall (23 TPs) | Corpus FP rate |
|---|---|---|
| Original co-occurrence | 100% | 39/156 (25%) |
| Sentence-scoped hedge | **65% — unsafe, rejected** | 2/44 |
| **Clause-scoped (shipped)** | **100%** | **4/156 (3%)** |

The 4 survivors are third-person epidemiology and fail safe. We stopped
there: each further exclusion trades recall for precision on a safety
rule.

---

## 10b. Citation format pinned ⚠️ EXTENDED (2026-09-11)

**EXTENDED** — `PromptBuilder.build` now tells the model how to cite.

Found in a live UI answer, not by testing: the model emitted
`【2†L9-L13】` and `【3†L1-L8】` — browsing-style markers carrying line
ranges. `EvidenceBundle.format_for_prompt` emits `[1] Source: <title>`
and nothing finer, so those line numbers refer to nothing. The content
was correctly grounded; the *citation* implied a traceability that does
not exist, which in a medical answer is its own kind of wrong.

Root cause: the prompt never said how to cite at all, so the model fell
back on its training convention. Both prompt branches now say to use the
supplied bracketed numbers and not to invent finer references.

Gate **criterion 9** detects the fabricated form. It appeared in 0 of 5
scripted answers but did appear in real UI use, so it is intermittent —
the check exists because absence from one sample is not evidence.

---

## 11. Phase 2 gate result

With §4, §5, §10 and §10b in place the CV-integration gate passes
**5/5 on all nine criteria**, up from 2/5 on eight. All 53 of the collaborator's original
tests still pass, unmodified. Suite total: **99**.

---

## 12. CV-side: live-feed latency measured (2026-09-11)

Not a RAG change, recorded because it unblocks the delivery mechanism.
`scripts/measure_cv8_latency.py` measures `DermaSensePipeline.predict()`
against a pre-committed rule (p50 < 2.0s → build the sync endpoint).

**Corrected 2026-09-11 — the first measurement was incomplete.** It
timed only first visits (one image, CV-7 idle) and reported p50 0.52s,
concluding sync was viable. Measuring the returning-visit path — prior
image supplied, so two segmentations plus CV-7, which is the shape CV-7
exists for — gives **p50 2.46s, 4.9× slower and over the 2.0s bar**.

| Request shape | p50 (CPU) | vs the 2.0s rule |
|---|---|---|
| First visit | 0.50s | passes |
| **Returning visit** | **2.46s** | **fails** |

End-to-end over HTTP, a returning visit plus its RAG answer measured
**~4.6s** wall clock (CV 2.5s + RAG 2.1s).

**So sync is NOT settled.** Per the rule fixed before measuring, the
sync-vs-async question goes back to the RAG side with this number. The
endpoint below is built and works, but that decision is theirs and is
open. The script now measures both shapes, because measuring only the
cheap path is how a latency budget gets set on the wrong number.

---

## 13. `POST /assess` — the live feed (2026-09-11)

**ADDED** — `src/serving/assess_api.py`, a FastAPI transport layer over
`DermaSensePipeline.predict()`. No new CV logic; every field comes from
`RiskAssessment.to_dict()` unchanged.

Built to `docs/build_on_baseline_1.md` Section A, including its
"explicitly not now" list, which is honoured in full: no auth, rate
limiting, scaling, retry/queue, streaming, TLS, or observability. Where
it runs remains an infra decision and is still open.

- Checkpoints load once at startup (~9s), never per request.
- `POST /assess` — multipart `image`, optional `prior_image`, optional
  `lesion_id` / `prior_timestamp` / `current_timestamp`.
- Returns one assessment per detected lesion: zero, one, or several.
- An undecodable upload is **400**, not a 500 with
  `TypeError: image_bgr must be a numpy.ndarray`.
- A non-ASSESSED outcome returns **200** with `assessed=false`, an empty
  list and the reason named. The orchestrator is explicit that
  `QUALITY_REJECTED` / `NO_CANDIDATES` must not read as "we looked and
  it was fine", so an empty success can never be mistaken for "no risk".

**Verified against Section A's own acceptance criterion (item 3):**
`tests/test_assess_api.py` posts the same real images and asserts the
response equals the delivered fixture payloads exactly — which is what
proves this is transport and not a second implementation of the
contract. The response is also parsed by the RAG side's own parser in
the same test. 9 tests; checkpoint-backed ones skip when the weights or
the dataset volume are unavailable.

Live over HTTP: first visit **0.54s** end-to-end, matching the p50, so
the transport layer adds nothing measurable. Full loop — HTTP → parser →
grounded answer — runs in ~4.6s for a returning visit.

---

## 14. Prior-measurement caching (2026-09-11)

**ADDED** — a returning visit re-segmented and re-measured the prior
image, which had already been measured at its own visit. That is ~0.5s
of repeated work per request on CPU.

`TemporalPipeline.assess_pair` now accepts `earlier_measurement`
instead of `earlier_image_bgr`; `predict()` takes `prior_measurement`
and, with `measure_current=True`, returns this visit's own measurement;
`/assess` hands that back as an opaque `measurement` token on the
envelope and accepts it as `prior_measurement`.

**It is an optimisation, not an approximation** — verified bit-identical,
including after a JSON round-trip: same verdict, same magnitude to nine
decimal places, same per-feature deltas. `compute_delta` consumes the
two measurements and never the pixels, which is what makes reuse exact.
A test asserts a cached returning visit reproduces the delivered fixture
payload exactly, so a future divergence fails loudly.

**Stateless by design.** The token round-trips through the caller, so
"who owns lesion history" (Section A question 4) does not have to be
answered to get the speedup.

A malformed token is a 400, never ignored: it feeds a real temporal
verdict, so a partially-defaulted measurement would yield a
confident-looking comparison against fabricated evidence — worse than
no comparison, which the contract already represents honestly as
`NO_PRIOR_DATA`.

**The measurement token is envelope-level, not inside an assessment.**
It describes the image rather than any one lesion, and keeping it out
preserves the property that assessments match
`docs/cv8_sample_outputs/` byte-for-byte. The fixture test caught this
when it was first placed wrongly.

### Result — and why the answer depends on image size

| request shape | 624×624 photo | 6000×4000 archival |
|---|---|---|
| first visit (returns token) | 1.25s | 2.05s |
| returning, prior image uploaded | 1.76s | 2.89s |
| returning, cached token | **1.23s** | 2.06s |
| **worst case** | **1.25s — passes** | 2.06s — fails |

29% better on both. But the earlier benchmarks used 24-megapixel
dermoscopic archive images, which no phone produces; on realistic photo
sizes the worst case clears the 2.0s bar with room.

Profiling also corrected two guesses: ruler calibration is ~1% of
measurement time (not a cost worth removing), and segmentation is
78–92% at ~0.54s, essentially fixed on CPU without a smaller model.

**So the sync/async question is really an image-size question**, and
that is a product decision — client-side downscaling before upload is
standard practice and saves bandwidth anyway.

---

## Change log

| Date | Commit | Change |
|---|---|---|
| 2026-09-11 | `84d8b5a` | **`integration-v1`** — CV and RAG baselines merge |
| 2026-09-11 | `bc7d70b` | CV context schema + parser; fix Blockers A and B; thread `cv_context` through the pipeline |
| 2026-09-11 | `b2841d2` | Safety-layer findings note for the collaborator; this file |
| 2026-09-11 | `b70e8dc` | Phase 2: CV modes in the CLI, CV-integration gate, corpus false-positive measurement |
| 2026-09-11 | `d6170b1` | CV-grounded tab in the Streamlit demo; citation format pinned; gate criterion 9 |
| 2026-09-11 | `9a65b29` | Containment metric for CV grounding; banned-phrase check narrowed (clause-scoped); CV-8 latency measured; Phase 2 gate 5/5 |
