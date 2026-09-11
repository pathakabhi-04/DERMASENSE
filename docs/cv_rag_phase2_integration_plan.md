# CV → RAG Phase 2 integration plan

**Date:** 2026-09-11
**Status:** Plan — written against the RAG side's real code, not its spec
**Inputs:** `rag_baseline.zip` (collaborator, 2026-09-11, 155 files), `docs/rag/RAG_EVOLUTION_AND_CV_INTEGRATION_STRATEGY_v2.md`, `docs/rag/rag_development_state.md`
**Companion:** `docs/cv8_contract_delta_v1.1.md` (the CV-side fixes)

> **Status: EXECUTED.** Written before the integration work, and kept as
> the record of what was found by reading their code rather than their
> spec. Both blockers in §1 and §2 are now fixed; §3's `top_score` gap
> remains open and remains theirs. Current state is in
> `docs/rag/CHANGES.md`, which supersedes the plan sections below.

---

## 0. Summary

The RAG baseline is real, complete, and well-built: retrieval → evidence →
prompt → Groq → deterministic safety check → fallback, with unit tests at
every layer. Their architecture boundary (`CV computes, RAG retrieves, LLM
explains`) is implemented faithfully, not just documented.

**No CV-8 parser exists yet** — no `CVAssessmentContext`, no
`cv_context/` module, `memory/` and `patient_context/` are empty
placeholders. Phase 2 has not started. That means the v1.1 contract fix
landed before the wrong confidence figure got baked into a parser, which
was the entire point of doing it first.

**Two blockers stand between the current code and Phase 2, both in the
safety layer, both anticipated by their own spec §5 but not implemented
for CV context.** Both are on the RAG side; neither is a design error;
both are cheap to fix *before* the parser is written and expensive after.

---

## 1. Blocker A — the grounding check rejects CV-grounded answers

### The mechanism

`check_source_presence` (`src/rag/safety/grounding_check.py:126`) requires
lexical Jaccard overlap ≥ `0.12` between the answer and **at least one
retrieved medical chunk**. `RagAnswerPipeline.answer` treats a failure as
`used_fallback=True`.

A Phase 2 answer's most valuable content is about *this patient's lesion* —
risk category, calibrated confidence, temporal verdict. That content has
no lexical overlap with a corpus page about actinic keratosis, because it
isn't corpus content. It's CV output.

### Measured, not assumed

Using their real `_lexical_overlap` and their real shipped 156-chunk index
(`data/rag/indexes/medical_v0.1/chunks.json`), against a realistic
CV-grounded narration of our fixture 2 (ACK, MEDIUM, 84% calibrated, STABLE):

| Answer style | Best overlap across the WHOLE index | Verdict |
|---|---|---|
| CV-only narration (risk + temporal, no corpus recitation) | **0.0412** | **falls back** — 2.9× below threshold |
| CV narration blended with corpus vocabulary | 0.1356 | passes, but only 13% of margin |
| Corpus-only narration (Phase 1 style, control) | 0.1429 | passes |

0.0412 is the *most favourable single chunk in the entire corpus*, not an
average. There is no retrieval result that rescues it.

### Why this is worse than a false negative

The fallback is not a soft degradation. It replaces the narration with raw
evidence text — so the failure mode is: **the better the answer is at
explaining the patient's actual CV assessment, the more likely it is to be
thrown away.** And it pushes the LLM toward padding answers with corpus
boilerplate to clear the threshold, which is the opposite of what the
constrained-paraphrase design wants.

### Fix (RAG side)

Treat CV context as a first-class grounding source alongside retrieved
evidence — an answer grounded in the CV payload *is* grounded. Their §4.3
already says CV context is "supplied evidence on exactly the same footing
as retrieved medical passages"; the safety layer just hasn't been told.

Smallest change consistent with that: give `check_source_presence` the CV
context too, and pass if the answer overlaps *either* a retrieved chunk or
the CV context's rendered text. Threshold should be re-derived from real
CV narrations, the same way they derived `0.12` from real data (their
§13) — not reused on faith.

---

## 2. Blocker B — the fallback silently drops the CV risk signal

### The mechanism

```python
build_fallback_answer(evidence: EvidenceBundle) -> str   # line 193
```

It takes retrieved medical evidence only. `RagAnswerPipeline.answer` calls
it on both fallback paths (LLM error, safety failure).

So once CV context exists: the LLM call fails, and the user receives
medical corpus text about actinic keratosis — with **no mention that their
lesion was assessed MEDIUM risk and flagged for review.**

### This violates their own spec, verbatim

> §5.4: "A user must never see nothing, or an error page, when structured
> CV evidence was successfully computed — **that evidence is safety-relevant
> and must reach them even if narration fails.**"

> §5.3: "losing the underlying safety-relevant signal because the narration
> step failed would be strictly worse than a plainer but correct answer."

The spec is right. The signature predates CV context and doesn't implement
it. Note this compounds with Blocker A: fallbacks will be *common* for CV
answers, not rare, so this path is the one users would actually hit.

### Fix (RAG side)

`build_fallback_answer(evidence, cv_context=None)`, rendering the CV
assessment — risk category, calibrated confidence, `requires_review`,
temporal verdict — before the corpus evidence. Unnarrated is fine; absent
is not.

---

## 3. Also worth fixing now: Phase 1 gate criterion 3 has no implementation

`EvidenceBundle.top_score` (`evidence.py:38`) exists and is unit-tested.
Its docstring says "Downstream components (prompt builder, safety layer)
use this to decide when to state uncertainty explicitly."

Nothing reads it. `grep top_score` hits only `evidence.py` and its own
test file. There is no threshold, no low-similarity branch in
`PromptBuilder.build`, and no uncertainty logic in `run_safety_check`.

Their Phase 1 gate (§6) requires:

> "States uncertainty explicitly whenever retrieval returned low-similarity
> evidence (define a threshold from the actual score distribution)."

With no implementation, that criterion can only pass by accident — the LLM
happening to hedge. Worth closing before the gate run, otherwise the gate
certifies something untested. Their own `rag_development_state.md` §7
already identifies the score range to calibrate against (case 11, ~0.33).

**This is Phase 1, not Phase 2** — it does not block us, and they should
not wait on the CV side for it.

---

## 4. What does NOT need to change

Confirmed by reading the code rather than assuming:

- **The generation approach.** Their §4.3 claim that constrained paraphrase
  survives CV integration unchanged is correct. `SYSTEM_PROMPT`
  (`prompt_builder.py:7`) already says "the evidence or the structured CV
  context you are given" — written for this.
- **The evidence formatter.** Top-3-deduplicated-by-document is
  implemented exactly as specced (`select_top_evidence`), with a wider
  candidate pool so dedup has something to draw from. No change needed.
- **The retriever, index, chunking, embeddings.** Out of scope, and their
  §7 rightly defers optimization.
- **`RagAnswer`.** Adding CV context needs no new field on the result type.

---

## 5. Interface

Their §25 target is already the right shape:

```python
response = rag_pipeline.answer(
    query=user_question,
    cv_context=cv_assessment,
    patient_context=patient_context,
)
```

Current signature is `answer(self, query: str)`. The extension is purely
additive — `cv_context=None` keeps every Phase 1 caller (CLI, Streamlit,
5 pipeline tests) working unchanged.

### One design decision to make before writing the parser: multi-lesion

**Take `cv_contexts: list | None`, not `cv_context: single | None`.**

CV-8 emits one assessment *per detected lesion candidate*, not per image —
`PipelineResult.candidates` is a list, so multi-lesion is a real code path,
not hypothetical. Their §20 baseline rule (lead with the highest
`risk_category`, mention the others, never silently drop one) and their
§26 Test 7 both require a list.

Building it singular now means refactoring the parser, the prompt builder,
and the fallback later. Cheap now, annoying later.

---

## 6. Ownership and sequence

### CV side (us) — done

- ✅ Contract v1.1: calibrated confidence in `risk_reason`, no numeric
  `magnitude`, `contract_version` added. `3a49231`.
- ✅ Fixtures regenerated against real checkpoints; delta is 10 lines, all
  other values byte-identical. 154/154 tests pass.
- ✅ `docs/cv8_contract_delta_v1.1.md` — the hand-off note.

### CV side (us) — open

1. **Answer their per-feature-delta question** if they want per-feature
   narration → ship thresholds or `crossed_threshold` booleans as 1.2.
   Not built speculatively.
2. **Measure `DermaSensePipeline.predict()` latency** with CV-5/CV-6
   enabled, one real image. Pre-committed decision rule: p50 < 2s → build
   the sync FastAPI wrapper per `build_on_baseline_1.md` §A; ≥ 2s → the
   sync/async question goes back to them before anything is built. One
   measurement, one decision.

### RAG side (them) — before writing the parser

3. Fix `check_source_presence` to accept CV context (Blocker A).
4. Fix `build_fallback_answer` to accept CV context (Blocker B).
5. Decide `cv_contexts` as a list (§5 above).

### RAG side (them) — Phase 1, independent of us

6. Implement the `top_score` → uncertainty path (§3 above).
7. Run the 16-query, 100%-pass Phase 1 gate.

### Joint — Phase 2

8. They build `CVAssessmentContext` + parser against the **v1.1** fixtures:
   fail loudly on missing key / unknown enum / unrecognized MAJOR;
   all three `per_feature_deltas` independently nullable; never parse
   `compared_timestamps` as dates; read `entry["payload"]`.
9. Prompt builder gains a CV CONTEXT block; system prompt unchanged.
10. Gate: their §26 Tests 1–7 pass. Tests 6 (malformed input) and 7
    (multi-lesion) are the two that protect both sides.

---

## 7. Merge mechanics

`main` and `rag-development` overlap in two files only:

- `.gitignore` — union merge, trivial.
- `requirements.txt` — semantic. They added `; sys_platform == "linux"`
  markers to all 15 `nvidia-*` packages and `triton` (correct — no Windows
  wheels exist for those), plus `faiss-cpu`, `sentence-transformers`,
  `streamlit`. We added `matplotlib`, `opencv-python`, `umap-learn`,
  `numba`, `pytest`. Resolution is take-both, keeping their markers.

The zip ships `data/rag/indexes/medical_v0.1/` (156 chunks, 384-dim,
`medical.faiss` + `chunks.json`), so the index does not need rebuilding
locally — but it is gitignored on their branch, so it will not arrive via
git. Keep the zip copy.

---

## 8. Security note

`rag_baseline.zip` contained a `.env` file (70 bytes) alongside
`.env.example`. Their `rag_development_state.md` §11 confirms
`GEMINI_API_KEY`/Groq credentials live there. `.gitignore` does not apply
to zips. **The Groq key should be rotated.** It was excluded from
extraction on our side and not read.
