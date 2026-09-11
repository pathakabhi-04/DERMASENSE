# For the RAG side — syncing up, and what we need from you

**Updated:** 2026-09-12 · **Branch to use:** `main`
**What changed in your code:** `docs/rag/CHANGES.md` (marked ADDED /
EXTENDED / ⚠️ CHANGED / NOT CHANGED)

---

## 1. Get up to date (5 minutes)

`rag-development` is now **superseded**. Everything on it is in `main` —
your 17 commits are genuine ancestors there, not a copy, so
`git log`/`git blame` on `src/rag` still credit you.

```bash
git fetch origin
git switch main          # or: git checkout main
git pull
```

Verify:

```bash
git log --oneline | grep "feat(rag)" | head -3   # your commits, present
git log --oneline --all | grep integration-v1    # the merge point
```

**Don't keep working on `rag-development`** — it has no CV code, so the
CV-8 fixtures and `src/risk/` aren't there and half the tests can't run.
If you have unpushed work on it:

```bash
git switch main
git cherry-pick <your-commit-sha>   # per commit
```

### Environment

```bash
pip install -r requirements.txt      # adds fastapi; faiss/sentence-transformers unchanged
```

**Your `GROQ_API_KEY` needs rotating** — the `.env` was inside
`rag_baseline.zip`. Put the new one in `.env` at the repo root (already
gitignored; `.env.example` is the template).

The FAISS index is gitignored, so it does not arrive via git. If you
don't already have `data/rag/indexes/medical_v0.1/`:

```bash
python -m src.rag.ingestion.acquire_corpus
python -m src.rag.vectorstore.build_index
```

### Check it works

```bash
python -m pytest src/rag -q                  # 115 pass, incl. all 53 of yours
python -m src.rag.retrieval.evaluate_retrieval   # Top-1 15/16, Top-3 16/16
streamlit run src/rag/streamlit_app.py       # now has a CV-grounded tab
```

---

## 2. ⚠️ Watch out: the Groq free tier is 200,000 tokens/day

Not requests — **tokens**, for `openai/gpt-oss-120b`. We burned a full
day's budget re-running evaluations, and the failures surfaced
confusingly (some as 429, some as DNS errors), which cost hours.

Rough costs:

| Run | Calls | ~Tokens |
|---|---|---|
| `evaluate_phase1_gate` (16 + 18 stress) | 34 | ~60–70k |
| `evaluate_cv_integration` | 5 | ~10k |

**About three full gate runs exhausts the day.** This is the same wall
that made you defer the Phase 1 gate under Gemini — same problem,
different provider. Worth budgeting before a batch run.

---

## 3. What we need from you

### 3.1 Sign off (or veto) two changes to your safety layer

Both in `src/rag/safety/grounding_check.py`, both reversible, both
justified by measurement in `docs/rag_safety_findings.md`. They are
marked ⚠️ CHANGED in `CHANGES.md`. We made them because we had your full
codebase and the evidence was decisive — not because the decisions
stopped being yours.

1. **Grounding accepts CV context as a source** (containment metric,
   threshold 0.25). Without it, 0 of 5 CV-grounded answers passed.
2. **Banned-phrase check narrowed** to require proximity *and* no hedge
   governing the clause. The original rule flagged **39 of 156 corpus
   chunks (25%)** — including *"A dermatologist can tell you if you have
   basal cell carcinoma"*.

If you disagree with either, say so and we'll revert — they're isolated
to one file.

### 3.2 One decision only you can make: **maximum upload dimension**

This is the last thing blocking the sync-vs-async question, and it turns
out to be an image-size question, not a latency one:

| worst-case request | 624×624 photo | 6000×4000 |
|---|---|---|
| after caching | **1.25s** ✅ | 2.06s ❌ |

**Does your client downscale before upload, and to what maximum?**
~1000px is standard and saves bandwidth anyway. If yes, sync is settled
and no async design is needed. If you can't control upload size, tell us
and we'll take async seriously.

### 3.3 Optional: do you want per-feature delta narration?

If yes we'll ship thresholds or `crossed_threshold` booleans as contract
1.2. If no, the current "never narrate deltas numerically" rule is
enough. We haven't built it speculatively.

### 3.4 Still jointly open

The 6-class vs 8-class taxonomy. Neither side should guess.

---

## 4. Things you can now do without waiting for us

- **`POST /assess`** is live: `python -m uvicorn src.serving.assess_api:app --port 8000`.
  Returns the locked contract unchanged, plus an opaque `measurement`
  token on the envelope — store it, send it back as `prior_measurement`
  next visit instead of re-uploading the previous photo (~0.5s saved,
  bit-identical results).
- **`src/rag/cv_context/`** parses a CV-8 payload into
  `CVAssessmentContext`. Delivery-agnostic by design, exactly as your §9
  asked.
- **Both gates** are runnable: `evaluate_phase1_gate`,
  `evaluate_cv_integration`.

---

## 5. Two things worth knowing about your baseline

Found while integrating; both are in the Phase 1 baseline rather than
anything CV introduced.

**An uncovered question returned dangerous content.** *"How is impetigo
treated?"* (top_score 0.2622) returned FDA dosing instructions for basal
cell carcinoma chemotherapy, introduced as *"Here is the relevant
evidence directly"*. Retrieval has no score floor, so it always returns
its top-k however poorly they match; the model correctly said the
sources don't cover impetigo; the grounding check correctly rejected
that (a statement about the *absence* of evidence can't overlap the
evidence); and the fallback dumped the unrelated chunks as relevant.
Fixed in `build_fallback_answer` — see `CHANGES.md` §17.

**Your §6 criterion 3 had no implementation.** `top_score` was computed
and unit-tested but nothing read it, so "states uncertainty when
retrieval was low-similarity" could only pass by the model happening to
hedge. Now wired, with the threshold calibrated from your real score
distribution as your spec required. `CHANGES.md` §15.
