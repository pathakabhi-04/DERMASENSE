# Update since the integration PR

**Date:** 2026-09-11 · **Branch:** `main` @ `8e1bc61`
**Previous notes:** `cv8_contract_delta_v1.1.md`, `rag_safety_findings.md`
**Full divergence log:** `docs/rag/CHANGES.md`

Four things landed after the PR. Three need nothing from you; one is a
question only you can answer.

---

## 1. Your Phase 1 gate criterion 3 now has an implementation

`EvidenceBundle.top_score` was defined and unit-tested, but nothing read
it — so §6's "states uncertainty explicitly whenever retrieval returned
low-similarity evidence" had no implementation. The gate could not fail
on it, which means running it would have certified less than it claims.

Threshold calibrated from the real distribution, per your own
instruction not to invent a number:

| bucket | min | p50 | max |
|---|---|---|---|
| in-scope (your 16 eval queries) | 0.3415 | 0.6930 | 0.8001 |
| dermatology-adjacent, uncovered | 0.4285 | 0.4639 | 0.5650 |
| out-of-scope | 0.0439 | 0.1220 | 0.1843 |

`LOW_SIMILARITY_THRESHOLD = 0.45` sits in a real gap inside the in-scope
set — 0.3415, then nothing until 0.4847 — so it flags exactly one
legitimate query: case 11, your own known Top-1 miss, which is the
answer that *should* hedge.

**A limit worth stating:** it does not cleanly separate covered from
uncovered. The dermatology-adjacent band overlaps genuine in-scope
queries (0.5280 is real), and no single threshold splits them. Erring
toward hedging is deliberate — a false positive costs a sentence, where
a grounding-check false positive costs the whole answer.

Verified live: "How do I treat psoriasis?" now says the sources don't
cover psoriasis treatment; the 0.7751 melanoma query doesn't hedge.

**Your 16-query gate should now be worth running.**

---

## 2. Citation format was unpinned

Spotted in a real UI answer: the model cited `【2†L9-L13】`. Your
formatter emits `[1] Source: <title>` and nothing finer, so those line
numbers refer to nothing — fabricated traceability, which in a medical
answer is its own kind of wrong even when the content is grounded.

Cause: there was no citation instruction in the prompt at all, so the
model used its training convention. Both branches now specify the
supplied `[n]` numbers.

---

## 3. `POST /assess` exists

A thin FastAPI layer over `DermaSensePipeline.predict()` — no new CV
logic. Tests assert its responses equal the delivered fixtures exactly,
and parse them with your own parser.

**Each response carries a `measurement` token on the envelope** (not
inside an assessment — the assessments stay byte-identical to
`docs/cv8_sample_outputs/`). Store it, send it back as
`prior_measurement` next visit instead of re-uploading the previous
photo. It removes ~0.5s per returning visit and is bit-identical to
re-measuring — same verdict, same magnitude to nine decimal places.
Treat it as opaque; send back exactly what you were given.

This keeps the service stateless, so "who owns lesion history" does not
have to be answered to get the speedup.

---

## 4. The one question for you: maximum upload size

We said we'd bring the sync-vs-async question back with a number. It
turned out to be an image-size question:

| worst-case request | 624×624 photo | 6000×4000 archival |
|---|---|---|
| before caching | 1.76s | 2.89s |
| **after caching** | **1.25s** | 2.06s |

Our earlier benchmarks used 24-megapixel dermoscopic archive images,
which no phone produces. On realistic photo sizes a synchronous call is
comfortably viable; on 24MP originals it is not.

**So: does your client downscale before upload, and to what maximum
dimension?** Downscaling to ~1000px is standard for photo upload and
saves bandwidth anyway. If yes, sync is settled and no async design is
needed. If you cannot control upload size, tell us and we will take the
async question seriously.

Numbers are CPU-only. GPU is not being assumed — it is not financeable
for deployment at this stage.

---

## Nothing else needed from you

Phase 2's gate passes 5/5 on nine criteria. 273 tests pass, including
all 53 of yours, unmodified — one of which caught a real bug we
introduced this week (a prompt rewrite briefly dropped the
`RETRIEVED EVIDENCE` section entirely). They are load-bearing; we have
not touched them.

Still open and still yours: the 6-vs-8-class taxonomy, and whether to
revisit the Groq temperature now that the banned-phrase check no longer
fires on 25% of ordinary medical prose.
