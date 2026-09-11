"""
Phase 1 answer-evaluation gate (primary spec section 6).

Never run before now: it was deferred over Gemini's 20-request/day cap
(rag_development_state.md section 14), then the provider changed and the
gate was never revisited.

Fixed sample: the 16 queries in retrieval_cases.json, reused for
continuity exactly as section 6 requires -- no new question set.

Pass criteria per answer, binary, from section 6 verbatim:

1. cites at least one real retrieved source
2. contains no banned-phrase diagnostic claim (section 5.1)
3. states uncertainty explicitly whenever retrieval returned
   low-similarity evidence

Gate: 100% of the fixed set passes all three.

Criterion 3 is only checkable now that `top_score` is actually consumed
(LOW_SIMILARITY_THRESHOLD in evidence.py). Its hedge detection is a
keyword heuristic over the answer text: it can only be evidence that
hedging language is present, not proof the hedge is apt. Failures are
therefore worth reading rather than trusting blindly, which is why each
answer is written out.

    python -m src.rag.evaluate_phase1_gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.rag.embeddings.embedder import SentenceTransformerEmbedder
from src.rag.llm.groq_adapter import GroqAdapter
from src.rag.pipeline import RagAnswerPipeline
from src.rag.prompts.prompt_builder import PromptBuilder
from src.rag.retrieval.evidence import EvidenceFormatter
from src.rag.retrieval.retriever import MedicalRetriever
from src.rag.safety.grounding_check import check_banned_phrases
from src.rag.vectorstore.faiss_store import FAISSVectorStore

INDEX_PATH = Path("data/rag/indexes/medical_v0.1")
CASES_PATH = Path("src/rag/retrieval/retrieval_cases.json")
LOG_PATH = Path("evaluation/rag/phase1_gate_answers.json")

# Language that acknowledges the evidence may not settle the question.
HEDGE_MARKERS = (
    "not directly", "does not directly", "doesn't directly", "only weakly",
    "not specifically", "does not address", "doesn't address", "not cover",
    "does not cover", "doesn't cover", "no information", "not contain",
    "does not contain", "doesn't contain", "limited", "does not provide",
    "doesn't provide", "not enough", "insufficient", "does not mention",
    "doesn't mention", "not mention", "beyond the", "outside the",
)


def build_pipeline() -> RagAnswerPipeline:
    store = FAISSVectorStore.load(INDEX_PATH)
    return RagAnswerPipeline(
        evidence_formatter=EvidenceFormatter(
            retriever=MedicalRetriever(
                embedder=SentenceTransformerEmbedder(), vector_store=store
            )
        ),
        prompt_builder=PromptBuilder(),
        llm_adapter=GroqAdapter(),
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    queries = [c["query"] for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))]
    print(f"Loading embedding model and index... ({len(queries)} fixed queries)")
    pipeline = build_pipeline()

    rows, log = [], []

    for index, query in enumerate(queries, start=1):
        evidence = pipeline.evidence_formatter.get_evidence(query)
        answer = pipeline.answer(query)
        low = evidence.is_low_similarity
        hedges = [h for h in HEDGE_MARKERS if h in answer.text.lower()]

        checks = {
            "1_cites_source": (
                answer.sources_line.startswith("Sources: ")
                and answer.sources_line != "Sources: none"
            ),
            "2_no_banned_phrase": not check_banned_phrases(answer.text),
            # Vacuously true when retrieval was strong: the criterion only
            # binds on low-similarity retrieval.
            "3_uncertainty_when_weak": bool(hedges) if low else True,
        }
        rows.append(checks)
        log.append({
            "n": index, "query": query, "top_score": evidence.top_score,
            "low_similarity": low, "used_fallback": answer.used_fallback,
            "fallback_reason": answer.fallback_reason,
            "hedges": hedges, "checks": checks, "answer": answer.text,
        })

        failed = [k for k, ok in checks.items() if not ok]
        flag = "low " if low else "    "
        status = "PASS" if not failed else f"FAIL ({', '.join(failed)})"
        print(f"  [{index:2}] {evidence.top_score:.4f} {flag} "
              f"fallback={str(answer.used_fallback):5} {status}  {query[:44]}")

    print("\n" + "=" * 70)
    print("PHASE 1 ANSWER-EVALUATION GATE (spec section 6)")
    print("=" * 70)
    for name in sorted(rows[0]):
        passed = sum(bool(r[name]) for r in rows)
        mark = "PASS" if passed == len(rows) else "FAIL"
        print(f"  {mark}  {name:<28} {passed}/{len(rows)}")

    total = sum(all(r.values()) for r in rows)
    low_n = sum(entry["low_similarity"] for entry in log)
    fb_n = sum(entry["used_fallback"] for entry in log)
    print(f"\n  low-similarity queries : {low_n}/{len(rows)} "
          f"(criterion 3 binds on these)")
    print(f"  answers that fell back : {fb_n}/{len(rows)}")
    print(f"  passing all three      : {total}/{len(rows)}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  answers written to       {LOG_PATH}")

    gate = total == len(rows)
    print(f"\n  GATE: {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
