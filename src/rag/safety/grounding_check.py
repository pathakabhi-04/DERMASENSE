from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from src.rag.retrieval.evidence import EvidenceBundle

# Direct-diagnosis certainty phrases, taken verbatim from the
# primary specification (section 5, point 1). A violation requires
# one of these AND a disease/condition name to appear in the same
# sentence -- neither alone is dangerous. This is intentionally
# broad and will over-flag some benign sentences (e.g. "this is a
# common growth"); the spec accepts that tradeoff at baseline and
# defers precision tuning to a later phase, based on real observed
# failures rather than preemptive narrowing.
CERTAINTY_PHRASES = [
    "you have",
    "this is",
    "confirmed",
    "definitely",
    "diagnosed with",
]

# Condition names drawn from the CV-native taxonomy (spec section
# 1.1/§18) plus the general terms used across the acquired corpus.
CONDITION_NAMES = [
    "melanoma",
    "basal cell carcinoma",
    "squamous cell carcinoma",
    "actinic keratosis",
    "actinic keratoses",
    "seborrheic keratosis",
    "seborrheic keratoses",
    "nevus",
    "nevi",
    "mole",
    "moles",
    "skin cancer",
]

DEFAULT_SOURCE_PRESENCE_THRESHOLD = 0.12

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z]{4,}")

# Common words excluded from the lexical-overlap check so overlap
# reflects shared medical content, not shared function words.
_STOPWORDS = {
    "this", "that", "with", "from", "have", "your", "about", "into",
    "than", "then", "them", "they", "were", "been", "should", "would",
    "could", "there", "these", "those", "when", "what", "which",
    "while", "based", "provided", "evidence", "answer", "question",
}


@dataclass
class SafetyCheckResult:
    """
    Result of the deterministic safety/grounding check (spec section
    5). `passed` is False if either sub-check fails; `reason`
    explains which one and why, for logging/fallback messaging.
    """

    passed: bool
    banned_phrase_violation: bool
    source_presence_violation: bool
    reason: str | None


def check_banned_phrases(answer_text: str) -> bool:
    """
    Return True if the answer contains a direct-diagnosis claim: a
    certainty phrase and a condition name co-occurring in the same
    sentence.

    Deterministic, no model call. Intentionally biased toward
    over-flagging -- a false positive costs a fallback to plain
    evidence, not a wrong diagnosis reaching the user.
    """

    sentences = _SENTENCE_SPLIT_RE.split(answer_text)

    for sentence in sentences:
        lowered = sentence.lower()

        has_certainty_phrase = any(
            phrase in lowered for phrase in CERTAINTY_PHRASES
        )

        if not has_certainty_phrase:
            continue

        has_condition_name = any(
            condition in lowered for condition in CONDITION_NAMES
        )

        if has_condition_name:
            return True

    return False


def _lexical_overlap(text_a: str, text_b: str) -> float:
    """
    Cheap, deterministic similarity: Jaccard overlap of lowercase
    words (length >= 4, common function words excluded). No model
    call -- the baseline's stated default (spec section 5, point 2).
    """

    words_a = set(_WORD_RE.findall(text_a.lower())) - _STOPWORDS
    words_b = set(_WORD_RE.findall(text_b.lower())) - _STOPWORDS

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)


SimilarityScorer = Callable[[str, str], float]


def check_source_presence(
    answer_text: str,
    evidence: EvidenceBundle,
    scorer: SimilarityScorer = _lexical_overlap,
    threshold: float = DEFAULT_SOURCE_PRESENCE_THRESHOLD,
) -> bool:
    """
    Return True if the answer shows meaningful lexical overlap with
    at least one retrieved evidence chunk -- a proxy for "the answer
    is actually about the evidence," not a full entailment check.

    Returns False (unsafe) if there is no evidence at all: an answer
    cannot be grounded in evidence that was never retrieved.
    """

    if evidence.is_empty or not answer_text.strip():
        return False

    return any(
        scorer(answer_text, chunk.text) >= threshold
        for chunk in evidence.chunks
    )


def run_safety_check(
    answer_text: str,
    evidence: EvidenceBundle,
    source_presence_threshold: float = DEFAULT_SOURCE_PRESENCE_THRESHOLD,
) -> SafetyCheckResult:
    """
    Run both deterministic checks (spec section 5, points 1-2) and
    combine them into one pass/fail result.
    """

    banned_phrase_violation = check_banned_phrases(answer_text)

    is_source_grounded = check_source_presence(
        answer_text,
        evidence,
        threshold=source_presence_threshold,
    )

    source_presence_violation = not is_source_grounded

    if banned_phrase_violation:
        reason = (
            "Answer contains a direct-diagnosis claim "
            "(certainty phrase + condition name in one sentence)."
        )
    elif source_presence_violation:
        reason = (
            "Answer does not show sufficient lexical overlap with "
            "any retrieved evidence chunk."
        )
    else:
        reason = None

    return SafetyCheckResult(
        passed=not (
            banned_phrase_violation or source_presence_violation
        ),
        banned_phrase_violation=banned_phrase_violation,
        source_presence_violation=source_presence_violation,
        reason=reason,
    )


def build_fallback_answer(evidence: EvidenceBundle) -> str:
    """
    Required fallback behavior (spec section 5, points 3-4): when the
    answer fails the safety check, or the LLM call itself fails or
    times out, return the retrieved evidence directly without LLM
    narration, plus a note that a full explanation isn't available.
    A user must never see nothing or a bare error when evidence was
    successfully retrieved.
    """

    if evidence.is_empty:
        return (
            "A full explanation isn't available right now, and no "
            "relevant medical evidence was found for this question. "
            "Please consult a healthcare professional."
        )

    return (
        "A full explanation isn't available right now. Here is the "
        "relevant evidence directly:\n\n"
        f"{evidence.format_for_prompt()}\n\n"
        f"{evidence.format_sources_line()}"
    )
