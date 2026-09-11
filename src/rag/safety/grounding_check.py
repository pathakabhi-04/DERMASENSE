from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from src.rag.cv_context.schema import render_cv_context
from src.rag.retrieval.evidence import EvidenceBundle

if TYPE_CHECKING:
    from src.rag.cv_context.schema import CVAssessmentContext

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
    cv_context: "CVAssessmentContext | list[CVAssessmentContext] | None" = None,
) -> bool:
    """
    Return True if the answer shows meaningful lexical overlap with at
    least one SUPPLIED SOURCE -- a retrieved evidence chunk, or the CV
    assessment -- a proxy for "the answer is actually about what it was
    given," not a full entailment check.

    ## Why CV context counts as a source

    Spec section 4.3 puts CV context "on exactly the same footing as
    retrieved medical passages": evidence the LLM may explain but not
    originate. Grounding must therefore be measured against both, or
    the check contradicts the architecture it exists to enforce.

    Measured on this repo's index before the change: a realistic
    CV-grounded narration ("your lesion was assessed as medium risk
    with 84% confidence; no meaningful change since your last photo")
    scored 0.0000 against the three chunks actually retrieved for a
    matching query, and at best 0.0412 against any of the 156 chunks in
    the corpus -- against a 0.12 threshold. It is not corpus content,
    so it cannot resemble corpus content.

    The consequence was perverse: the better an answer explained the
    patient's own assessment, the more certainly it was discarded, and
    the LLM's only way to pass was to pad with corpus boilerplate --
    the opposite of the constrained-paraphrase goal.

    Returns False (unsafe) when NOTHING was supplied: an answer cannot
    be grounded in sources that never existed.
    """

    if not answer_text.strip():
        return False

    sources: list[str] = [chunk.text for chunk in evidence.chunks]

    cv_block = render_cv_context(cv_context)
    if cv_block:
        sources.append(cv_block)

    if not sources:
        return False

    return any(scorer(answer_text, source) >= threshold for source in sources)


def run_safety_check(
    answer_text: str,
    evidence: EvidenceBundle,
    source_presence_threshold: float = DEFAULT_SOURCE_PRESENCE_THRESHOLD,
    cv_context: "CVAssessmentContext | list[CVAssessmentContext] | None" = None,
) -> SafetyCheckResult:
    """
    Run both deterministic checks (spec section 5, points 1-2) and
    combine them into one pass/fail result.

    `cv_context`, when supplied, counts as a grounding source alongside
    the retrieved evidence -- see `check_source_presence`.
    """

    banned_phrase_violation = check_banned_phrases(answer_text)

    is_source_grounded = check_source_presence(
        answer_text,
        evidence,
        threshold=source_presence_threshold,
        cv_context=cv_context,
    )

    source_presence_violation = not is_source_grounded

    if banned_phrase_violation:
        reason = (
            "Answer contains a direct-diagnosis claim "
            "(certainty phrase + condition name in one sentence)."
        )
    elif source_presence_violation:
        reason = (
            "Answer does not show sufficient lexical overlap with any "
            "supplied source (retrieved evidence chunk or CV assessment)."
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


def build_fallback_answer(
    evidence: EvidenceBundle,
    cv_context: "CVAssessmentContext | list[CVAssessmentContext] | None" = None,
) -> str:
    """
    Required fallback behavior (spec section 5, points 3-4): when the
    answer fails the safety check, or the LLM call itself fails or
    times out, return the retrieved evidence directly without LLM
    narration, plus a note that a full explanation isn't available.
    A user must never see nothing or a bare error when evidence was
    successfully retrieved.

    `cv_context` is REQUIRED to reach the user whenever one exists, and
    is rendered FIRST. Spec section 5.4 is explicit about why:

        "A user must never see nothing, or an error page, when
        structured CV evidence was successfully computed -- that
        evidence is safety-relevant and must reach them even if
        narration fails."

    Before this parameter existed, a failed LLM call on a lesion
    assessed HIGH risk and flagged for review produced a page of
    general corpus text that never mentioned the assessment at all.
    That is the exact failure the spec forbids, and it is the common
    case rather than a rare one: CV answers reach this path whenever
    narration fails for any reason.

    Multiple lesions are ordered most-severe-first (spec section 20);
    none is dropped.
    """

    cv_block = render_cv_context(cv_context)

    if evidence.is_empty and not cv_block:
        return (
            "A full explanation isn't available right now, and no "
            "relevant medical evidence was found for this question. "
            "Please consult a healthcare professional."
        )

    parts = ["A full explanation isn't available right now."]

    if cv_block:
        parts.append(
            "Here is the assessment of your photo, and the relevant "
            "medical information, directly:\n\n" + cv_block
            if not evidence.is_empty
            else "Here is the assessment of your photo directly:\n\n" + cv_block
        )

    if not evidence.is_empty:
        parts.append(
            (
                "Relevant medical information:\n\n"
                if cv_block
                else "Here is the relevant evidence directly:\n\n"
            )
            + evidence.format_for_prompt()
            + "\n\n"
            + evidence.format_sources_line()
        )

    return "\n\n".join(parts)
