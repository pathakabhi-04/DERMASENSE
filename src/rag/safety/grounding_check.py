from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from src.rag.cv_context.schema import render_cv_context
from src.rag.retrieval.evidence import EvidenceBundle

if TYPE_CHECKING:
    from src.rag.cv_context.schema import CVAssessmentContext

# Direct-diagnosis certainty phrases, taken verbatim from the primary
# specification (section 5, point 1). Neither a certainty phrase nor a
# condition name is dangerous alone.
#
# The spec's original rule flagged the two CO-OCCURRING anywhere in a
# sentence, accepting over-flagging and deferring precision tuning
# until real failures justified it. Those failures arrived with CV
# integration (25% of the corpus self-flagged), so the rule is now
# narrowed -- see check_banned_phrases for the evidence and the two
# conditions that replaced bare co-occurrence.
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

# How far after a certainty phrase a condition name still counts as the
# thing being asserted. "you have melanoma" (0) is a claim; "this is
# simply the label the system uses to track the mole" (9) is not.
MAX_CERTAINTY_CONDITION_GAP = 3

# A hedge governs its own clause only -- see check_banned_phrases.
_CLAUSE_SPLIT_RE = re.compile(
    r"[,;:]|\b(?:but|although|though|because|however|while|whereas|so)\b"
)

# Markers that make a clause conditional, hypothetical, or statistical
# rather than an assertion about this patient.
_HEDGE_RE = re.compile(
    r"\b(?:if|whether|unless|should|in case|tell you if|can tell you|"
    r"to (?:see|find out|know|determine)|suspects?|might|may|could|"
    r"possible|possibly|risk of|likelihood of|chance of|more likely to|"
    r"likely to)\b"
)

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


def _certainty_condition_pairs(clause: str):
    """
    Yield (gap_in_tokens, text_before_the_certainty_phrase) for every
    certainty-phrase/condition-name pair in one clause, where the
    condition name follows the certainty phrase.
    """

    for phrase in CERTAINTY_PHRASES:
        start = 0
        while (index := clause.find(phrase, start)) != -1:
            after = clause[index + len(phrase):]
            for condition in CONDITION_NAMES:
                offset = after.find(condition)
                if offset != -1:
                    yield len(after[:offset].split()), clause[:index]
            start = index + 1


def check_banned_phrases(answer_text: str) -> bool:
    """
    Return True if the answer asserts a diagnosis: a certainty phrase
    followed closely by a condition name, in a clause that is not
    hedged or conditional.

    Deterministic, no model call. Still biased toward over-flagging --
    a false positive costs a fallback to plain evidence, not a wrong
    diagnosis reaching the user.

    ## Why this is narrower than bare co-occurrence

    The original rule flagged any sentence containing both a certainty
    phrase and a condition name. Measured against the indexed corpus,
    **39 of 156 chunks (25%) tripped it** -- authoritative AAD/NCI text
    that the system is supposed to ground answers in:

        "A dermatologist can tell you if you have basal cell carcinoma
         and if you do, what treatment is recommended."
        "the lifetime risk of being diagnosed with melanoma was 2.9%"
        "If you have a raised mole on skin that you shave, you may nick
         the mole."

    None is a diagnostic claim: two are conditionals, one an
    epidemiological statistic. A faithful constrained paraphrase
    inherits that phrasing and was rejected for accurately explaining
    the evidence it was given -- penalising exactly the behaviour the
    architecture asks for. CV narration made this acute, because such
    answers say "this is [the label / the risk category]" constantly
    while mentioning mole/nevus throughout.

    ## The two conditions, and why BOTH are needed

    - **Proximity** (condition within `MAX_CERTAINTY_CONDITION_GAP`
      tokens after the certainty phrase) rejects "this is simply the
      label the system uses to track the mole".
    - **No conditional/hedge governing the clause** rejects "can tell
      you *if* you have basal cell carcinoma", which proximity alone
      cannot -- its gap is 0, textually identical to a real claim.

    ## Why the hedge is scoped to the CLAUSE, not the sentence

    This is the part that is easy to get dangerously wrong. Scoping the
    hedge to the whole sentence means any earlier hedge suppresses the
    flag, so every one of these slips through:

        "If you were wondering, you have melanoma."
        "Although a biopsy may help, you have skin cancer."
        "It is possible to treat this, but you have melanoma."

    A sentence-scoped version missed 8 of 8 such cases in testing. A
    conditional governs its own clause and no further, so the clause is
    the correct unit. See `test_banned_phrase_precision.py` for the
    23-case true-positive set, 8 of which exist purely to hold this
    line, plus the measured corpus false-positive rate.
    """

    for sentence in _SENTENCE_SPLIT_RE.split(answer_text):
        for clause in _CLAUSE_SPLIT_RE.split(sentence.lower()):
            if not clause:
                continue

            for gap, before in _certainty_condition_pairs(clause):
                if gap > MAX_CERTAINTY_CONDITION_GAP:
                    continue
                if _HEDGE_RE.search(before):
                    continue
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


def _containment(answer_text: str, source_text: str) -> float:
    """
    What fraction of the SOURCE's vocabulary appears in the answer:
    `|A n B| / |B|`, not Jaccard's `|A n B| / |A u B|`.

    Used for the CV assessment, not for corpus chunks -- see
    CV_SOURCE_PRESENCE_THRESHOLD.
    """

    answer_words = set(_WORD_RE.findall(answer_text.lower())) - _STOPWORDS
    source_words = set(_WORD_RE.findall(source_text.lower())) - _STOPWORDS

    if not answer_words or not source_words:
        return 0.0

    return len(answer_words & source_words) / len(source_words)


# The CV assessment is scored by containment against this threshold,
# while retrieved corpus chunks keep their calibrated Jaccard score at
# DEFAULT_SOURCE_PRESENCE_THRESHOLD. Two metrics, deliberately.
#
# Jaccard divides by the UNION, so it penalises a thorough answer for
# its own length. The CV block is short (~600 chars) and a good CV
# answer is long, so real CV answers scored 0.1049-0.1776 against a 0.12
# threshold -- the worst of them BELOW the bar despite being correct.
# Grounding would have depended on answer verbosity rather than on
# whether the answer actually used what it was given.
#
# Containment asks the question grounding actually cares about ("how
# much of the supplied source did the answer use?") and is insensitive
# to answer length. Measured on real data: the same five answers score
# 0.3659-1.0000, while 780 negative pairs (all 156 corpus chunks x 5 CV
# contexts) peak at 0.1622. 0.25 is the geometric midpoint of that gap
# -- 1.5x above the worst negative, 1.46x below the worst positive.
#
# Corpus chunks keep Jaccard because 0.12 was calibrated against corpus
# text (spec section 13) and Phase 1 behaviour must not shift.
CV_SOURCE_PRESENCE_THRESHOLD = 0.25


SimilarityScorer = Callable[[str, str], float]


def check_source_presence(
    answer_text: str,
    evidence: EvidenceBundle,
    scorer: SimilarityScorer = _lexical_overlap,
    threshold: float = DEFAULT_SOURCE_PRESENCE_THRESHOLD,
    cv_context: "CVAssessmentContext | list[CVAssessmentContext] | None" = None,
    cv_threshold: float = CV_SOURCE_PRESENCE_THRESHOLD,
    *,
    include_diagnosis: bool = True,
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

    cv_block = render_cv_context(cv_context, include_diagnosis=include_diagnosis)

    if not evidence.chunks and not cv_block:
        return False

    # Corpus chunks: Jaccard at the threshold calibrated for corpus text.
    if any(
        scorer(answer_text, chunk.text) >= threshold
        for chunk in evidence.chunks
    ):
        return True

    # CV assessment: containment, which does not penalise a thorough
    # answer for its length. See CV_SOURCE_PRESENCE_THRESHOLD.
    if cv_block:
        return _containment(answer_text, cv_block) >= cv_threshold

    return False


def run_safety_check(
    answer_text: str,
    evidence: EvidenceBundle,
    source_presence_threshold: float = DEFAULT_SOURCE_PRESENCE_THRESHOLD,
    cv_context: "CVAssessmentContext | list[CVAssessmentContext] | None" = None,
    cv_threshold: float = CV_SOURCE_PRESENCE_THRESHOLD,
    *,
    include_diagnosis: bool = True,
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
        cv_threshold=cv_threshold,
        # MUST match what the prompt supplied and the fallback displays,
        # or "grounded" stops meaning anything (render_cv_context docstring).
        include_diagnosis=include_diagnosis,
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
    *,
    include_diagnosis: bool = True,
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

    cv_block = render_cv_context(cv_context, include_diagnosis=include_diagnosis)

    # Evidence that barely matched must NOT be introduced as "the
    # relevant evidence". Retrieval returns its top-k with no minimum
    # score, so for a question the corpus does not cover, those chunks
    # are merely the least-unrelated text available.
    #
    # Measured before this guard existed: "How is impetigo treated?"
    # (top_score 0.2622) returned FDA dosing instructions for basal cell
    # carcinoma chemotherapy, introduced as "Here is the relevant
    # evidence directly". Three individually-correct behaviours composed
    # into that -- retrieval has no score floor, the model correctly said
    # the sources do not cover impetigo, and the grounding check
    # correctly rejected that sentence, because a statement about the
    # ABSENCE of evidence cannot lexically overlap the evidence.
    #
    # The fix belongs here, not in the grounding check: letting an answer
    # skip grounding by claiming to have no evidence would be a loophole
    # any hallucination could use.
    evidence_is_usable = not evidence.is_empty and not evidence.is_low_similarity

    if not evidence_is_usable and not cv_block:
        return (
            "A full explanation isn't available right now, and the medical "
            "sources available to me don't cover this question. Please "
            "consult a healthcare professional."
        )

    parts = ["A full explanation isn't available right now."]

    if cv_block:
        # CV context describes THIS patient and is safety-relevant, so it
        # reaches the user whatever retrieval returned (spec 5.4).
        parts.append(
            "Here is the assessment of your photo directly:\n\n" + cv_block
        )

    if evidence_is_usable:
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
    else:
        parts.append(
            "The medical sources available to me don't cover this question, "
            "so I haven't included them here. Please consult a healthcare "
            "professional."
        )

    return "\n\n".join(parts)
