# RAG safety layer — two findings from integrating CV context

**Date:** 2026-09-11
**From:** CV side
**About:** `src/rag/safety/grounding_check.py`
**Status:** One fixed (with your architecture's own justification), one left for you to decide

We've integrated CV-8 output into your Phase 1 baseline locally and run
real answers through it. Two things in the safety layer need your call,
because both are changes to *safety semantics* and that's your decision,
not ours. Everything below is measured on your code, your index, and
real Groq answers — no estimates.

---

## 1. Grounding rejected CV-grounded answers — we fixed this, here's why we felt able to

### What we measured

`check_source_presence` scores lexical Jaccard overlap between the
answer and each retrieved corpus chunk, threshold 0.12. But a Phase 2
answer's most valuable content is about *this patient's lesion* — risk
category, calibrated confidence, temporal verdict. That isn't corpus
content, so it can't resemble corpus content.

Across five real CV-grounded answers (one per delivered fixture),
overlap against the retrieved chunks was:

| Answer | best corpus overlap | grounded at 0.12? |
|---|---|---|
| 1 | 0.0920 | no |
| 2 | 0.0882 | no |
| 3 | 0.0829 | no |
| 4 | 0.1005 | no |
| 5 | 0.0617 | no |

**0 of 5.** Not marginal — a CV narration scored **0.0000** against the
three chunks actually retrieved for a matching query, and 0.0412
against the single most favourable chunk in the entire 156-chunk
corpus.

The failure mode is perverse: the better an answer explains the
patient's own assessment, the more certainly it is thrown away. And the
LLM's only route to passing is to pad the answer with corpus
boilerplate — the exact opposite of what constrained paraphrase is for.

### What we changed

CV context now counts as a grounding source alongside retrieved
evidence. We were comfortable making this one because **your own §4.3
already says it is one**: CV context is supplied evidence "on exactly
the same footing as retrieved medical passages." The safety layer just
hadn't been told. Measuring grounding only against corpus text
contradicts the architecture the check exists to enforce.

After: **4/5 ground.** The check did not get weaker — off-topic text
still scores 0.0000 against CV context, and a banned-phrase claim still
fails regardless of grounding. Both are regression-tested.

### The part we want your opinion on: the metric, not the threshold

The four passing scores are **0.1203, 0.1269, 0.1489, 0.1210** against
a 0.12 threshold. One of them clears it by 0.3%. Case 4 ("why couldn't
you compare it with my previous photo?") failed at 0.0730.

That thinness is structural, not bad luck. Jaccard divides by the
**union**, so a thorough 469-word answer measured against a ~600-char
CV block is penalised *for being thorough* — the answer's own vocabulary
inflates the denominator.

**Suggestion: use containment rather than Jaccard for the CV source**
— `|A ∩ B| / |B|`, i.e. "what fraction of the supplied source actually
appears in the answer." That asks the question grounding actually cares
about ("did the answer use what it was given?") and is insensitive to
answer length. We have NOT made this change: swapping the similarity
metric is a real semantic decision about your safety layer, and it
would need its own threshold calibrated against real data the way you
calibrated 0.12 (your §13).

If you'd rather keep Jaccard, raising the CV block's information
density would also help, but it's a weaker fix.

---

## 2. The banned-phrase check false-positives on CV narration — we did NOT change this

### What we measured

The check fired on **1 of 5** otherwise-excellent answers. Both triggers
in that answer are false positives:

```
"...this is simply the label the system uses to keep track of the mole"
    certainty phrase "this is"  +  condition name "mole"

"...if you have it, the earlier photo that showed the mole"
    certainty phrase "you have"  +  condition name "mole"
```

Neither is a diagnostic claim. One explains what a lesion ID is; the
other asks the patient to bring a photo.

### A hypothesis we tested and discarded

We suspected the sentence splitter: `(?<=[.!?])\s+` doesn't break on
newlines, so a markdown bulleted list could collapse into one giant
"sentence" and make the same-sentence co-occurrence rule meaningless.

**We tested it. It's not the cause.** A newline-aware splitter flags the
identical two cases, because both genuinely sit within one sentence.
Mentioning it so you don't spend time on the same idea.

The real cause is that the check is substring co-occurrence with no
grammar, and **CV narration systematically inflates its false-positive
rate**: CV answers constantly say "this is [the label / the risk
category / the identifier]" while mentioning mole/nevus throughout.
Phase 1 corpus answers rarely produce that collocation; Phase 2 answers
produce it constantly.

### Why we left it alone

Your §5 deliberately chose over-flagging, and said to narrow it "later
only if real failures justify it, not preemptively." That's the right
call and we're not going to quietly reverse it — a false positive costs
a fallback, and a false negative costs a wrong diagnosis reaching a
patient. But you now have real failures, so the "later" condition is
met, and this is worth a decision rather than drift.

**Options, in the order we'd rank them:**

1. **Require the certainty phrase and condition name to be
   syntactically related**, not merely co-present — e.g. the condition
   name falls within a few tokens after the certainty phrase. Kills
   "if you have it, the earlier photo that showed the mole" while
   keeping "you have a basal cell carcinoma". Cheap, still
   deterministic, no model call.
2. **Exclude a small set of clearly non-diagnostic collocations**
   ("this is simply", "this is the label", "if you have"). Cheapest,
   but a blocklist of a blocklist — it will need maintenance.
3. **Leave it.** Defensible: it fails safe, and the fallback now
   carries the CV assessment (see our other note), so a false positive
   costs narration quality rather than the safety signal. Worse UX,
   never worse safety.

We'd lean 1. But it's your check and your risk call.

### Decisive evidence: 25% of your own corpus fails this check

After writing the above we ran the check over the indexed corpus
itself. **39 of 156 chunks (25%) contain a sentence that trips the
banned-phrase rule** — authoritative AAD / NCI / MedlinePlus text, the
very evidence the system exists to ground answers in:

```
[AAD_BASAL_CELL_CARCINOMA_001]  "you have" + "basal cell carcinoma"
   "A dermatologist can tell you if you have basal cell carcinoma and
    if you do, what treatment is recommended."

[NCI_MOLES_MELANOMA_001]        "diagnosed with" + "melanoma"
   "in 2017-2018, the lifetime risk of being diagnosed with melanoma
    was 2.9% (1 in 34) for White people but 0.1% (1 in 1,000)..."

[AAD_ACTINIC_KERATOSIS_SYMPTOMS_001]  "you have" + "skin cancer"
   "Should that change be an AK, you have a greater risk of developing
    skin cancer."

[AAD_MOLE_PROBLEM_001]          "you have" + "mole"
   "If you have a raised mole on skin that you shave, you may nick the
    mole, causing it to bleed."
```

None of these is a diagnostic claim. Two are conditionals, one is an
epidemiological statistic, one is advice about shaving. The rule cannot
distinguish "a dermatologist can tell you **if** you have X" from "you
have X".

Three consequences worth weighing:

1. **A faithful constrained paraphrase inherits the phrasing.** An LLM
   asked to explain a chunk that says "a dermatologist can tell you if
   you have basal cell carcinoma" will quite reasonably reuse that
   construction — and be rejected for accurately paraphrasing the
   evidence it was given. The check penalises the behaviour the
   architecture asks for.

2. **The fallback can contain what the check rejected.** The fallback
   renders retrieved evidence verbatim, so an answer rejected for a
   banned phrase may be replaced by corpus text carrying the same
   phrasing. Whatever the rule is protecting against, this path isn't
   protected.

3. **This is the real source of the temperature variance in your
   §0.1.** Not sampling noise in the abstract — the model is drawing on
   source text that is 25% "unsafe" by this rule, so whether a given
   generation trips it is close to a coin flip on phrasing.

This moves our recommendation from "lean option 1" to **option 1 is the
one worth doing**: requiring the condition name to fall within a few
tokens *after* the certainty phrase would clear every example above
while still catching "you have a basal cell carcinoma". It stays
deterministic and needs no model call.

We have measured this but not changed it. Still your call.

### One related note

We suspect this is also the real story behind your §0.1 decision to
drop Groq's temperature to 0.1 "because the same query could pass or
fail the banned-phrase check across identical calls." That variance is
a symptom of the check's sensitivity to incidental phrasing. Lowering
temperature reduces the variance without addressing why the check is
that sensitive. Worth revisiting once (1) or (2) lands — you may be able
to raise temperature again and get less stilted answers.

---

## What we'd like back

1. **A yes/no on containment vs Jaccard** for the CV grounding source.
   We'll implement whichever you pick; we just won't swap your metric
   unilaterally.
2. **A pick from options 1/2/3** on the banned-phrase check.
3. Nothing else — Phase 1's gate, `top_score`, and the corpus are all
   yours and we're not touching them.

Everything we've changed is in `src/rag/`, documented in
`docs/rag/CHANGES.md`, with the reasoning and measurements in the
commit messages.
