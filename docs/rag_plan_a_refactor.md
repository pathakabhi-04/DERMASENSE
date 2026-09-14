# RAG under Plan A: narrow mode

**Date:** 2026-09-15
**Implements:** `plan_a_narrow_product_spec.md` §5 on the RAG side.
**Status:** implemented, opt-in, 121 existing RAG tests unchanged and
passing.

## 1. The problem

`CVAssessmentContext.format_for_prompt()` rendered seven lines into the
LLM prompt, three of which Plan A forbids from reaching a user:

```
- Most likely class from the image classifier: NEV     <- forbidden
- Assessment confidence: 78% (calibrated)              <- forbidden
- Risk category: LOW                                   <- forbidden
- Flagged for professional review: no
- Change since the previous photo: ...                 <- the product
- Image quality notes: ...                             <- the product
```

Plan A's rule is stronger than the original §2.4, which only banned the
`LOW` label. A user told *"most likely class: NEV"* has been reassured
whether or not a risk category came with it.

## 2. A conflict with the collaborator's spec, and how it was resolved

This is not a neutral refactor. The RAG spec **§5.4** says, and the code
comments repeat, that CV context is *required* to reach the user
whenever it exists:

> "A user must never see nothing, or an error page, when structured CV
> evidence was successfully computed — that evidence is safety-relevant
> and must reach them even if narration fails."

That was written after a real failure: an LLM timeout on a HIGH-risk,
review-flagged lesion produced a page of general corpus text that never
mentioned the assessment. Plan A appears to ask for exactly the
behaviour that fix removed.

**The resolution is that §5.4's underlying concern survives intact, and
only its referent changes.** §5.4 protects *safety-relevant CV evidence
reaching the user*. Under Plan A the safety-relevant evidence is no
longer the risk category — it is the change-over-time verdict, the
review flag, and the capture-quality notes. Narrow mode withholds the
diagnosis and keeps all of those. Nothing that can warn a user is
dropped; only the claim the product does not make.

**Flagged for the RAG collaborator** rather than silently changed: this
alters the intent of a fix they made deliberately, and they should say
if they read §5.4 differently.

## 3. What changed

`include_diagnosis: bool = True`, keyword-only, threaded through every
consumer:

| function | file |
|---|---|
| `CVAssessmentContext.format_for_prompt` | `cv_context/schema.py` |
| `render_cv_context` | `cv_context/schema.py` |
| `PromptBuilder.build` | `prompts/prompt_builder.py` |
| `check_source_presence` | `safety/grounding_check.py` |
| `run_safety_check` | `safety/grounding_check.py` |
| `build_fallback_answer` | `safety/grounding_check.py` |
| `RagPipeline(narrow_product=...)` | `pipeline.py` |

**The default is `True` everywhere** — Phase 1/2 behaviour is byte-identical
and every existing test passes untouched. Plan A opts in with
`RagPipeline(..., narrow_product=True)`.

## 4. The invariant that made this more than a one-line change

`render_cv_context`'s own docstring states it:

> "If the grounding check scored an answer against different text than
> the prompt supplied or the fallback displayed, 'grounded' would stop
> meaning anything."

So narrow mode cannot be applied at the prompt alone. If the prompt
omitted the risk category but the safety check still scored against
text containing it, an answer could be rejected as ungrounded for
faithfully using the evidence it was given — or worse, pass while
citing something the user never saw. All three consumers take the same
flag from one place (`RagPipeline.narrow_product`), and
`test_grounding_and_prompt_see_identical_text` pins it.

## 5. Why not change the default, or version the contract

**Default stays permissive** because Plan B (dermoscopy) turns the
diagnosis back on with no code change, and because flipping a default
would silently alter behaviour for every existing caller and test. The
cost is that a Plan A deployment must remember the flag — recorded here
and in the Plan A spec as a deployment requirement, and worth a
startup assertion in whatever service wires it.

**No CV-8 contract change.** CV-8 keeps emitting its full v1.2 payload;
only the *consumption* narrows. The contract is correct and well tested,
Plan B needs the full payload, and narrowing at the consumer is
reversible where narrowing the contract is not.

## 6. What this does NOT cover

Narrow mode stops the diagnosis reaching the **LLM prompt, the safety
check and the fallback**. It does not, and cannot, stop a UI from
rendering `native_class` straight off the CV-8 payload. That obligation
is stated in `plan_a_narrow_product_spec.md` §5 and belongs to whoever
builds the client — the RAG layer cannot enforce it.

## 7. Tests

`src/rag/cv_context/test_narrow_product.py` — 6 tests: diagnosis absent
in narrow mode; change/quality/review still present; default unchanged;
propagation through `render_cv_context`; the fallback user-surface
carries no diagnosis; and prompt/grounding text identity.
