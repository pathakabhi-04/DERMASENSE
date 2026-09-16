# Plan A: ship checklist

**Date:** 2026-09-16
**Spec:** `plan_a_narrow_product_spec.md`

What is done, what is not, and what must not be skipped. Items marked
**BLOCKING** would make the shipped product dishonest or would lose data
that cannot be recovered later.

## Done

- [x] **API withholds diagnosis and risk by default.**
      `DERMASENSE_PRODUCT_MODE` defaults to `narrow`; `full` must be set
      explicitly. Unrecognised values raise at startup rather than per
      request. Allow-list, so a field added to CV-8 later is withheld
      until someone decides it is safe.
- [x] **End-to-end proof, in the default configuration.**
      `tests/test_assess_api.py::NarrowModeApiTests` asserts a real
      `/assess` response contains no `native_class`, `risk_category`,
      `risk_reason`, `probabilities`, and none of the six class labels.
- [x] **`requires_review` is escalation-only** — emitted when True,
      omitted when False, because `false` is reassurance by implication.
- [x] **RAG narrow mode** — `RagPipeline(narrow_product=True)` withholds
      the same fields from the prompt, the grounding check and the
      fallback together.
- [x] **Full assessment still logged server-side** for Plan C.
- [x] **Contract unchanged.** CV-8 still emits v1.2 in full; the
      narrowing is a presentation boundary at the API edge.

## Blocking, not yet done

- [ ] **BLOCKING — the client must not render what it is not given.**
      Narrow mode stops the fields leaving the server. It cannot stop a
      UI inventing a verdict from `requires_review`, or a screen headed
      "Your result". Copy review against §5 of the spec is a human task
      and nobody has done it.
- [ ] **BLOCKING — consent and the Plan C linkage identifier.**
      Patient ID, retention consent covering model training, and a
      linkage code that survives to a biopsy result.
      **Cannot be retrofitted to photos already captured**, so it must be
      in the first release or Plan C starts from zero later.
- [ ] **BLOCKING — a decision on `narrow` vs `full` in the deployment
      environment**, written down, with whoever owns the deploy aware
      that `full` emits diagnoses.

## Required before real users, not blocking a pilot

- [ ] Body-site tap at capture (Plan C metadata, cheap now).
- [ ] Self-reported Fitzpatrick type, optional, once.
- [ ] 3-month reminder scheduling. **CV-7 is the headline feature and
      never fires without a second visit** — without reminders the
      product has no core.
- [ ] Prediction logging schema that Plan C can join against.
- [ ] Regulatory opinion on whether a measure-and-compare product with
      no clinical assertion sits outside SaMD. Flagged in the spec as
      needing an opinion, not an engineering assumption.

## Known gaps, accepted knowingly

- **The first visit has no lesion-specific output.** CV-7 cannot fire
  without a prior. One attempt to close this (`abcd_features_result.md`)
  failed, and the geometric features ran backwards. Mitigations are the
  clinician handoff, self-check education, and telling a worried user to
  see someone now.
- **CV-2's phone-domain performance is unmeasured.** It is trained on
  TBP-rig imagery. Its misses are 75% benign on clinical photographs
  (`cv1_5_routing_resolution.md`), but phone photos are a third domain.
- **CV-7's ruler calibration has ~4% confident coverage.** The headline
  feature works in millimetres only when a ruler is in frame; otherwise
  it compares in pixels, which is valid within a lesion's own history
  but not across users.

## The rule everything else serves

> The narrow product never emits reassurance. Its worst output is
> "we can't tell — see a clinician". It never says low risk, benign, or
> probably fine, and it never names a diagnosis.

If a change makes that harder to guarantee, it is not a Plan A change.
