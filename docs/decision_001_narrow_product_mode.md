# Decision 001: production runs `narrow` product mode

**Date:** 2026-09-16
**Status:** ACCEPTED — this is the shipped configuration
**Supersedes:** nothing. First decision record in this project.

## Decision

> **Every deployment serving members of the public runs
> `DERMASENSE_PRODUCT_MODE=narrow`, which is also the default when the
> variable is unset. `full` is permitted only for internal evaluation
> and for a Plan B (clinician/dermoscopy) deployment, and only with a
> named owner who understands it emits diagnoses.**

## Why this needed writing down rather than just defaulting

The default is already `narrow`, and `src/serving/product_mode.py`
raises on an unrecognised value. So why a decision record?

Because the failure mode is silent and asymmetric. Setting
`DERMASENSE_PRODUCT_MODE=full` is one environment variable, it produces
no error, no warning and no visible difference to an operator — the
response simply gains four fields — and the product starts telling
worried members of the public a class name and a risk category it cannot
support. Nothing in the code can distinguish "an engineer deliberately
enabled Plan B" from "someone copied a `.env` from the evaluation
harness". Only a written decision can.

## Context

`external_6class_result.md` measured the shipped pipeline on 792
external clinical photographs: **57.9% of melanomas reach a clinician**,
and 34.6% are assessed and sent to `MONITOR`. Four investigations
(`data_constraint_spec.md`) established the gap is a data constraint,
not one more modelling problem.

A product that names a class at that accuracy is not a product with a
caveat; it is a product that sometimes tells someone with melanoma they
have a mole. Plan A's rule exists for that reason:

> The narrow product never emits reassurance. Its worst output is
> "we can't tell — see a clinician". It never says low risk, benign, or
> probably fine, and it never names a diagnosis.

## What `narrow` actually withholds

Withheld from the client: `diagnosis` (`native_class`, `probabilities`),
`risk_category`, `risk_reason`, and `uncertainty` as a block.

Returned: `lesion_id`, `temporal` (the change verdict), `quality_flags`,
`contract_version`, and `requires_review` **only when true** — `false`
is reassurance by implication.

The mechanism is an allow-list, so any field CV-8 gains later is
withheld until someone decides it is safe to show.

## Consequences

**Accepted costs.** The first visit has no lesion-specific output
(`plan_a_narrow_product_spec.md` §8b). One attempt to close that gap
failed (`abcd_features_result.md`). Users who want a diagnosis will not
get one, and some will be disappointed. That is the intended behaviour,
not a defect to fix later.

**What this does not constrain.** CV-8's contract is unchanged and still
emits v1.2 in full. The full assessment is still computed and logged
server-side, because those predictions beside a future biopsy result are
the evaluation Plan C exists to make possible. Narrowing is a
presentation boundary at the API edge, not a contract change — the RAG
layer sits *inside* that boundary and consumes the full object, doing
its own narrowing of what it narrates.

**Reversal.** Setting `full` is the reversal, and it is one variable. It
requires a new decision record superseding this one, naming the person
accountable and the evidence that changed. The bar is evidence, not
convenience: specifically, a melanoma-routing measurement **on the
deployment capture path** (`data_constraint_spec.md` §4), not another
in-domain number.

## Enforcement, and its limit

Enforced in code: default `narrow`; unrecognised values raise at
startup; allow-list narrowing; `tests/test_product_mode.py` and
`tests/test_assess_api.py::NarrowModeApiTests` assert a real `/assess`
response contains none of the six class labels.

**Not enforceable in code:** a client that invents a verdict from
`requires_review`, or a screen headed "Your result". The API can only
withhold data; it cannot stop a UI implying a conclusion from what it
does receive. That remains a human review against §5, and it is listed
as blocking in `plan_a_ship_checklist.md`.
