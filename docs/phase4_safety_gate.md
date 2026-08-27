# Phase 4 — Safety Gate

## Status

**Phase 4 core implementation: COMPLETE**

Full repository test suite:

```text
14 passed


```markdown
---

## 1. Objective

Phase 4 introduces a downstream product-safety layer between native
classifier output and automatic product-action release.

The safety layer does not attempt to correct the native classifier.
Instead, it prevents a prediction that would produce the lowest-action
product category (`MONITOR`) from being automatically released.

The Phase 4 architecture is:

Image
  ↓
Native classifier
  ↓
Native diagnosis
  ↓
Product action mapping
  ↓
Safety gate
  ↓
AUTO_RELEASE / REVIEW

---

## 2. Investigation Closed Before Phase 4

The preceding native-classification investigation established:

1. SCC errors disproportionately move toward BCC.
2. The phenomenon exists at lesion level.
3. Lesion size is associated with problematic cases.
4. Size alone does not explain the representation margin.
5. Hurt status does not explain the whole effect.
6. SupCon/F1 does not reliably fix the problematic SCC cases.
7. Geometry changes correlate strongly with classifier-logit changes.
8. F1 does not improve the problematic SCC cases overall.
9. SCC recall is seed-sensitive.
10. MEL has the same failure pattern under both models, including the
   same MEL → NEV miss.

This investigation is formally closed.

Further model-specific SCC/BCC investigation was not selected as the
next engineering intervention. The focus therefore moves downstream
to product-level safety.

---

## 3. C1 vs F1 Model Selection

The C1 vs F1 decision was made using the pre-committed minimum
malignant-recall rule.

Malignant classes:

- BCC
- SCC
- MEL

### C1

| Class | Recall |
|---|---:|
| BCC | 71.83% |
| SCC | 28.57% |
| MEL | 66.67% |

Minimum malignant recall: **28.57%**

### F1

| Class | Recall |
|---|---:|
| BCC | 73.24% |
| SCC | 17.86% |
| MEL | 66.67% |

Minimum malignant recall: **17.86%**

Therefore, **C1 was selected over F1**.

Although F1 performs better on several aggregate metrics, the
pre-committed safety rule selects C1 because its worst malignant-class
recall is higher.

The small ECE difference was not treated as meaningful evidence for
model selection.

---

## 4. Product Action Mapping

Native diagnoses remain unchanged.

They are mapped downstream to product-level actions:

| Native diagnosis | Product action |
|---|---|
| BCC | URGENT_EVALUATION |
| SCC | URGENT_EVALUATION |
| MEL | URGENT_EVALUATION |
| ACK | EVALUATE_SOON |
| NEV | MONITOR |
| SEK | MONITOR |

The mapping deliberately sits downstream of the dataset-native target
definitions.

The classifier therefore answers:

> What native diagnosis does this image most resemble?

The product layer separately answers:

> What action does that diagnosis imply?

---

## 5. Safety Consequence Audit

A native misclassification is not automatically equivalent to a
product-safety failure.

For example:

BCC → ACK

is a native diagnostic error, but ACK maps to:

EVALUATE_SOON

and therefore does not produce the lowest-action product outcome.

For C1, the high-risk downgrade rates were:

- BCC: 26 / 142 = 18.3%
- SCC: 5 / 28 = 17.9%
- MEL: 1 / 9 = 11.1%

The raw number of BCC downgrade errors is larger primarily because
BCC has many more test examples.

The evidence therefore does not justify treating BCC as an isolated
safety bottleneck.

The relevant product-level finding is that multiple high-risk native
classes can be downgraded into lower-action categories.

---

## 6. Dangerous Failure Definition

A dangerous product-level failure is defined as:

True diagnosis ∈ {BCC, SCC, MEL}
AND
Predicted product action == MONITOR

In other words:

> A genuinely high-risk lesion is predicted into the lowest-action
> product category.

On the 352-image PAD-UFES test set:

- C1 dangerous high-risk → MONITOR: **7**
- F1 dangerous high-risk → MONITOR: **6**

---

## 7. C1 Dangerous Cases

The seven identified C1 dangerous cases were:

| True class | Predicted class | Confidence |
|---|---|---:|
| BCC | SEK | 0.9503 |
| MEL | NEV | 0.9874 |
| BCC | SEK | 0.9614 |
| BCC | SEK | 0.3812 |
| BCC | SEK | 0.8450 |
| SCC | SEK | 0.8926 |
| SCC | NEV | 0.7277 |

These cases demonstrate that dangerous failures are not restricted
to low-confidence predictions.

---

## 8. Confidence Analysis

Confidence was evaluated as a possible safety signal.

A simple global confidence threshold was not selected as the Phase 4
policy.

Safety-gate simulation showed that:

- low thresholds catch relatively few dangerous failures;
- higher thresholds substantially increase review workload;
- high confidence does not guarantee safety.

The key stress case is:

MEL → NEV

For C1:

- confidence = 0.9874
- predicted high-risk probability = 0.0081

For F1:

- confidence = 0.9766
- predicted high-risk probability = 0.0103

Therefore, confidence alone cannot guarantee protection against
dangerous product-level errors.

It may remain an auxiliary uncertainty signal, but it is not the
primary Phase 4 safety mechanism.

---

## 9. Locked Phase 4 Safety Policy

The locked policy is:

Predicted action == MONITOR
        ↓
      REVIEW

All other known product actions proceed normally.

Unknown or malformed actions fail conservatively to review.

| Product action | Gate decision |
|---|---|
| MONITOR | REVIEW |
| EVALUATE_SOON | AUTO_RELEASE |
| URGENT_EVALUATION | AUTO_RELEASE |
| UNKNOWN | REVIEW |

The safety gate does not alter the classifier prediction.

It only determines whether the resulting product action may be
automatically released.

---

## 10. Implementation

Phase 4 introduces:

- `src/inference/native.py`
- `src/inference/pipeline.py`
- `src/risk/action_mapping.py`
- `src/risk/safety_gate.py`

### Native inference

Provides native model inference using the established C1 checkpoint
and was validated against the existing C1 evaluation path.

### Product action mapping

Provides deterministic native-diagnosis → product-action mapping.

### Safety gate

Implements the locked Phase 4 policy.

### Inference pipeline

Connects:

native inference
    ↓
action mapping
    ↓
safety gate

---

## 11. Validation

The established C1 checkpoint was independently evaluated on the
PAD-UFES test set.

Test set:

- 352 images

C1 results:

| Metric | Result |
|---|---:|
| Accuracy | 0.7017 |
| Macro F1 | 0.6470 |
| SCC recall | 0.2857 |
| MEL recall | 0.6667 |

Native inference reproduction tests confirmed that the new inference
implementation reproduces the established classifier behavior.

Known-dangerous-case tests confirmed that identified high-risk →
MONITOR cases are intercepted by the safety gate.

---

## 12. Test Coverage

Phase 4 validation covers:

- native inference reproduction;
- product action mapping;
- safety-gate behavior;
- end-to-end inference;
- known dangerous-case interception.

Final repository test suite:

14 passed

No training was performed as part of the Phase 4 implementation.

No model checkpoint was modified.

---

## 13. Architectural Boundary

Phase 4 deliberately does not:

- retrain the classifier;
- modify model weights;
- modify native class definitions;
- alter classifier logits;
- replace native diagnoses with risk categories;
- rely on a global confidence threshold;
- claim clinical validation.

The architectural separation is:

MODEL
"What native diagnosis does the image resemble?"
        ↓
PRODUCT
"What action does that diagnosis imply?"
        ↓
SAFETY
"Can that action be automatically released?"

Phase 4 addresses the third question.

---

## 14. Current Limitation

The current safety gate is an engineering guardrail:

MONITOR → REVIEW

It is not a clinically validated decision rule.

The current results demonstrate interception of the identified
dangerous product-level failure mode on the evaluated PAD-UFES test set.

They do not establish:

- clinical effectiveness;
- clinical sensitivity;
- prospective performance;
- real-world clinician workload;
- deployment safety;
- clinical readiness.

Further validation on appropriately designed datasets will be
required before real-world clinical use.

---

## 15. Completion Criteria

- [x] Native inference wrapper implemented.
- [x] Native prediction reproduction verified.
- [x] Native diagnosis → product action mapping implemented.
- [x] Safety gate implemented.
- [x] `MONITOR` predictions routed to `REVIEW`.
- [x] Unknown actions fail safely.
- [x] End-to-end inference pipeline implemented.
- [x] Known dangerous cases intercepted.
- [x] Full repository test suite passes.

---

## 16. Conclusion

Phase 4 establishes the first explicit product-level safety boundary
in the DermaSense CV pipeline.

The native classifier remains responsible for native diagnosis.

The downstream risk layer converts that diagnosis into an action
category.

The safety gate prevents the lowest-action outcome from being
automatically released.

The locked policy is therefore:

Native diagnosis
      ↓
Product action
      ↓
MONITOR?
   ↙       ↘
 YES        NO
  ↓          ↓
REVIEW   AUTO_RELEASE

**Phase 4 core implementation is complete.**

The model-specific SCC/BCC investigation remains closed. Future work
should focus on the next independent component of the DermaSense CV
pipeline rather than reopening the completed Phase 4 investigation
without new evidence.