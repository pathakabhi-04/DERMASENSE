**# CV-2 Detection Specification**

**## DermaSense Computer Vision Pipeline**

**\*\*Status:\*\*** Accepted specification  

**\*\*Component:\*\*** CV-2 — Lesion Candidate Detection  

**\*\*Primary dataset:\*\*** iToBoS 2024 — Skin Lesion Detection with 3D-TBP Images  

**\*\*Downstream component:\*\*** CV-3 — Lesion Segmentation

**---**

**## 1. Purpose**

CV-2 is the **\*\*lesion candidate localization\*\*** component of DermaSense.

Its purpose is to answer:

\> **\*\*"Is there one or more lesion candidates in this image, and approximately where are they?"\*\***

CV-2 is intentionally distinct from CV-3.

CV-2 provides **\*\*coarse localization\*\***.

CV-3 provides **\*\*precise lesion boundary segmentation\*\***.

The two components therefore solve different problems and should remain independently evaluable.

**---**

**## 2. Why CV-2 Exists**

The initial DermaSense architecture was:

\`\`\`text

Image

  ↓

CV-1 Quality

  ↓

CV-2 Detection

  ↓

CV-3 Segmentation

  ↓

CV-4 Classification

  ↓

CV-5 Explainability

  ↓

CV-6 Uncertainty

  ↓

CV-7 Temporal

  ↓

CV-8 Severity

  ↓

Structured clinical/risk context

\`\`\`

The original architecture assumed that an incoming image may contain:

\- one or more lesions;

\- lesions at arbitrary locations;

\- substantial surrounding skin;

\- irrelevant background;

\- no lesion at all;

\- non-lesion objects or visual structures.

This assumption is important for the eventual DermaSense product because real-world images may not resemble the tightly framed, single-lesion dermoscopic images used by CV-3's initial training dataset.

**### 2.1 Why CV-3 alone is insufficient for wide-field inputs**

The initial CV-3 baseline was trained and evaluated on ISIC 2018 Task 1 segmentation data.

That dataset is appropriate for learning lesion boundaries, but its task formulation does not require the model to answer:

\> "Is there a lesion anywhere in this image?"

Instead, the model is presented with a lesion-centric image and is trained to produce a lesion mask.

If that model were applied directly to arbitrary wide-field images, it could produce a segmentation-like output even when:

\- no lesion is present;

\- the lesion is small;

\- the lesion is off-center;

\- multiple lesions are present;

\- irrelevant background objects are visible.

This creates an important failure mode: a segmentation model may be forced to explain an image even when the correct product-level answer is **\*\*"no lesion candidate found."\*\***

CV-2 therefore provides an explicit candidate-localization stage before CV-3 for the wide-field processing path.

**---**

**## 3. CV-2 / CV-3 Division of Responsibility**

The components have deliberately different responsibilities.

**### CV-2 — Detection**

Answers:

\> **\*\*Is there a lesion candidate, and approximately where is it?\*\***

Produces:

\- zero or more candidate bounding boxes;

\- a confidence score for each candidate;

\- candidate coordinates;

\- an explicit zero-candidate outcome.

**### CV-3 — Segmentation**

Answers:

\> **\*\*What is the precise boundary of this candidate lesion?\*\***

Produces:

\- a pixel-level lesion mask;

\- segmentation confidence / quality information;

\- derived geometric information such as bounding box and lesion area.

The resulting relationship is:

\`\`\`text

                 Wide-field image

                       │

                       ▼

                    CV-2

              Candidate Detection

                       │

            ┌──────────┴──────────┐

            │                     │

       No candidates          Candidates

            │                     │

            ▼                     ▼

       Stop analysis          Crop / frame

                                  │

                                  ▼

                                CV-3

                             Segmentation

\`\`\`

A zero-candidate CV-2 result is therefore a **\*\*valid first-class output\*\***, not an error condition that CV-3 must recover from.

**---**

**## 4. Task Definition**

CV-2 is defined as:

\> **\*\*Lesion candidate localization in non-lesion-centric images.\*\***

The task is deliberately narrower than clinical diagnosis.

CV-2 does **\*\*not\*\*** determine:

\- benign vs. suspicious vs. malignant status;

\- lesion subtype;

\- histopathological diagnosis;

\- severity;

\- temporal change;

\- clinical urgency.

Those responsibilities belong to downstream components.

CV-2 is also not intended to replace CV-3 segmentation.

Its purpose is to establish whether candidate lesions exist and where they are located sufficiently well for downstream processing.

**---**

**## 5. Primary Dataset: iToBoS 2024**

The primary training dataset selected for CV-2 is:

**\*\*iToBoS 2024 — Skin Lesion Detection with 3D-TBP Images\*\***

The dataset is appropriate because it directly represents the detection problem that CV-3's ISIC 2018 Task 1 training data does not provide.

The iToBoS training set contains:

\- **\*\*8,473 images\*\***;

\- **\*\*29,403 lesion bounding boxes\*\***;

\- **\*\*1,750 images with zero annotated lesions\*\***;

\- **\*\*6,723 images with at least one lesion\*\***;

\- one detection class: \`lesion\`.

The dataset therefore provides both:

1\. positive localization examples; and

2\. genuine negative examples where the correct output is zero candidates.

This is particularly important for DermaSense because zero-candidate behavior is part of the product contract.

**---**

**## 6. Multi-Lesion Characteristics**

The iToBoS dataset is not restricted to one lesion per image.

The training annotation audit produced the following lesion-density distribution:

\| Lesion count | Images | Percentage |

\|---|---:|---:|

\| 0 | 1,750 | 20.65% |

\| 1–3 | 4,226 | 49.88% |

\| 4–9 | 1,761 | 20.78% |

\| 10+ | 736 | 8.69% |

The dataset contains images with substantially more than one lesion, including images with dozens of annotated lesions.

The observed maximum in the training annotations was:

**\*\*72 lesions in a single image.\*\***

This is important because CV-2 is intended to learn genuine candidate localization rather than simply rediscovering the single-lesion framing assumption of the CV-3 training data.

**---**

**## 7. Annotation Format**

The iToBoS training distribution provides standard object-detection annotations.

The dataset contains:

\- YOLO-format label files;

\- COCO-format \`labels.json\`;

\- a single detection category: \`lesion\`.

The COCO annotation audit confirmed:

\`\`\`text

categories:

    id: 0

    name: lesion

\`\`\`

The first inspected annotation follows the standard bounding-box representation:

\`\`\`text

bbox = [x, y, width, height]

\`\`\`

The CV-2 implementation should therefore treat the task as **\*\*single-class lesion detection\*\***.

The category represents:

\> **\*\*lesion candidate\*\***

It must not be interpreted as a clinical lesion subtype.

**---**

**## 8. Annotation Integrity Audit**

The iToBoS training annotation audit confirmed:

\`\`\`text

Images:                    8473

Labels:                    8473

Images without labels:       0

Labels without images:       0

Total boxes:             29403

Malformed rows:              0

Invalid normalized coords:   0

Classes:                    0 only

\`\`\`

The training annotations therefore provide a clean starting point for detector implementation.

**---**

**## 9. Dataset Metadata**

The training metadata contains:

\- \`image\_id\`;

\- \`age\_at\_baseline\`;

\- \`body\_part\`;

\- \`sun\_damage\_level\`;

\- \`pixel\_spacing\`.

The dataset also provides imaging metadata that can be used for auditing and stratified evaluation.

The \`age\_at\_baseline\` field contains a literal \`"Unknown"\` value in some records.

The training set contains:

\- 108 images with unknown age.

The test set contains:

\- 129 images with unknown age.

These are not represented as numerical missing values and must therefore be handled explicitly during analysis.

**---**

**## 10. Official Train / Test Distribution Audit**

The official iToBoS train/test distributions were audited before model development.

**### 10.1 Age**

\`\`\`text

TRAIN | numeric=8365 | unknown=108

       \| min=30 | median=52.0 | mean=55.2 | max=75

TEST  | numeric=8352 | unknown=129

       \| min=23 | median=52.0 | mean=53.7 | max=76

\`\`\`

The most notable difference is the age composition.

The test set contains more representation below age 30 and above age 70 than the training set.

This is not treated as a reason to reject the dataset.

Instead, it is recorded as an evaluation consideration.

Age-stratified performance should therefore be considered when interpreting detector generalization.

**### 10.2 Body part**

The body-part distributions are broadly similar between train and test.

The largest observed absolute differences were small relative to the overall dataset composition.

Body part should nevertheless remain available for stratified analysis.

**### 10.3 Sun damage**

The distributions are highly similar:

\`\`\`text

Level 1:

    Train = 81.36%

    Test  = 81.70%

Level 2:

    Train = 17.12%

    Test  = 17.09%

Level 3:

    Train =  1.51%

    Test  =  1.21%

\`\`\`

This is reassuring for this metadata dimension.

**### 10.4 Pixel spacing**

Observed pixel-spacing statistics:

\`\`\`text

TRAIN:

    min    = 0.0647

    median = 0.1175

    mean   = 0.1194

    max    = 0.5190

TEST:

    min    = 0.0632

    median = 0.1142

    mean   = 0.1156

    max    = 0.3330

\`\`\`

Pixel spacing should be considered when evaluating whether detector performance is sensitive to lesion scale.

**---**

**## 11. Internal Validation Split**

The official iToBoS test set must remain untouched for final evaluation.

An internal validation set should be created from the official training data.

The split should preserve lesion-density composition using the following buckets:

\`\`\`text

0

1–3

4–9

10+

\`\`\`

The purpose is to prevent the validation set from accidentally becoming dominated by either:

\- zero-lesion images; or

\- dense multi-lesion images.

**### 11.1 Post-split balance verification**

Lesion-density stratification does not guarantee that other important variables remain balanced.

After creating the internal train/validation split, the distributions of at least the following variables must be compared between the two subsets:

\- age bucket;

\- \`"Unknown"\` age;

\- body part;

\- sun-damage level;

\- lesion-density bucket.

The purpose is not to require perfect matching.

The purpose is to detect accidental composition shifts introduced by the split.

If a material imbalance is found, the split should be regenerated using a multi-variable stratification strategy rather than accepting the imbalance silently.

The official iToBoS test set remains untouched throughout this process.

**---**

**## 12. Provenance and Dataset Overlap**

iToBoS belongs to the broader ISIC/UQ dermatology research ecosystem.

The dataset also uses 3D total-body photography (3D-TBP), creating a potential provenance relationship with other DermaSense datasets, including:

\- UQ Longitudinal;

\- SLICE-3D.

Participant-level identifiers sufficient to establish direct overlap are not exposed in the downloaded iToBoS distribution.

Therefore, a definitive participant-level overlap audit cannot currently be performed from the available data.

This is recorded as an **\*\*accepted unresolved provenance limitation\*\***, rather than an unresolved implementation task.

**### 12.1 Operational independence guardrail**

Until participant-level overlap can be established, iToBoS and UQ Longitudinal / SLICE-3D must **\*\*not be treated as jointly independent datasets for the purpose of a single reported evaluation claim\*\***.

In particular, an evaluation must not combine them in a way that presents the resulting metric as independent cross-dataset validation.

Where both datasets contribute to an experiment, their roles and provenance must be explicitly reported rather than assuming independence.

Literal or near-duplicate image hashing could detect some duplicate imagery but would not establish participant-level independence and is therefore not considered a substitute for participant identifiers.

**---**

**## 13. Dataset License**

The downloaded iToBoS annotation metadata identifies the dataset license as:

**\*\*Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0).\*\***

This differs from an earlier secondary-source interpretation of the dataset license and the dataset's own metadata is treated as authoritative for this project record.

The license permits commercial use but includes ShareAlike requirements.

The precise application of CC BY-SA obligations to:

\- trained model weights;

\- derived machine-learning artifacts;

\- the broader DermaSense software/product;

is a legal interpretation question and is not resolved by this technical specification.

Therefore:

\> **\*\*License compatibility must be reviewed before any production or commercial distribution of a CV-2 model trained on iToBoS.\*\***

For research and development purposes, the dataset is retained as the accepted CV-2 development dataset subject to that licensing review.

**---**

**## 14. Why iToBoS Was Selected**

iToBoS was selected because it directly addresses the missing capability identified in the original DermaSense architecture.

CV-3's ISIC 2018 Task 1 dataset is well suited to:

\`\`\`text

single lesion

     ↓

precise mask

\`\`\`

iToBoS instead provides:

\`\`\`text

wide-field image

     ↓

zero / one / many lesions

     ↓

candidate bounding boxes

\`\`\`

This makes iToBoS substantially better aligned with CV-2's defined responsibility.

The choice avoids introducing an unvetted collection of scraped smartphone images merely to manufacture a detection dataset.

Dataset provenance and annotation quality are considered more important than acquiring superficially similar but poorly controlled data.

**---**

**## 15. Domain Limitation**

iToBoS is a clinical / research 3D-TBP imaging dataset.

It is **\*\*not equivalent to ordinary consumer smartphone photography\*\***.

Therefore, training CV-2 on iToBoS does not establish that CV-2 will perform correctly on:

\- arbitrary smartphone photographs;

\- poorly framed photographs;

\- clothing/background-heavy images;

\- unusual lighting;

\- motion blur;

\- non-clinical camera systems.

iToBoS solves the **\*\*wide-field multi-lesion localization problem\*\***.

It does not, by itself, solve the complete smartphone-domain generalization problem.

This distinction must remain explicit in evaluation and product claims.

**---**

**## 16. Zero-Candidate Behavior**

Zero-candidate output is a first-class CV-2 result.

The detector must be capable of returning:

\`\`\`text

No lesion candidates detected

\`\`\`

when no candidate exceeds the defined acceptance criteria.

This output should cause the downstream wide-field pipeline to short-circuit:

\`\`\`text

CV-2

  │

  ├── no candidates

  │       ↓

  │   stop analysis

  │

  └── candidates

          ↓

        CV-3

\`\`\`

The system must not silently force a candidate through CV-3 when CV-2 has determined that no candidate is sufficiently supported.

This behavior is considered more important to the product than maximizing a leaderboard metric at the expense of excessive false positives.

**---**

**## 17. CV-2 Output Contract**

For each input image, CV-2 must produce a structured result containing:

\`\`\`text

candidate\_count

candidates[]

\`\`\`

Each candidate should contain at minimum:

\`\`\`text

x1

y1

x2

y2

confidence

\`\`\`

The result must support:

**### Zero candidates**

\`\`\`text

candidate\_count = 0

\`\`\`

**### One candidate**

\`\`\`text

candidate\_count = 1

\`\`\`

**### Multiple candidates**

\`\`\`text

candidate\_count > 1

\`\`\`

The downstream system must not infer candidate existence from segmentation output.

**---**

**## 18. Detection Threshold**

The detector's confidence threshold is an implementation parameter and must be selected using validation data.

The threshold should not be selected solely to maximize aggregate mAP.

Threshold selection must consider:

\- candidate recall;

\- false-positive rate;

\- zero-candidate behavior;

\- downstream CV-3 burden;

\- candidate multiplicity;

\- dense-image behavior.

The final threshold must be documented with the corresponding validation evidence.

**---**

**## 19. Evaluation Philosophy**

CV-2 will be evaluated using standard object-detection metrics while maintaining product-specific metrics.

Primary evaluation should include:

\- precision;

\- recall;

\- mAP\@0.5;

\- mAP\@0.5:0.95.

However, aggregate mAP is **\*\*not sufficient\*\*** to establish that CV-2 is suitable for DermaSense.

The evaluation must additionally examine:

\- zero-lesion detection performance;

\- false positives on zero-lesion images;

\- lesion recall;

\- performance by lesion-density bucket;

\- performance by relevant metadata strata;

\- candidate count behavior.

**---**

**## 20. Lesion-Density Stratified Evaluation**

The following evaluation strata should be reported:

\`\`\`text

0 lesions

1–3 lesions

4–9 lesions

10+ lesions

\`\`\`

This is necessary because an image containing 72 lesions represents a materially different detection problem from an image containing one lesion.

A detector can obtain a strong aggregate score while still behaving poorly on dense images.

Conversely, a model optimized for dense scenes could potentially perform unnecessarily poorly on sparse images.

Therefore, aggregate and stratified metrics should be considered together.

The purpose is **\*\*diagnostic understanding\*\***, not to create an additional optimization rabbit hole.

**---**

**## 21. Product-Oriented Priority**

The CV-2 development hierarchy is:

\`\`\`text

1\. Correct zero-candidate behavior

2\. High lesion candidate recall

3\. Controlled false positives

4\. Reliable candidate localization

5\. Robust behavior across lesion densities

6\. Standard detection metrics

7\. Further optimization

\`\`\`

This ordering reflects the actual DermaSense product requirement.

A detector that achieves a marginally better mAP while frequently hallucinating candidates on lesion-free images is not considered a product improvement.

Likewise, CV-2 should not be subjected to endless architecture or hyperparameter experimentation once a credible baseline has been established.

The goal is a **\*\*reliable downstream interface\*\***, not leaderboard optimization for its own sake.

**---**

**## 22. Pre-Committed Acceptance Thresholds**

Section 21 orders CV-2's priorities but does not, by itself, define a pass/fail bar. Consistent with the decision discipline used elsewhere in DermaSense, numeric acceptance thresholds must be fixed before baseline training begins, not derived after seeing results.

The following thresholds are now pre-committed as the initial CV-2 development acceptance targets:

```text
Candidate recall on real lesions (validation):        ≥ 95%
False-positive rate on zero-lesion images:             ≤ 5%
Recall on the 10+ lesion-density bucket:              ≥ 90%
```

Metric definitions:

- Candidate recall is the proportion of ground-truth lesions
  matched by a detector prediction at IoU ≥ 0.50.

- False-positive rate on zero-lesion images is the proportion
  of images containing zero ground-truth lesions for which the
  detector produces at least one accepted prediction after
  confidence-threshold and NMS processing.

- 10+ lesion recall is calculated only over images containing
  ten or more ground-truth lesions.

These thresholds are development acceptance gates for the CV-2
component. They are not clinical safety thresholds and do not,
by themselves, establish clinical validity.

The thresholds may be revised only with an explicit, documented
rationale recorded before the revised threshold is applied.
They must not be silently loosened to match observed baseline
results.

A baseline that fails one or more thresholds is not automatically
discarded. The failure must instead be explicitly recorded, with
a decision to either:

1. accept the limitation with justification;
2. revise the dataset, split, or training procedure;
3. investigate a concrete model failure mode; or
4. formally revise the threshold with documented rationale.

The baseline must not be declared successful merely because it
achieves a strong aggregate mAP while failing the pre-committed
product-oriented acceptance criteria.

**Status: Accepted — implementation may proceed.**

**Pre-committed CV-2 acceptance thresholds:**
- ≥95% lesion recall
- ≤5% false-positive rate on zero-lesion images
- ≥90% recall on the 10+ lesion-density bucket

These thresholds must remain fixed during the initial baseline
experiment and must not be selected or modified based on the
observed baseline results.

**---**

**## 23. Detector Architecture**

The initial CV-2 detector should use a conventional YOLO-family object-detection architecture.

The exact model variant is an implementation decision rather than a specification-level requirement.

The first implementation should prioritize:

\- reproducibility;

\- reasonable training cost;

\- reliable inference;

\- clear evaluation;

\- straightforward checkpointing;

\- maintainability.

More sophisticated detector architectures may be considered later if the baseline exposes a concrete failure mode that justifies the additional complexity.

**---**

**## 24. Training Requirements**

The initial detector experiment must record:

\- dataset version;

\- train/validation split;

\- model architecture;

\- input resolution;

\- optimizer;

\- learning rate;

\- batch size;

\- number of epochs;

\- augmentation configuration;

\- confidence threshold;

\- NMS configuration;

\- random seed;

\- checkpoint selection criterion.

Training artifacts should be retained locally and associated with the corresponding experiment configuration.

**---**

**## 25. Failure Analysis**

Quantitative metrics must be accompanied by qualitative failure analysis.

At minimum, inspect examples of:

\- false positives on zero-lesion images;

\- missed single lesions;

\- missed small lesions;

\- duplicate detections;

\- merged detections;

\- dense multi-lesion images;

\- unusual body locations;

\- low/high pixel-spacing cases;

\- low-confidence detections.

The purpose of failure analysis is to identify actual model limitations rather than generate an unbounded list of possible future experiments.

**---**

**## 26. CV-2 → CV-3 Interface Validation**

A detector performing well independently does **\*\*not\*\*** establish that CV-2 and CV-3 compose correctly.

The CV-2 → CV-3 interface is therefore a required validation gate.

For a representative sample of real CV-2 detections:

\`\`\`text

Input image

    ↓

CV-2 detection

    ↓

candidate box

    ↓

crop / frame normalization

    ↓

CV-3 segmentation

\`\`\`

CV-3 segmentation quality must be measured on these detector-generated inputs.

The evaluation must not rely solely on CV-3's original ISIC test results.

**---**

**## 27. Interface Evaluation Requirements**

The CV-2 → CV-3 interface evaluation should separately examine:

**### Sparse detections**

Images containing relatively few lesions.

**### Dense detections**

Images containing many lesions.

This is important because dense images can produce:

\- smaller candidate boxes;

\- tighter lesion spacing;

\- greater crop interaction between neighboring lesions;

\- more challenging normalization;

\- greater sensitivity to detector localization error.

The interface evaluation should compare detector-generated crops against an appropriate reference framing strategy.

If CV-3 performance degrades materially on detector-generated crops, the next step should be determined from the observed failure mode, such as:

\- crop-margin adjustment;

\- normalization changes;

\- CV-3 fine-tuning;

\- detector localization improvements.

No assumption should be made that independently strong CV-2 and CV-3 models will automatically compose well.

**---**

**## 28. Product Architecture and Domain Routing**

CV-2 is primarily intended for the **\*\*wide-field / non-lesion-centric branch\*\*** of the DermaSense architecture.

It should not automatically be assumed that every input must pass through CV-2.

The eventual product may contain domain-specific routing, for example:

\`\`\`text

                         IMAGE

                           │

                           ▼

                         CV-1

                        Quality

                           │

                           ▼

                  Domain / Input Routing

                     │             │

                     │             │

              lesion-centric    wide-field

                     │             │

                     │             ▼

                     │           CV-2

                     │      Candidate Detection

                     │             │

                     │             ▼

                     └──────────► CV-3

                              Segmentation

                                   │

                                   ▼

                                  CV-4

                             Classification

                                   │

                                   ▼

                              CV-5 / CV-6

                          Explainability / UQ

                                   │

                                   ▼

                                  CV-7

                                Temporal

                                   │

                                   ▼

                                  CV-8

                                Severity

\`\`\`

The domain-routing mechanism itself is a separate architectural component and is not considered part of the CV-2 detector specification.

The immediate CV-2 implementation should therefore remain focused on the detection task defined in this document.

**---**

**## 29. Initial CV Pipeline Position**

The following diagram shows the **\*\*wide-field processing branch in isolation\*\***.

It is not intended to imply that every DermaSense input must pass through CV-2.

The eventual product architecture may route different input domains through different processing paths, as described above.

The current wide-field branch is:

\`\`\`text

                 IMAGE

                   │

                   ▼

                CV-1

               Quality

                   │

                   ▼

                CV-2

       Lesion Candidate Detection

                   │

                   ▼

                CV-3

          Lesion Segmentation

                   │

                   ▼

                CV-4

             Classification

                   │

                   ▼

                CV-5

            Explainability

                   │

                   ▼

                CV-6

             Uncertainty

                   │

                   ▼

                CV-7

          Temporal / Change

                   │

                   ▼

                CV-8

               Severity

                   │

                   ▼

        Structured Clinical /

             Risk Context

\`\`\`

**---**

**## 30. Related DermaSense Specifications**

CV-2 should be interpreted together with the broader DermaSense dataset, taxonomy, and product-definition documents.

Relevant specifications include:

\- **\*\*CV dataset specification:\*\*** \`CV\_DATASET\_SPEC\_v1.0\`

\- **\*\*Stage-1 label / taxonomy definition:\*\*** accepted Stage-1 clinical/risk label definition

\- **\*\*CV-3 segmentation specification:\*\*** \`docs/cv3\_segmentation\_baseline.md\`

\- **\*\*CV-2 detection specification:\*\*** this document

These documents serve different purposes.

The CV-2 specification defines:

\- the detection task;

\- dataset choice;

\- provenance considerations;

\- evaluation requirements;

\- downstream interface requirements.

The CV-3 specification defines:

\- the segmentation baseline;

\- segmentation-specific experimental evidence;

\- segmentation evaluation.

The Stage-1 taxonomy defines the downstream clinical/risk interpretation and should not be redefined by CV-2 detection outputs.

Where documents overlap conceptually, the more specific component specification should define implementation behavior while preserving the higher-level product and clinical definitions.

**---**

**## 31. Scope Boundaries**

CV-2 does **\*\*not\*\*** attempt to solve:

\- clinical diagnosis;

\- lesion classification;

\- lesion segmentation;

\- lesion severity;

\- temporal change;

\- explainability;

\- uncertainty estimation;

\- smartphone-domain generalization in its entirety.

These remain downstream or separate concerns.

CV-2's responsibility ends at reliable candidate localization.

**---**

**## 32. Implementation Sequence**

CV-2 implementation should proceed in the following order:

\`\`\`text

1\. Dataset loader

        ↓

2\. Internal stratified train/validation split

        ↓

3\. Post-split metadata balance audit

        ↓

4\. Confirm pre-committed acceptance thresholds

        ↓

5\. Baseline detector training

        ↓

6\. Validation threshold selection

        ↓

7\. Official test evaluation

        ↓

8\. Zero-candidate analysis

        ↓

9\. Lesion-density stratified analysis

        ↓

10\. Failure-case analysis

        ↓

11\. CV-2 → CV-3 interface validation

        ↓

12\. Decision: accept / revise baseline

\`\`\`

The implementation should stop and reassess if a fundamental dataset or interface assumption fails.

It should not automatically proceed into increasingly complex model optimization.

**---**

**## 33. Acceptance Criteria**

CV-2 should be considered ready to move forward when:

1\. the dataset loader is verified;

2\. the internal split is reproducible;

3\. the official test set remains untouched;

4\. zero-candidate behavior has been evaluated;

5\. the pre-committed acceptance thresholds have been evaluated
   on the validation set;

6\. candidate recall is acceptable for downstream use;

7\. false-positive behavior is understood;

8\. performance has been examined across lesion-density buckets;

8\. standard detection metrics have been recorded;

9\. major failure modes have been documented;

10\. the CV-2 → CV-3 interface has been tested;

11\. the detector's checkpoint and configuration are reproducible.

The objective is not to establish state-of-the-art detection performance.

The objective is to establish a **\*\*credible, reproducible, product-relevant detection component\*\***.

**---**

**## 34. Decision Record**

**### Decision**

**\*\*CV-2 will be implemented as a dedicated lesion candidate detector using iToBoS 2024 as the primary development dataset.\*\***

**### Rationale**

iToBoS provides the missing supervision required for the detection problem:

\`\`\`text

zero / one / many lesions

        \+

approximate localization

\`\`\`

This is complementary to CV-3's segmentation supervision:

\`\`\`text

one lesion

     \+

precise pixel-level boundary

\`\`\`

**### Accepted limitations**

\- iToBoS is a 3D-TBP clinical/research dataset rather than consumer smartphone imagery.

\- Participant-level overlap with UQ Longitudinal / SLICE-3D cannot currently be resolved from the exposed identifiers.

\- The official train/test age distributions are not perfectly matched.

\- The dataset carries a CC BY-SA 4.0 license whose implications for downstream model/product distribution require legal review.

**### Standing guardrails**

\- iToBoS and UQ Longitudinal / SLICE-3D must not be treated as jointly independent datasets in a single reported evaluation claim until overlap can be established.

\- The official iToBoS test set must remain untouched during model development.

\- Zero-candidate behavior must remain a first-class evaluation target.

\- Aggregate mAP must not be used as the sole criterion for detector acceptance.

\- The initial baseline must be evaluated against the pre-committed
  thresholds in Section 22: ≥95% lesion recall, ≤5% false-positive
  rate on zero-lesion images, and ≥90% recall on the 10+ lesion-density
  bucket.

\- CV-2 → CV-3 interface performance must be evaluated before declaring the two-stage pipeline validated.

**---**

**## 35. Final Definition**

The official DermaSense CV-2 component is:

\> **\*\*A lesion candidate detection system for wide-field, non-lesion-centric images that determines whether one or more lesion candidates are present and approximately localizes them for downstream CV-3 segmentation.\*\***

CV-2 is therefore not a redundant precursor to segmentation.

It exists because:

\`\`\`text

CV-2 asks:

"Where might the lesion(s) be?"

CV-3 asks:

"What exactly is the boundary of this lesion?"

\`\`\`

This separation preserves a clean division of responsibility and allows each component to be independently trained, evaluated, audited, and improved.

**---**

**\*\*Status: Accepted — implementation may proceed.\*\***