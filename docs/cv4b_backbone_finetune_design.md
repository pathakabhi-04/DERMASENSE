# CV-4b backbone fine-tune: design for a single effective GPU run

**Date:** 2026-09-13
**Status:** design, pre-committed. Nothing here is chosen after seeing a
result.
**Prerequisite reading:** `analysis/quality/mel_sensitivity/cv4b_retrain_result.md`
(the linear-probe refit this escalates from).

## 1. Why a GPU run at all

The linear-probe refit recovered non-ISIC AUC 0.60 → 0.73 and stopped
short of its 0.80 bar, while pushing non-ISIC benign referral to 54%.
Refitting the *linear layer* on broader data helps but cannot change
what the frozen backbone extracted in the first place. The remaining
question is representational, and answering it requires gradients
through the backbone. That is the only reason to spend GPU here.

## 2. A cost correction worth making explicitly

The standing constraint recorded for this project is: **"we can not
finance the GPU for all deployment at this stage."** That is about
*serving* — a GPU running continuously behind every user request. It is
not the same cost as this.

This is a one-off fine-tune of ResNet-50 **layer4 only**, over ~19.2K
images of 224×224, for ≤30 epochs. The honest headline is that compute
is not the binding constraint — **data loading is**, by a wide margin.

| item | value | basis |
|---|---|---|
| FLOPs/epoch | ~135 TFLOPs | 19.2K imgs × (~4 GFLOP fwd + ~3 GFLOP layer4-only bwd) |
| compute time/epoch | ~10–20 s | estimate: T4-class card at a conservative 15–20% of peak |
| **decode time/epoch, originals** | **~12 min** | **measured**, 26 img/s, 4 workers, this machine |
| **decode time/epoch, pre-resized** | **~0.6 min** | **measured**, 498 img/s, same machine |
| wall time per run (pre-resized) | ~20–30 min | 30 epochs + per-epoch validation |
| runs planned | 3 (§5) | pooled + 2 LOSO folds |
| **total GPU time** | **~1.5–3 hours** | |
| **total GPU cost** | **~$2–10** | T4 ≈ $0.35/hr, A10G ≈ $0.75/hr, spot cheaper |

**Correction, and the reason the pre-flight gate exists.** An earlier
draft of this section assumed ~20–60 s/epoch for data loading. Pre-flight
check 9 measured **26 img/s on full-resolution originals — roughly 12
minutes per epoch**, 15× worse than assumed, which across 3 runs × 30
epochs would have been ~18 GPU-hours of paid time spent almost entirely
on JPEG decoding.

Diagnosis (§11): a page-cache re-read ran at the *same* rate, so this is
CPU-decode-bound, not disk-bound — and therefore does not fix itself on a
faster machine's disk. Pre-resizing takes it to **498 img/s, a 19×
speedup**, which is what makes the table above hold.

That single check is the difference between a ~$5 experiment and a ~$40
one that looks identical from the outside.

**The compute row remains an estimate**; only the decode rows are
measured. If the first epoch on the rented card runs far slower than
this table, stop and diagnose rather than paying for 30 of them.

The point of the table: the thing blocking deployment (continuous
serving GPU) and the thing needed here (a few hours of one card) are
different orders of magnitude. This decision should not inherit the
deployment constraint by association.

## 3. What gets trained, and what deliberately does not

**Trainable:** ResNet-50 `layer4` (backbone `features[7]`) + a new binary
referral head. Everything else frozen, verified by exact parameter-count
assertion at startup (the `experiment_c1_partial_finetune.py` precedent,
which asserts `trainable_backbone == trainable_layer4` and refuses to
run otherwise).

**Why layer4 only:** the non-ISIC training pool is ~1,140 images (§5).
Full-backbone fine-tuning on that is an overfitting machine. layer4 is
also where task-specific/texture-and-context features live, which is
where a domain gap of this kind would be expected to bite. C1 used
exactly this recipe to produce the currently-shipped checkpoint, so it
is a known-working configuration in this codebase, not a new bet.

**The shipped CV-4 checkpoint is NOT modified.** The fine-tune writes a
*separate* checkpoint used only by CV-4b. CV-4's 6-class narration keeps
running on the existing, already-validated backbone.

| | if shared backbone is mutated | separate backbone (chosen) |
|---|---|---|
| CV-4 6-class metrics | all need re-validation | untouched, still valid |
| rollback | restore + re-verify pipeline | delete one file |
| blast radius on failure | whole diagnosis path | CV-4b routing only |
| inference cost | unchanged | **+1 ResNet-50 forward pass** |

The inference cost is the real price and it is not hand-waved: CV-4b
already extracts features through the classifier's backbone
(`orchestrator._refer_crop`), so this makes that a second, separate
forward. **This must be measured, not assumed** — it is a listed
deliverable in §10, and if it materially moves `/assess` latency the
shared-backbone option gets revisited as a deliberate trade rather than
discovered as a regression.

This also stays consistent with the CV-8 v1.2 contract, which already
treats the referral signal as an independent routing input rather than
something derived from `native_class`.

## 4. The defect found while designing this (and fixed)

`cv4b_retrain_split.csv` — the split behind the already-committed
linear-probe result — **splits DDI-2 by image, but DDI-2 has repeat
patients**: 89 patients contribute 204 images. **25 of its 132 held-out
DDI-2 test images come from patients that also appear in training.**

That is patient-level leakage. Direction of the bias is optimistic, so
the linear-probe conclusion ("did not clear the bar") is not overturned
by it — if anything the true number is slightly worse — but the number
itself is contaminated and is now annotated as such in
`cv4b_retrain_result.md`. It is not silently rewritten.

`scripts/build_cv4b_splits.py` replaces it, grouping by
`deidentified_patient_id` for DDI-2 and asserting that **no group and no
image spans two splits** before writing anything.

Stated honestly: DDI and Fitzpatrick17k carry **no patient identifier at
all**, so each of their images is treated as its own group. If those
datasets contain repeat patients, this split cannot detect it. That is a
limit of the data, disclosed, not a solved problem.

Also checked while here: **zero byte-identical duplicate images across
the three sources** (1,905 files, 1,905 unique MD5s). This does not rule
out re-encoded near-duplicates.

## 5. Splits, and the distinction that actually matters

Two different questions, deliberately measured separately:

**`pooled`** — 60/20/20 stratified by (source, is_malignant), patient-
grouped. Held-out images from sources the model *did* train on. This is
the in-distribution number and it is the **optimistic** one.

**`loso_atlas` / `loso_stanford`** — leave-one-source-out. Train on one
source group, test on an entirely unseen one.

> Deployment is a **fourth, unseen source**: a user's phone camera.
> Held-out images from sources present in training systematically
> overstate that. LOSO is the honest proxy. Where the two disagree, LOSO
> is the number that describes the product.

DDI and DDI-2 are both Stanford AIMI releases and DDI has no patient IDs
to cross-check, so **cross-dataset patient overlap between them cannot
be ruled out**. They are therefore treated as one source group
(`stanford`) and never split against each other. Holding out DDI while
training on DDI-2 would not be an unseen-source test.

Actual composition (from the built split):

| fold | train | val | test | test melanomas |
|---|---:|---:|---:|---:|
| pooled | 1,140 | 381 | 379 | 22 |
| loso_atlas | 987 (stanford) | 330 | 583 (fitz) | 83 |
| loso_stanford | 437 (fitz) | 146 | 1,317 (stanford) | 25 |

`pooled` test carries only 22 melanomas (DDI-2's test slice has 0) —
a direct consequence of stratifying on `is_malignant` rather than on
melanoma, with patient grouping on top. **Melanoma routing on `pooled`
is therefore not a usable headline number**; malignant-vs-benign AUC is.
`loso_atlas` (83 test melanomas) is the fold where melanoma routing is
worth reporting as more than an anecdote.

## 6. The mixing problem — the most likely way this run does nothing

ISIC train is 18,062 images. The non-ISIC train pool is ~1,140. At
natural frequency, clinical photos are **~6% of the gradient signal**,
and the most likely outcome of a naive run is a model that barely moves
and a conclusion that "fine-tuning doesn't help" — which would be a
false negative caused by the sampler, not by the representation.

**Mitigation (pre-committed):** `WeightedRandomSampler` with per-sample
weights set so that each batch is ~50% non-ISIC in expectation, with
epoch length fixed to the ISIC-scale count so an "epoch" stays
comparable. Non-ISIC images are therefore heavily repeated within an
epoch, which raises overfitting risk on them — bounded by (a) layer4-only
training, (b) the stronger augmentation in §7, (c) early stopping on the
§8 metric, all three of which are chosen for this reason.

**ISIC stays in the mix.** Dropping it and fine-tuning on 1,140 clinical
images alone would produce catastrophic forgetting, and CV-4b's only
validated numbers live on ISIC. §8 makes ISIC regression a run-failure
condition, not a footnote.

## 7. Augmentation — where the negative audit result earns its keep

The composition audit ruled out framing, zoom, segmentation reliability
and simple blur/contrast as explanations for the domain gap. What it
explicitly left open: **colour/skin-tone distribution, white balance,
lighting temperature, compression and camera characteristics.**

So the augmentation is targeted at exactly the untested hypothesis
rather than turned up uniformly:

| knob | current default | fine-tune | why |
|---|---|---|---|
| brightness | 0.10 | **0.35** | lighting varies wildly across clinical capture |
| contrast | 0.10 | **0.35** | ditto |
| saturation | 0.10 | **0.35** | camera processing differs per device |
| hue | 0.02 | **0.08** | white balance; the skin-tone axis |
| flips / rotation | keep | keep | already correct, lesions have no canonical orientation |
| RandomResizedCrop | off | **on, scale (0.7, 1.0)** | framing varies; audit showed real spread in area fraction |
| JPEG re-compression | — | **optional, q 30–90** | Fitzpatrick is heavily compressed, DDI is not |

This is domain randomisation aimed at a named, specific hypothesis. If
the run succeeds, an ablation over this block is the natural follow-up
to learn *which* axis mattered — but that is a later, separately-scoped
experiment, not something to bolt onto this run.

Validation and test use **deterministic** preprocessing, unchanged.

## 8. Metric, model selection, and the pre-committed decision rule

**Selection metric: non-ISIC validation malignant-vs-benign AUC.** Not
macro-F1. The existing `Trainer.fit` selects `best_epoch` on
`val.metrics.macro_f1`, which is the right metric for the 6-class task
and the *wrong* one here — picking a checkpoint on the wrong metric is a
classic way to waste a run that otherwise worked. The fine-tune script
therefore does its own selection and does not reuse `Trainer.fit`.

Logged every epoch, per source, on validation only: malignant AUC,
melanoma routing at the operating threshold, benign-referral rate, loss.

**Pre-committed rule — evaluated once, on test, after training ends:**

| criterion | bar | rationale |
|---|---|---|
| LOSO test AUC (`loso_atlas`) | **≥ 0.80** | the bar the linear probe missed at 0.73 |
| ISIC test AUC | **≥ 0.85** | same floor as the probe refit; no forgetting |
| non-ISIC benign referral | **≤ 0.45** | probe hit 0.54; a "win" bought by referring everything is not a win |

**All three must hold.** AUC alone is not sufficient — the linear probe's
headline melanoma-routing improvement was substantially bought by
referring 54% of benign clinical photos, and that failure mode must not
be repeatable by a model that merely scores well on AUC.

Outcomes, decided in advance:

- **All three met** → the representation *was* the limit and it is
  fixable at this cost. Proceed to wiring (§10), then re-measure
  `/assess` latency before anything ships.
- **ISIC floor breached** → catastrophic forgetting. The mixing ratio or
  LR is wrong. **One** retry with a corrected ratio is permitted; a
  second is not, because at that point it is a search, not a test.
- **LOSO < 0.80 but pooled ≥ 0.80** → adaptation works for *seen*
  sources and does not generalise to unseen ones. This is the most
  likely informative-failure outcome and it is a genuinely important
  product finding: it would mean per-deployment-source calibration data
  is required, and that no amount of training on these three sources
  makes a user's phone camera safe by itself.
- **Everything flat** → the gap is not layer4-shaped. Stop. Do not
  escalate to full-backbone fine-tuning on 1,140 images on momentum;
  that decision needs its own scope and probably its own data.

Test is touched **once**, at the end, by a script that loads the
selected checkpoint. No test-set number is available during training.

## 9. Pre-flight: the CPU gate that must pass before renting anything

Every one of these runs locally on CPU. **The GPU is not rented until all
pass.** The entire purpose is that the rented card never spends time
discovering a bug that a laptop could have found for free.

1. **Split integrity** — no group spans splits, no image spans splits,
   every referenced file exists and opens, class balance reported per
   split. (Already enforced inside `build_cv4b_splits.py`.)
2. **Freeze mask** — exact trainable parameter count equals layer4 +
   referral head, to the parameter. Assert, don't print.
3. **Overfit-one-batch** — 32 images, no augmentation, ~100 steps; loss
   must collapse toward ~0. If a model cannot memorise 32 images, the
   training loop, LR or label wiring is broken, and this catches it in
   minutes for free. **This is the single highest-value check here.**
4. **Determinism** — same seed, two 5-step runs, identical losses to
   ~1e-6. Guards against silent nondeterminism making results
   irreproducible.
5. **Checkpoint round-trip** — save, reload into a fresh model, assert
   bit-identical predictions on a fixed batch.
6. **Resume fidelity** — 4 steps straight through vs. 2 + resume-from-
   checkpoint + 2 must land on the same weights (model, optimizer *and*
   RNG state). A rented/spot instance can die mid-run; resume that
   silently restarts the LR schedule or reshuffles data is worse than no
   resume.
7. **Pre-resize fidelity** (§11) — re-run the existing baseline
   evaluation on pre-resized images and require it to reproduce the
   full-resolution numbers within ±0.01 AUC. If not, pre-resizing is
   rejected and the full data goes up instead.
8. **Metric sanity** — the AUC/routing functions on a synthetic
   known-answer case (perfect separation → 1.0, random → ~0.5).
9. **Throughput extrapolation** — time the loader (excluding worker
   spawn), compare against §2. Flags an order-of-magnitude surprise
   before it is being paid for by the hour.

**Status: 9/9 passing** (`scripts/preflight_cv4b_finetune.py`). Two of
the nine found real problems on the first run and are the reason this
section exists:

- check 9 caught the 15× data-loading underestimate (§2);
- check 7 rejected the first pre-resize configuration outright, then
  turned out to be mis-designed itself and was rebuilt on paired
  statistics (§11).

Both were found on a laptop, for free. Neither would have been visible
in a training log — the run would have completed and produced numbers
that looked fine.

The training and evaluation scripts have also been smoke-tested
end-to-end on CPU (3 steps, tiny batch): the loop trains, checkpoints,
selects, and the evaluator loads the selected checkpoint and reports
against §8. As expected for an untrained head, that smoke run scored
AUC ≈ 0.50 — itself a sanity signal. Those throwaway artifacts were
deleted so they cannot be mistaken for a result.

## 10. If it succeeds — what must happen before anything ships

Not automatic. In order:

1. Re-measure `/assess` latency with the second backbone loaded (§3).
2. Refit the referral head's operating point on val, same 40%-budget
   convention, and re-verify the floor-not-ratchet behaviour in CV-8 is
   unchanged (`tests/test_referral_head.py`, `tests/test_risk_convergence.py`).
3. Note that `referral_head.json` gains a backbone-checkpoint reference:
   a head fitted on adapted features is **silently wrong** if loaded
   against the original backbone. The loader must fail loudly on
   mismatch rather than score anyway — the same principle
   `ReferralHead.decide()` already applies to feature-dimension
   mismatch.
4. Only then, contract/docs update.

Point 3 is a new, real failure mode introduced by this design and it is
the one most likely to cause a silent, dangerous wrong answer in
production. It gets a test.

## 11. Data logistics (the unglamorous blocker)

ISIC2019 raw is **19 GB**, on an external HDD that has intermittently
disconnected during this project. Uploading that to a rented box is slow
and pointless: every image is resized to 224×224 before it reaches the
model anyway.

**Done** (`scripts/preresize_cv4b_dataset.py`): 26,739 images pre-resized
once, on CPU → **19 GB → 1.7 GB**, one upload. Augmentation still happens
on the pre-resized image and the final resize to 224 is unchanged.
**Validation and test use the same pre-resized images as training**, so
no train/eval preprocessing skew is introduced.

### The geometry was chosen by measurement, and the first attempt failed

Pre-flight check 7 compares each image against itself — original vs
pre-resized — through the real backbone and the shipped referral head.
Mean cosine similarity of the 2048-d features, over 240 validation
images:

| variant | feature cosine | decisions flipped |
|---|---:|---:|
| 256 BILINEAR q95 (first attempt) | 0.9633 | 23/240 |
| 256 LANCZOS q95 | 0.9757 | 21/240 |
| 256 LANCZOS q100 | 0.9876 | 15/240 |
| **320 LANCZOS q100 (chosen)** | **0.9932** | **9/240** |
| 256 LANCZOS PNG | 0.9955 | 9/240 (but ~10× the bytes) |

Two separate effects, both real and both larger than expected:
**BILINEAR without antialiasing** loses detail on a big downscale, and
**JPEG quantisation moves features more than the resize does.** The
obvious first choice (256/BILINEAR/q95) was rejected on measurement.

Worth recording as an aside, not pursued: this backbone is measurably
sensitive to JPEG compression, and **compression was one of the
hypotheses the composition audit explicitly left open** — Fitzpatrick17k
images are far more compressed than DDI's. That is a lead for a
separately-scoped experiment, not something to fold into this run.

### The check itself had to be fixed first

The original check 7 gated on the gap between two *independent* AUC
estimates with a 0.01 tolerance. That is unmeasurable: at n=240 the
standard error of AUC alone is ~0.04, so the tolerance sat well inside
the noise. It failed faithful configurations and ranked them
non-monotonically — PNG, the most faithful variant by every paired
measure, scored a *worse* AUC gap than a lossier one.

It now gates on **paired** statistics (feature cosine ≥ 0.99, score
Spearman ≥ 0.98, decision-flip rate ≤ 5%) and reports the AUC gap as
context only. Recorded because the failure mode is general: a check that
compares two noisy aggregate estimates cannot certify a tolerance
smaller than its own sampling noise, and one that looks strict while
being uninformative is worse than no check at all.

Also required, and easy to forget until the instance is already running:
checkpoints written to persistent storage (not the ephemeral disk),
every epoch, with optimizer and RNG state (§9.6).

## 12. Runbook: RunPod, network volume, data-before-GPU

The ordering principle is **nothing is billed at GPU rates until the data
is present and verified**. Data transfer, integrity checking and code
checkout all happen before a GPU is attached.

### 12.1 The portability problem this had to solve first

The split CSV and pre-resize manifest both store **absolute local paths**
(`/home/abhinav-pathak/...`), and `isic_rows()` resolves ISIC images
through `CVDataset`, which builds paths against a raw-data root. Shipping
those as-is to a pod means a crash on the first batch — or worse, needing
the 19 GB raw ISIC tree present just to construct filenames.

`scripts/build_gpu_bundle.py` produces a flat, relative, self-describing
bundle instead:

```
cv4b_bundle/
  dataset.csv    26,739 rows: relative_path, source, label, is_melanoma,
                 isic_split, pooled, loso_atlas, loso_stanford
  images/<source>/<stem>.jpg
  SHA256SUMS     integrity for the S3 round trip
  BUNDLE.json    counts + provenance (git commit, source checkpoint)
```

Nothing on the pod needs `CVDataset`, the raw datasets, or this machine's
layout — only the repo (from git) and the bundle (from the volume).

**The bundle is canonical locally too.** Pre-flight, training and
evaluation all read it via `--data-root`, so the configuration validated
on the laptop is byte-for-byte the one that runs on the rented card,
rather than a second code path first exercised when it costs money.

### 12.2 Sequence

```bash
# --- local, no GPU ---
PYTHONPATH=. python3 scripts/preresize_cv4b_dataset.py     # 19 GB -> 1.7 GB
PYTHONPATH=. python3 scripts/build_gpu_bundle.py --tar     # + cv4b_bundle.tar
PYTHONPATH=. python3 scripts/preflight_cv4b_finetune.py    # must be 10/10

# --- upload to the network volume (no GPU attached) ---
aws s3 cp data/processed/cv4b_bundle.tar s3://<bucket>/cv4b_bundle.tar
# then land it on the volume and untar to /workspace/cv4b_bundle

# --- on the pod, still no GPU / cheapest instance ---
git clone <origin> /workspace/dermasense && cd /workspace/dermasense
pip install -r requirements.txt
python scripts/verify_gpu_bundle.py --data-root /workspace/cv4b_bundle
#   -> "Bundle verified. Safe to attach a GPU."   (exits non-zero otherwise)

# --- only now attach/rent the GPU ---
tmux new -s cv4b
for FOLD in pooled loso_atlas loso_stanford; do
  PYTHONPATH=. python3 scripts/finetune_cv4b_backbone.py \
    --fold $FOLD --data-root /workspace/cv4b_bundle \
    --run-root /workspace/runs --workers 8
done
# detach with ctrl-b d; the run survives an SSH drop

# --- after training, test touched once per fold ---
PYTHONPATH=. python3 scripts/evaluate_cv4b_finetune.py \
  --fold pooled --data-root /workspace/cv4b_bundle --run-root /workspace/runs
```

Resume after a dead instance — checkpoints carry model, optimizer and RNG
state, and pre-flight check 6 proves resume is bit-identical to running
straight through:

```bash
PYTHONPATH=. python3 scripts/finetune_cv4b_backbone.py \
  --fold pooled --data-root /workspace/cv4b_bundle \
  --run-root /workspace/runs --resume
```

### 12.3 How much network-volume storage to buy

Measured, not guessed. Checkpoint sizes come from actually saving one.

| item | size | notes |
|---|---:|---|
| bundle, untarred | 1.70 GB | 26,739 images at 320px |
| bundled source checkpoint | 0.09 GB | included in the above tar |
| `cv4b_bundle.tar` | 1.84 GB | delete after untar |
| **peak during untar** | **3.5 GB** | tar + extracted tree coexist |
| repo clone | ~0.1 GB | code only; data is gitignored |
| `best.pt` per fold | 0.09 GB | model + binary head |
| `latest.pt` per fold | 0.21 GB | + AdamW state + RNG, for resume |
| checkpoints, 3 folds | 0.93 GB | |
| peak during atomic save | +0.21 GB | `.tmp` before rename |

Steady state after untar is ~2.8 GB; the transient peak is ~3.6 GB.

**Buy 20 GB.** That is ~5× the peak, and the reason for the margin is not
the numbers above — it is that pip may install into the volume rather
than the container image depending on how the pod is set up, and a CUDA
PyTorch install is **2.5–3.5 GB on its own**. If it lands on the volume,
10 GB gets uncomfortable. 20 GB also leaves room to keep `latest.pt` for
all three folds while a fourth run is in flight.

Do not go below 10 GB. Volume storage is billed continuously even when no
pod is running, so **delete the volume once the checkpoints are pulled
down** — that recurring charge, not the GPU hours, is what quietly
accumulates on an idle account.

### 12.4 Things that will bite, in rough order of likelihood

- **Network volumes are region-locked.** The volume lives in one
  datacenter and only GPUs in that datacenter can mount it. **Confirm the
  GPU type you want is actually available in that region *before*
  uploading 1.7 GB there** — otherwise the data is stranded and has to be
  re-uploaded elsewhere. This is the single most expensive ordering
  mistake available here, and it is invisible until the last step.
- **`--run-root` must point at the network volume** (`/workspace/runs`),
  not container-local disk, or checkpoints die with the pod and the
  resume path is worthless.
- **Upload the tar, not 26,739 loose files.** Per-object overhead
  dominates at that count; one 1.75 GB object transfers far faster.
- **`--workers 8`** on the pod (the local default of 4 was sized for this
  laptop). Decode is the bottleneck (§2), so match it to the pod's vCPUs.
- **Verify before renting, not after.** `verify_gpu_bundle.py` exits
  non-zero, so chain it: `python scripts/verify_gpu_bundle.py ... && <rent>`.
- **The source checkpoint is gitignored**, so cloning the repo does *not*
  bring the 90 MB checkpoint the fine-tune starts from. It now ships
  inside the bundle and `verify_gpu_bundle.py` checks for it — without
  that, the failure surfaces only after the GPU is attached and billing.
- A truncated S3 object still decodes as a valid JPEG — it just decodes
  to *different pixels*. Only the checksum catches that, which is why
  the default verifies all 26,739 files rather than a sample.

I have not re-verified RunPod's current API surface or pricing while
writing this; treat the region and mount-point specifics as "confirm in
the console", not as fact. The parts this repo controls — bundle layout,
checksums, resume, `--data-root`/`--run-root` — are tested.

## 13. What this design deliberately refuses to do

- **No hyperparameter search.** One configuration, pre-committed. The
  linear-probe result's own conclusion was that chasing the number with
  more probe variants was out of scope; that discipline does not lapse
  because the hardware got more expensive.
- **No full-backbone fine-tune** on 1,140 clinical images.
- **No new data acquisition** mid-experiment.
- **No touching the shipped CV-4 checkpoint.**
- **No test-set peeking** to decide when to stop.
