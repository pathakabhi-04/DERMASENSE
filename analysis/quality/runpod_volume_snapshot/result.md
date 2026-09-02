# RunPod Volume Snapshot — Before Termination

**Status: DONE. Verdict: safe to terminate now.** This snapshot found
a real gap between what was assumed to be safely local-only and what
was actually only on the volume — that gap has been closed (see
"Action taken" below) before this verdict was given.

## Why this exists

The user asked whether the RunPod network volume (holding ISIC2018,
ISIC2019, iToBoS, PAD-UFES, and the staged UQ Longitudinal subset)
could be terminated now, re-provisioning only what's needed when
Phase 2 work actually starts. The plan itself was sound (matches this
project's bounded, just-in-time provisioning discipline throughout).
Before acting on an irreversible operation, `docs/build_on_baseline_1.md`'s
own answer to that question recommended a paper-trail snapshot first
— this is that snapshot, done properly rather than assumed.

## What was found: the volume is a full repo mirror, not just datasets

The initial assumption (stated in conversation before this snapshot)
was that the volume held only `dermasense/data/raw/{isic2018,isic2019,
itobos,pad_ufes,uq_longitudinal}/` — the five research datasets. That
assumption was **incomplete**: `dermasense/` on the bucket mirrors the
entire repository working directory, including `.git/`, `.venv/`,
`checkpoints/`, `runs/`, `evaluation/`, and a `_verified_backup/`
directory, plus a bucket-root `.cache/pip/` (pip's HTTP cache from
package installs on the pod).

## Full inventory (verified via `aws s3 ls --recursive --summarize`,
per-prefix, since a single bucket-wide recursive listing degrades on
this volume size — the same RunPod API limitation already documented
in `analysis/quality/cv7_temporal_data/result.md`)

| Path | Objects | Size | Category |
|---|---|---|---|
| `dermasense/data/raw/isic2018/` | 5,192 | 11.26 GB | Public dataset — re-downloadable |
| `dermasense/data/raw/isic2019/` | 25,337 | 19.58 GB | Public dataset — re-downloadable |
| `dermasense/data/raw/itobos/` | 17,099 | 9.55 GB | Public dataset — re-downloadable |
| `dermasense/data/raw/pad_ufes/` | 2,303 | 10.79 GB | Public dataset — re-downloadable |
| `dermasense/data/raw/uq_longitudinal/` | 8,754 | 16.08 GB | Re-extractable from the retained source zip |
| `.cache/pip/` | 551 | 4.13 GB | Disposable — pip HTTP cache, regenerates automatically |
| `dermasense/checkpoints/` | 34 | 2.04 GB | **Trained model weights — see "Action taken"** |
| `dermasense/runs/` | 118 | 106 MB | **Training/eval run logs — see "Action taken"** |
| `dermasense/evaluation/` | 101 | 86.5 MB | **Evaluation metrics/artifacts — see "Action taken"** |
| `dermasense/_verified_backup/` | ~2 | ~10 MB | Confirmed identical to local copy (byte-for-byte size match) |
| `dermasense/weights/` | 1 | 5.5 MB | A pretrained YOLO weight — re-downloadable |
| `dermasense/.git/` | 385 | 7.7 MB | Redundant with GitHub (pathakabhi-04/DERMASENSE) |
| `dermasense/.venv/` | (not fully enumerated — large, times out like isic2019/itobos did) | — | Disposable — reproducible from `requirements.txt` |
| `dermasense/{src,tests,docs,scripts,configs,analysis,logs}/` | ~315 combined | ~3 MB combined | Redundant with the git repo / GitHub |

**Total datasets alone: 58,685 objects, 67.27 GB** (matches the figure
already used in `docs/build_on_baseline_1.md` Section C).

## Action taken: closed a real gap before it became data loss

Comparing the volume's `checkpoints/` listing against the local
`checkpoints/` directory found **8 files that existed ONLY on the
volume**, not locally — these would have been permanently lost on
termination, with no straightforward way to reproduce them exactly
(retraining is possible but not guaranteed to reproduce the same
weights):

- `cv3_512/config.json`, `history.json`, `last.pt` (only `best.pt` was local)
- `cv3_768/best.pt`, `config.json`, `history.json`, `last.pt` (entire variant, not local at all)
- `pad_ufes_c1_partial_finetune_best.pt` (+ `.sha256`) — at checkpoints root, distinct path from the `archive/` copies already local
- `pad_ufes_c1_partial_finetune_seed123_best.pt` (+ `.sha256`) — at root
- `pad_ufes_c1_partial_finetune_seed42_best.pt` (+ `.sha256`) — at root
- `pad_ufes_c2_full_finetune_best.pt` (+ `.sha256`) — not local at all
- `pad_ufes_e1_scc_weighted_best.pt` (+ `.sha256`) — not local at all
- `pad_ufes_f1_supcon_best.pt` (107 MB, + `.sha256`) — not local at all

**All 8 were downloaded to local `checkpoints/` and verified**: MD5 of
each downloaded file was cross-checked against S3's own `ETag`
(confirms the transfer was byte-for-byte correct, independent of the
`.sha256` sidecars). 5/6 checked `.sha256` sidecars matched; one
(`pad_ufes_c1_partial_finetune_best.pt.sha256`) did not match its file
— traced to the sidecar itself being stale on the volume (the file's
MD5 matches S3's ETag exactly, so the downloaded `.pt` file is correct;
the `.sha256` companion was already wrong before this snapshot touched
it, not a transfer error here).

`runs/dermasense/runs/cv2/` (106 MB) and `evaluation/{cv2,cv3,cv3_bce,
cv3_dice}/` (86.5 MB) — training/eval logs and per-image metrics not
present locally (local `runs/` was 5.4MB/1 file vs. the volume's
106MB/118 objects) — were also synced down via `aws s3 sync` for the
same reason: lower severity than trained weights, but still
genuinely hard to regenerate exactly (would require re-running full
evaluation scripts) and worth the ~2 minutes it took.

Local disk after all downloads: 23GB free (was already at 92% used;
the ~1GB combined download did not create a space problem).

## What was confirmed already safe, no action needed

- `_verified_backup/cv2_diagnostics_backup.tar.gz` and its `MANIFEST.sha256`
  — byte-for-byte size match against the local copies already sitting
  untracked in the repo root.
- `.git/`, `src/`, `tests/`, `docs/`, `scripts/`, `configs/`, `analysis/`,
  `logs/` on the volume — redundant with the GitHub repository, which
  is the actual canonical copy.
- `.venv/` and `.cache/pip/` — fully reproducible from `requirements.txt`,
  zero risk regardless of size.
- The five research datasets — all re-obtainable from their original
  public sources (or, for UQ Longitudinal, re-extractable from the
  retained source zip) if ever needed again.

## Verdict

**Safe to terminate the volume now.** Nothing irreplaceable remains
exposed to it — the one real gap this snapshot found (8 checkpoint
files + run/eval logs that existed only on the volume) has been closed
by downloading and verifying them locally, not merely noted. Re-staging
any of the datasets later, if a `docs/build_on_baseline_1.md`
Section C trigger fires, follows the plan already written there: the
participant lists and staging scripts needed are already committed to
git, so nothing about this termination requires re-discovering how to
re-provision — only re-running already-known steps.
