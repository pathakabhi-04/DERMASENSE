# Start Here — DermaSense Academic Track

**Branch:** `academic` (orphan: no shared history with `main`)
**Local folder:** `~/dermasense_academic`
**Audience:** anyone running or extending the academic experiments,
including the RAG collaborator (Section 5).

---

## 1. What to read, in order

| # | File | What you get from it |
|---|---|---|
| 1 | this file | Setup, the RunPod loop, and branch rules |
| 2 | `docs/academic_joint_seg_cls_spec.md` §2 | **Why the design is what it is**: why detection, segmentation and classification cannot be one jointly trained model, and why segmentation + classification is the pair we merged |
| 3 | `docs/academic_joint_seg_cls_spec.md` §1, §3–§10 | Experiment 1: one shared-encoder model (segmentation + classification) on HAM10000, three setups × 3 seeds, decision rule, stop conditions |
| 4 | `docs/academic_resnet50_three_task_spec.md` | Experiment 2: the same ResNet-50 backbone trained separately for CV-2, CV-3 and CV-4, plus Grad-CAM attention maps and how they are scored |

Rules shared by both specs (and any new one):
- Every experiment fixes its question, sample, metrics and decision rule
  **before** it runs, and ends with an anti-rabbit-hole boundary.
- Results are reported whichever way they come out. A failed match is a
  finding, not a reason to tune until it passes.

Useful background that lives on `main`, not here (read with
`git show origin/main:<path>`):
`docs/project_state.md` (the full production pipeline),
`docs/cv2_status.md` (why CV-2 recall stops at 0.81),
`docs/cv4b_backbone_finetune_design.md` §12 (the original RunPod runbook
this workflow is based on).

---

## 2. Local folder setup

The academic work lives in its own folder, with `academic` as its default
branch. The product repo at `~/dermasense` stays on `main` and is not touched.

```bash
git clone -b academic https://github.com/pathakabhi-04/DERMASENSE.git ~/dermasense_academic
cd ~/dermasense_academic
git branch --show-current            # -> academic

# Datasets are not in git. Point data/raw at the same HDD store main uses.
mkdir -p data
ln -s /mnt/hdd/dermasense_data/raw data/raw
ls data/raw/ham10000_masks/HAM10000_segmentations_lesion_tschandl | wc -l   # -> 10015

python3 -m venv .venv && . .venv/bin/activate
```

### Bringing code over from `main`

`academic` starts with specs only. Copy what an experiment needs from `main`
explicitly, path by path, and commit it here. This works across the
unrelated histories:

```bash
git fetch origin
git checkout origin/main -- src/explainability/gradcam.py src/explainability/evidence.py
git checkout origin/main -- scripts/measure_cv2_revised_metrics.py scripts/evaluate_cv3.py
git checkout origin/main -- data/splits/isic2019 data/splits/isic2018_task1 data/splits/itobos_detection
git checkout origin/main -- requirements-pod.txt
git commit -m "Bring over <what> from main"
```

Copy only what a spec uses. After copying, the file belongs to `academic`,
and changes to it are not merged back into `main` automatically.

---

## 3. Training loop on RunPod

Same discipline as the CV-4b runs: **nothing is billed at GPU rates until the
data is on the volume and verified.**

S3 settings for the RunPod network volume (credentials live in `~/.aws`,
never in git):

```bash
export S3="--region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io"
export BUCKET=s3://4tlwcuo1xg
aws s3 ls $BUCKET/ $S3               # confirm the volume still exists first
```

The volume mounts at `/workspace` on the pod. **Academic data and runs go
under the `dermasense_academic/` prefix**, so they never overwrite the
product repo's `/workspace/dermasense` tree.

### 3.1 Build and upload the data (local, no pod)

The split CSVs store absolute local paths (`/home/abhinav-pathak/...`), which
do not exist on a pod. As with `scripts/build_gpu_bundle.py` on `main`,
package each experiment's data as a flat, relative bundle: `dataset.csv`,
`images/`, masks if needed, `SHA256SUMS`, `BUNDLE.json`. Verify it locally,
then upload:

```bash
aws s3 sync data/processed/<bundle>/ $BUCKET/dermasense_academic/data/<bundle>/ $S3
aws s3 ls --recursive --summarize $BUCKET/dermasense_academic/data/<bundle>/ $S3
#   object count and total bytes must match the local bundle
```

### 3.2 Push code, then start the pod

```bash
git push origin academic             # the pod only ever runs pushed code
```

Start a pod with the network volume attached (PyTorch template). Then:

```bash
cd /workspace
git clone -b academic https://github.com/pathakabhi-04/DERMASENSE.git dermasense_academic \
  || (cd dermasense_academic && git fetch origin && git checkout academic && git pull)
cd /workspace/dermasense_academic
git log --oneline -1                 # must equal the commit you just pushed

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # do NOT reinstall torch
pip install -r requirements-pod.txt  # never requirements.txt on the pod

(cd /workspace/dermasense_academic/data/<bundle> && sha256sum -c SHA256SUMS --quiet) \
  && echo "Bundle verified. Safe to train."
```

### 3.3 Train

```bash
tmux new -s academic
PYTHONPATH=. python3 scripts/<train_script>.py \
  --data-root /workspace/dermasense_academic/data/<bundle> \
  --run-root  /workspace/dermasense_academic/runs/<run_name>
# detach: ctrl-b d. The run survives an SSH drop.
```

Checkpoints go to `/workspace/...` (the volume) every epoch, never to the
pod's ephemeral disk.

### 3.4 Pull back everything, then terminate

```bash
# local machine
aws s3 sync $BUCKET/dermasense_academic/runs/<run_name>/ ~/dermasense_academic/runs/<run_name>/ $S3
aws s3 sync $BUCKET/dermasense_academic/evaluation/ ~/dermasense_academic/evaluation/ $S3
```

Before terminating, compare object counts and bytes on both sides (the
`--summarize` check above). Checkpoints only on the volume are lost when it
is deleted; that nearly happened on 2026-09-02
(`docs/project_state.md` on `main`, "RunPod volume snapshot").

Then:
1. **Terminate the pod** (GPU billing stops).
2. Commit results locally: metric CSVs, contact sheets and the `*_result.md`
   are committed; weights (`runs/`, `*.pt`) are not (see `.gitignore`).
3. `git push origin academic`.
4. The network volume itself is kept or deleted as a separate, explicit
   decision (it is billed monthly).

---

## 4. Branch rules

- `academic` is the only branch for this track. Never merge it into `main`
  or `main` into it. Move code across with `git checkout origin/main -- <path>`.
- `git pull --rebase origin academic` before every push. **Never force-push**
  `academic`: other people build on it.
- Never commit data, weights, `.env` or API keys. `.gitignore` covers the
  usual paths. Check `git status` before `git add`.
- Every experiment gets a spec in `docs/` before its first run and a
  `*_result.md` after it.

---

## 5. For the RAG collaborator

The same method applies to the RAG side of the academic project.

1. **Set up** your own clone exactly as in Section 2 (`git clone -b academic ...`).
   Bring over the RAG code you need from `main` once:
   `git checkout origin/main -- src/rag docs/rag` (add the CV-8 fixtures and
   `src/risk/` too if you run the CV-grounded tests; without them those tests
   fail, the same problem `rag-development` had).
2. **Write a spec before changing the architecture.** Put it in
   `docs/rag/academic_<topic>_spec.md`, in the same format: question, what
   changes, metrics (e.g. retrieval Top-1/Top-3 on the fixed eval set),
   decision rule, anti-rabbit-hole boundary.
3. **Record what changed** in `docs/rag/CHANGES.md` (ADDED / EXTENDED /
   ⚠️ CHANGED / NOT CHANGED), so the CV side can see what affects the
   CV↔RAG interface.
4. **Push to `academic`** following Section 4: pull with rebase first, no
   force-push, no `.env`, no FAISS indexes (they are rebuilt, not committed).
5. **GPU work** (embedding, fine-tuning, indexing large corpora) follows
   Section 3 exactly, under a separate prefix:
   `$BUCKET/dermasense_academic/rag/`. Hosted-API work (the current Gemini
   baseline) needs no pod.
6. **Interface changes** that affect the CV side (anything consuming CV-4 /
   CV-8 outputs) must be noted in the spec **and** in `CHANGES.md` as
   ⚠️ CHANGED before pushing.
