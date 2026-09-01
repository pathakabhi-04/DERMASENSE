# CV-7 Data Staging — UQ Longitudinal Bounded Sample

**Status:** Bounded sample extracted and staged to the RunPod volume.
Full dataset deliberately not uploaded — see reasoning below. This is a
data-staging record, not a technical CV-7 spec (that still needs this
sample inspected further: image sizes, per-lesion pairing logic, and a
first prototype pass before the model/training spec can be written).

## Source

*A longitudinal dataset of tile and corresponding dermoscopic images
with metadata for identifying skin cancers*, Scientific Data (Nature),
2025. Repository: UQ eSpace, DOI `10.48610/a13deaf`. Downloaded by the
user directly from the repository as a BagIt-packaged zip (~62GB
compressed) to `data/raw/UQ_zip/` (not committed — matches the
gitignored-raw-data convention already used for iToBoS/ISIC/PAD-UFES).

## What's actually in the zip (inspected via `zipfile`/`unzip -l`, no
full extraction needed for this step)

| Category | Uncompressed size | Files |
|---|---|---|
| Dermoscopic Images | 63.77 GB | 35,914 |
| Tile Images (3D-TBP crops) | 3.55 GB | 1,074,970 (tiny, ~3KB avg) |
| Metadata (2 xlsx + 1 pdf) | ~76 MB | 3 |

**Correction to an earlier assumption:** the plan going into this
assumed the zip would contain a much larger non-longitudinal portion
that could be trimmed away (the pattern that worked for iToBoS's test
split). That doesn't apply here — the Dermoscopic Images folder (63.77GB)
essentially *is* the documented longitudinal subset already (35,914
files vs. the paper's reported 35,909). Filtering to participants with
≥2 visits (the actual longitudinal definition) only removes 2.86GB
(127 single-visit participants) from the 63.77GB, leaving 331
participants / 57.7GB / 7,672 lesions — consistent with the paper's
340/7,038 figures (small discrepancy plausibly from dataset-version
drift between publication and current download, not a parsing error).
There was no hidden bloat to trim; 58GB is close to the real floor for
the complete dataset.

Filename structure is flat and self-describing:
`{General|HighRisk}{ParticipantID}_Lesion{N}_visit{V}[-{dup}].jpg`, two
cohorts (General: 176 longitudinal participants / ~35GB; HighRisk: 155 /
~28.7GB).

## Why a bounded sample, not the full 58GB

Two independent storage constraints, discovered during this step, not
assumed going in:
- **Local disk**: only ~23GB free (now ~18GB after this extraction) —
  the full 58GB longitudinal set cannot be extracted locally at all
  without freeing significant space first.
- **RunPod volume**: ~25-30GB free (71/100GB already used) — the full
  set doesn't fit there either without a resize.

Rather than resize/free space to fit the complete dataset immediately,
staged a **bounded, stratified, seeded sample first** — same discipline
used for every other dataset this session (CV-1.5's 150-image held-out
set instead of iToBoS's full 8,481-image test split; CV-3's 1,000-crop
sample instead of 5,686). Validate the CV-7 approach against this before
committing the resize cost/time to the full 58GB.

## Sample composition

Seed 42. Proportional stratification across cohorts (176:155 →
16 General : 14 HighRisk of 30 total), drawn from the 331 true
longitudinal (≥2-visit) participants only.

- **30 participants** (16 General, 14 HighRisk) — full list:
  `analysis/quality/cv7_temporal_data/sampled_participants.txt`
- **2,772 dermoscopic images** + the 3 metadata files = **2,775 files,
  5.12 GB**
- Extracted via Python's `zipfile` module directly (not shell `unzip`) —
  the archive's paths contain spaces ("Dermoscopic Images"), which broke
  shell-based extraction via word-splitting; `zipfile` reads member names
  exactly regardless of spaces.

## Where it lives

- Local: `data/raw/uq_longitudinal/` (gitignored, matches convention).
  The source zip remains at `data/raw/UQ_zip/` — not deleted, since
  re-downloading 62GB would be costly and more participants may be
  sampled from it later without needing a fresh download.
- RunPod volume: `s3://4tlwcuo1xg/dermasense/data/raw/uq_longitudinal/`
  (same relative path convention as every other dataset on the volume).

## What this unblocks, and what it doesn't

Unblocks: inspecting real UQ Longitudinal images/metadata to write the
technical CV-7 spec (image format/resolution, per-lesion visit pairing,
what the two metadata xlsx files actually encode).

Does not yet unblock: training on the full dataset. If the bounded-sample
prototype validates the approach, pulling the remaining ~301 participants
requires either a volume resize (~70-80GB more, ~$5-6/month at RunPod's
$0.07/GB/month rate) or freeing local+pod space first — not decided here,
deferred until the sample proves the approach is worth scaling.
