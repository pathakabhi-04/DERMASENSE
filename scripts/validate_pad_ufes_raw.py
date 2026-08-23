from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/pad_ufes")
METADATA = ROOT / "metadata.csv"
IMAGE_ROOT = ROOT / "images"

print("=" * 60)
print("PAD-UFES-20 RAW DATA VALIDATION")
print("=" * 60)

# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------
df = pd.read_csv(METADATA)

print("\n[1] Metadata")
print(f"Rows:              {len(df)}")
print(f"Unique img_id:     {df['img_id'].nunique()}")
print(f"Unique lesion_id:  {df['lesion_id'].nunique()}")
print(f"Unique patient_id: {df['patient_id'].nunique()}")

# ---------------------------------------------------------
# Basic duplicate checks
# ---------------------------------------------------------
duplicate_images = df["img_id"].duplicated().sum()

print("\n[2] Duplicate checks")
print(f"Duplicate img_id: {duplicate_images}")

# ---------------------------------------------------------
# Diagnostic counts
# ---------------------------------------------------------
print("\n[3] Native diagnostic counts")
print(df["diagnostic"].value_counts().sort_index().to_string())

# ---------------------------------------------------------
# Find actual PNG files
# ---------------------------------------------------------
image_files = list(IMAGE_ROOT.rglob("*.png"))

actual_image_ids = {
    path.name
    for path in image_files
}

metadata_image_ids = set(df["img_id"].astype(str))

missing_images = sorted(metadata_image_ids - actual_image_ids)
extra_images = sorted(actual_image_ids - metadata_image_ids)

print("\n[4] Image ↔ metadata correspondence")
print(f"Metadata image IDs: {len(metadata_image_ids)}")
print(f"PNG image files:    {len(actual_image_ids)}")
print(f"Missing images:     {len(missing_images)}")
print(f"Extra images:       {len(extra_images)}")

if missing_images:
    print("\nFirst missing images:")
    for x in missing_images[:10]:
        print("  ", x)

if extra_images:
    print("\nFirst extra images:")
    for x in extra_images[:10]:
        print("  ", x)

# ---------------------------------------------------------
# Final audit expectations
# ---------------------------------------------------------
expected = {
    "images": 2298,
    "lesions": 1641,
    "patients": 1373,
}

print("\n[5] Audit consistency")

checks = {
    "image_count": len(df) == expected["images"],
    "unique_image_count": df["img_id"].nunique() == expected["images"],
    "lesion_count": df["lesion_id"].nunique() == expected["lesions"],
    "patient_count": df["patient_id"].nunique() == expected["patients"],
    "duplicate_images": duplicate_images == 0,
    "missing_images": len(missing_images) == 0,
    "extra_images": len(extra_images) == 0,
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {name}")

print("\n" + "=" * 60)

if all(checks.values()):
    print("STATUS: PASS")
    print("PAD-UFES-20 raw dataset is internally consistent.")
else:
    print("STATUS: FAIL")
    print("Raw dataset validation found inconsistencies.")

print("=" * 60)
