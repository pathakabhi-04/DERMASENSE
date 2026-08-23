# ISIC 2019 Identity & Split Audit v0.1

## Scope

This audit evaluates lesion identity, image grouping, diagnostic consistency, and the guarantees available for future train/validation/test splitting. It does not create dataset splits.

## Dataset identity

- Total images: **25331**
- Images with lesion ID: **23247**
- Images without lesion ID: **2084**
- Unique identified lesions: **11847**
- Unknown-ID proportion: **8.23%**

## Images per lesion

- Single-image lesions: **6788**
- Multi-image lesions: **5059**
- Multi-image lesion rate: **42.70%**

## Diagnostic consistency

- Lesions with multiple native diagnoses: **0**

- **PASS:** Every identified lesion has one consistent native diagnosis.

## Patient identity

- `patient_id` is **not present** in the supplied ISIC 2019 metadata.
- A patient-level split guarantee therefore cannot be established from this dataset alone.

## Split implication

- The **23247** images with identified lesion IDs can participate in a lesion-disjoint split.
- The **2084** images without lesion IDs cannot be proven lesion-disjoint.
- These unknown-ID records should not be silently treated as independent lesions.
- No train/validation/test split is frozen by this audit.

## Generated artifacts

- `docs/audits/isic2019/lesion_summary.csv`
- `docs/audits/isic2019/class_identity_summary.csv`
- `docs/audits/isic2019/ISIC2019_IDENTITY_AUDIT_v0.1.md`
