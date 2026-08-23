# PAD-UFES-20 Data Layer v0.1

Status: FROZEN

## Dataset

Images: 2298
Patients: 1373
Source lesion IDs: 1641
Operational lesion UIDs: 1891

## Native diagnoses

ACK: 730
BCC: 845
MEL: 52
NEV: 244
SCC: 192
SEK: 235

## Split

Train:
- Patients: 961
- Operational lesions: 1323
- Images: 1610

Validation:
- Patients: 204
- Operational lesions: 278
- Images: 336

Test:
- Patients: 208
- Operational lesions: 290
- Images: 352

## Leakage constraints

Patient overlap: 0
Operational lesion overlap: 0
Image overlap: 0

## Identity policy

`lesion_id` is retained as the native PAD-UFES identifier.

`lesion_uid = patient_id + "__" + lesion_id`

`lesion_uid` is the operational lesion identity used
for leakage checks.

Patient-level separation is mandatory.

## Validation

Independent split validation: PASS