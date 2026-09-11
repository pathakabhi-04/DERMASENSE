# DermaSense Medical Knowledge Corpus

**Version:** v0.1  
**Status:** Draft  
**Purpose:** Patient-aware RAG for the DermaSense medical guidance assistant

---

## 1. Purpose

This document defines the initial medical knowledge corpus that will be used by the DermaSense Retrieval-Augmented Generation (RAG) pipeline.

The corpus is intended to provide grounded medical information for:

- explaining computer-vision findings
- explaining suspicious skin-lesion characteristics
- explaining temporal changes
- providing general first-aid guidance
- identifying warning signs
- supporting severity-based escalation
- answering patient questions
- providing context-aware conversational guidance

The corpus is **not** intended to train the computer-vision models or to establish medical diagnoses.

---

## 2. Knowledge Sources

The initial corpus will prioritize authoritative medical organizations.

### Primary sources

1. American Academy of Dermatology (AAD)
2. National Cancer Institute (NCI)
3. MedlinePlus
4. NHS

### Source selection principles

Documents should be:

- authoritative
- medically relevant
- reasonably current
- accessible for verification
- attributable to the original source
- suitable for retrieval and citation

Random blogs, forums, social-media posts, commercial medical websites, and unverified articles should not be included in the primary medical corpus.

---

# 3. Knowledge Domains

## 3.1 Melanoma

Purpose:

- melanoma overview
- common warning signs
- suspicious visual characteristics
- ABCDE concepts
- evolving lesions
- reasons for professional evaluation
- general diagnosis-related information
- treatment/management information where relevant to patient questions

Priority:

**HIGH**

---

## 3.2 Suspicious Skin Lesions

Purpose:

- characteristics that may warrant professional assessment
- asymmetry
- border irregularity
- color variation
- diameter/size
- evolution over time
- changes in appearance
- changes in texture or sensation where supported by authoritative sources

Priority:

**HIGH**

---

## 3.3 Benign/Common Skin Lesions

Purpose:

- general information about common benign lesions
- characteristics that may distinguish common benign findings from concerning changes
- explanation of uncertainty

Important:

The assistant must not use this information to definitively rule out malignancy.

Priority:

**MEDIUM**

---

## 3.4 Skin Injuries

### Abrasions / Scratches

Purpose:

- basic wound care
- cleaning
- protecting the affected area
- signs of infection
- situations requiring professional evaluation

Priority:

**HIGH**

### Minor Cuts

Purpose:

- basic first aid
- bleeding control
- wound cleaning
- dressing
- infection warning signs
- escalation criteria

Priority:

**HIGH**

---

## 3.5 Burns

Purpose:

- basic first aid
- appropriate immediate actions
- actions to avoid
- severity indicators
- when professional/urgent medical care is appropriate

Priority:

**HIGH**

---

## 3.6 Infection Warning Signs

Purpose:

- redness
- swelling
- increasing pain
- discharge/pus
- fever
- spreading symptoms
- other authoritative warning signs

The exact criteria must be derived from the selected medical sources.

Priority:

**HIGH**

---

## 3.7 When to Seek Medical Attention

Purpose:

Support the Severity Engine and conversational assistant with authoritative information concerning:

- routine evaluation
- prompt professional evaluation
- urgent evaluation
- emergency situations

The RAG layer must not override the system's deterministic Severity Engine.

Priority:

**CRITICAL**

---

## 3.8 General Skin Health

Purpose:

- general skin care
- sun protection
- prevention-oriented information
- general patient education

Priority:

**LOW / MEDIUM**

---

# 4. Corpus Structure

The raw corpus should be organized as follows:

```text
data/
└── rag/
    ├── raw/
    │   ├── melanoma/
    │   ├── lesions/
    │   ├── wounds/
    │   ├── burns/
    │   ├── infection/
    │   ├── escalation/
    │   └── general/
    │
    ├── processed/
    │   ├── documents.jsonl
    │   └── chunks.jsonl
    │
    └── indexes/
        └── medical/