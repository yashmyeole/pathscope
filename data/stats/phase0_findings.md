# Phase 0 findings — what PathVQA actually looks like

Date: 2 Oct 2026 · Source: `flaviagiammarino/path-vqa` (HF) · Numbers from `pathvqa_stats.json`
(regenerate with `pathscope manifest && pathscope stats`).

## The numbers

| | train | validation | test | total |
|---|---|---|---|---|
| Q/A rows | 19,654 | 6,259 | 6,719 | 32,632 |
| **Unique images** (pixel MD5) | **2,599** | **832** | **858** | **4,288** |
| Q/A per image | 7.6 | 7.5 | 7.8 | |
| Yes/no answers | 49.6% | 49.9% | 50.0% | 49.8% |
| Images with ≥1 open-ended answer | 60% | 59% | 60% | |

- **Exact duplicate images across splits: 0.** MD5 over decoded pixels finds no shared images between
  train/val/test. pHash near-duplicates still need checking in Phase 1 (pHash is already in the manifest).
- **Question text repeats heavily:** 67% of test rows use a question string that also appears in train,
  and 30% of unique test (question, answer) pairs appear verbatim in train. This is template reuse
  ("what is present?", "what does this image show?"), not image leakage — but it means any "VQA accuracy"
  number on this dataset is soft. Not our metric, good to know.
- Median answer is **1 word**. Open-ended answers average 6 words (p90 = 11, max 34). Joining all
  non-yes/no answers per image gives a "pseudo-caption" of ~10 words on average (median 7, max ~90).
- Images: median 792×528 px; 452 of 4,288 are stored as **CMYK** (10.5%) and 18 as greyscale. Convert to
  RGB before PLIP (the manifest code already does this for hashing).

## What the images are (from two 20-image contact sheets, seeds 594 and 7)

Roughly: **~45% H&E / other micrographs** (incl. blood smears, cytology, Gram-type stains),
**~35% gross specimen photographs** (organs on blue/black backgrounds, often with rulers),
**~15% clinical photographs of patients** (faces, hands, skin, mouths, external genitalia),
**~5% other** (diagrams, radiographs, a test tube of fluid). Sample sheets are in `contact_sheets/`.

Consequences:

1. **This is not a histology dataset; it is a pathology-textbook figure dataset.** PLIP was trained on
   microscopy from Twitter. Expect retrieval to behave very differently on gross specimens and clinical
   photos. Phase 3 should report retrieval quality *per image type*, which means we need an image-type
   label. Plan: a quick PLIP zero-shot classifier (`"a microscopy image"` / `"a gross specimen photograph"`
   / `"a clinical photograph of a patient"` / `"a diagram"`) spot-checked by hand on ~100 images.
2. **The demo gallery must be curated, not random.** Clinical photographs of patients (some sensitive) are
   fine for an offline evaluation but should not be on a public resume URL. Restrict the Phase 6 gallery to
   micrographs + gross specimens, selected with the image-type label above.
3. **Tier C ("case-specific") claims do exist in this data:** "small (700 g)", "premature 30-week gestation
   infant", "20 ml hemoperitoneum", "slides 7182 and 7183". Good — the tier framework has something to bite on.

## How the questions were generated (visible in the data, matters for Phase 4)

Every image carries a caption-derived cluster: one "what does this image show?" with the caption fragment
as the answer, a "where is this? / what is present?" with the organ system, a yes/no that re-asserts the
caption ("does this image show thyroid, follicular adenoma? yes"), and a **negative yes/no built by swapping
in a random subject** ("does peritoneum show thyroid, follicular adenoma? no", "does child show
mild-to-moderate acute rejection? no").

So for claim verification:

- **Gold text for a test image** = the union of its non-yes/no answers + the subjects of its *yes* questions.
  Do **not** mine the *no* questions — their subjects are random distractors.
- Yes/no rows are useless as captions but are a free source of **hard negatives** for the judge calibration
  set in Phase 4 (we know "peritoneum" is false for that image).

## Corpus size check

Guide assumed "~18k = PathVQA train + ARCH". PathVQA train is 2,599 unique images (not 19.6k — that was
Q/A rows), so corpus ≈ 2.6k + 15.2k ARCH ≈ **17.8k**. The estimate holds, but ARCH is 85% of the corpus,
and ARCH captions are real figure captions while PathVQA's are reconstructed pseudo-captions. Phase 3/4
should break results down by corpus source, and the Phase 1 caption-overlap check matters most *within*
PathVQA (same textbook → same caption fragments on both sides).

## Open questions to settle in Phase 1

- Should validation (832 images) join the corpus, the query set, or be held out for prompt tuning in
  Phase 4? Current lean: **prompt tuning** — we need a set to iterate claim-extraction prompts on that is
  not the one we report on.
- pHash threshold for "same figure": start at Hamming ≤ 12/256 and inspect the pairs near the boundary.
