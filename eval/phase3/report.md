# Phase 3: is PLIP retrieval any good on this corpus?

Short answer: **mediocre.** PLIP almost always returns the same *kind* of picture and gets the
organ right a bit over half the time, but a lot of that comes from photo style and from sister
images of the same case, not from understanding the tissue.

Numbers below come from `metrics.json` (`pathscope retrieval-check`). Queries are the 744 clean
PathVQA test images, corpus is the 10,141 images from split_v1, k = 5, exact cosine on the PLIP
vectors.

## 1. The 50 cases looked at by hand

50 random test queries (seed 594), sheets in `sheets/queries_01..10.png`, one line per case in
`review.csv`. The review is a first pass and should be checked; the `reviewer` column says who
did it.

| | |
|---|---|
| Related (top neighbours share organ or the main finding) | 25 / 50 |
| Loosely related (same kind of image, partial overlap) | 16 / 50 |
| Unrelated | 9 / 50 |
| Top-1 is the same kind of image (micrograph / gross / clinical / diagram) | 50 / 50 |
| Top-1 is the same organ | 30 / 47 (64%) |
| Top-1 has the same stain (micrographs only) | 18 / 24 (75%) |
| Top-1 at a similar magnification (micrographs only) | 22 / 24 (92%) |
| A neighbour in the top 5 looks like the **same case** (sister panel, same specimen) | 12 / 50 (24%) |
| Neighbours (of 5) that are actually related, on average | 2.0 |

Micrographs do better than photos: 14/24 micrographs were related against 10/23 gross and
clinical photos. Seven of the nine unrelated cases are gross photos, and there the failure is
very consistent: the neighbours share the **background** (blue towel, black cloth, cyan board)
and the overall colour of the specimen, not the organ. Q13 (myeloma in the sternum) comes back
with larynx, nose, peritoneum and breast, all photographed on the same towel. For micrographs the
misses are colour driven too: special stains get matched to anything with the same colour (a
Warthin-Starry silver stain to brown IHC, Congo red to other two-panel pink figures).

## 2. Automatic checks over all 744 queries

| | rate | chance |
|---|---|---|
| Top-1 shares an organ with the query (organ words in both captions, n = 381) | 56.7% | 6.5% |
| Any of the top 5 shares an organ (n = 470) | 69.1% | |
| Top-1 has the same predicted image type | 96.2% | 47.8% |

The image type agreement is optimistic, since those labels come from a classifier trained on these
same embeddings. The organ check is crude (keyword list in `src/pathscope/organs.py`, about two
thirds of queries get a tag) but it is independent of the image.

Organ agreement splits the same way as the hand review: micrographs 68%, photos 43%. It is also
much higher for `label_match` queries, the ones that share an exact short label with some corpus
image: 71% against 46%.

**Where the neighbours come from.** 87.9% of all top-5 neighbours are PathVQA train images, though
they are only 25.6% of the corpus. ARCH books and PubMed (74% of the corpus) supply 12%. PathVQA
test and train come from the same textbooks, photographed and compressed the same way, so the
nearest neighbour is mostly "another picture from the same book". This matters for Phase 4: the
captions being transferred are almost all short PathVQA pseudo-captions, not ARCH captions.

**Label twins.** For the 237 `label_match` queries, the corpus image with the identical label sits
at median rank 105 out of 10,141 (random would be about 5,070). It is in the top 5 for 27% of
them and in the top 50 for 44%.

## 3. Against published PLIP numbers

The cleanest comparison is cross-modal retrieval on ARCH, which is what Table 2 of the Quilt-1M
paper (Ikezogwo et al., NeurIPS 2023) reports for PLIP. Same model (`vinid/plip`, ViT-B/32), our
7,542 ARCH images and 6,459 unique captions; a book caption covers all panels of its figure.

| | R@1 | R@10 | R@50 | R@200 |
|---|---|---|---|---|
| image to text, ours | 0.74 | 5.06 | 14.70 | 31.24 |
| image to text, Quilt-1M paper | 0.74 | n/a | 13.75 | 29.46 |
| text to image, ours | 0.68 | 5.25 | 14.91 | 33.18 |
| text to image, Quilt-1M paper | 0.56 | n/a | 13.10 | 29.85 |

(percent; the paper reports R@1/50/200 only)

We land on top of the published numbers, slightly higher, which fits our pool being a bit smaller
than theirs (about 7.5k pairs). Two things follow. The embeddings are fine, there's no bug
dragging retrieval down. And PLIP really is weak on textbook and PubMed figures: the right caption
is in the top 10 about 5% of the time.
Book figures are easier than PubMed (text to image R@10 10.3% vs 4.9%), probably because many book
captions describe several panels.

## 4. What this means going forward

- Retrieval is good enough to be worth studying, not good enough to trust. Roughly half the
  neighbours a query gets are genuinely related; the rest share colour, background or layout.
- **Same-case sisters are the biggest caveat.** 24% of the reviewed queries have a sister image
  of the same case in their top 5 even after the Phase 1 leakage removal. pHash catches
  re-photographs of the same picture, not a second photo of the same specimen. Case-specific
  claims (Tier C) will transfer for those queries for the wrong reason, so Phase 4 should report
  TCR with and without them. A cheap automatic proxy is `label_match` plus a high top-1 score.
- Report everything split by image kind (micrograph vs photo) and by neighbour source. Gross photos
  will probably look worse than micrographs on every metric.
- Nobody should read a lot into "same kind of image 100%": the classifier work in Phase 2 showed
  PLIP separates micrographs, photos and diagrams easily, so that is close to free.

Sources: Ikezogwo et al., "Quilt-1M: One Million Image-Text Pairs for Histopathology",
https://arxiv.org/abs/2306.11207 (Table 2).
