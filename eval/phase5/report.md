# Phase 5: generation and the two controls

250 test queries, each described three times, all scored. Numbers come from `results.json`
(`pathscope gen-score`). Claims and verdicts are in `verdicts.csv`, the descriptions in
`descriptions.csv`.

## Setup

- **Queries.** 250 of the 744 test queries, drawn at random (seed 594).
- **Generator.** `gemini-3.5-flash-lite` with `prompts/describe.txt`: 3 to 5 plain sentences on the
  kind of image, the organ or tissue, what is seen, and the likely diagnosis. The prompt says the
  reference captions may not apply.
- **Conditions.**
  - `full`: the image and its 5 retrieved captions.
  - `blind`: the same 5 captions and no image.
  - `random`: the image and 5 random corpus captions. Each one comes from the same source as the
    neighbour it replaces, so the caption style stays the same and only relevance changes.
- **Scoring.** The Phase 4 extractor and judge, unchanged. Each description is split into claims,
  and each claim is checked against the query's own PathVQA answers. 750 descriptions gave
  3,562 claims.
- **Gains.** Multimodal gain = TCR(full) − TCR(blind). Retrieval gain = TCR(full) − TCR(random).
  Both are paired by query, with bootstrap CIs over queries.
- **Copy rate.** The share of claims where at least 80% of the content words appear together in one
  input caption.

## Main result

| condition | TCR | 95% CI | decided | supported / description | contradicted / description | copy rate |
|---|---|---|---|---|---|---|
| full | **0.71** | 0.63–0.78 | 21% | 0.75 | 0.31 | 7.0% |
| random | 0.59 | 0.50–0.67 | 18% | 0.50 | 0.35 | 2.5% |
| blind | 0.47 | 0.39–0.55 | 27% | 0.54 | 0.60 | 17.8% |

| gain | all | A visual | B diagnostic |
|---|---|---|---|
| multimodal (full − blind) | **+0.23** [0.15, 0.32] | +0.24 [0.16, 0.34] | +0.19 [0.05, 0.32] |
| retrieval (full − random) | **+0.12** [0.03, 0.21] | +0.14 [0.05, 0.23] | +0.08 [−0.08, 0.23] |

The full system beats both controls, and both CIs exclude zero. The image matters about twice as
much as which captions it gets. Tier C has 12 to 28 claims per condition, too few to say anything,
just as in Phase 4.

**Generation filters the neighbours.** Raw neighbour claims held 58% of the time in Phase 4.
Descriptions written from those same neighbours, with the image, hold 71% of the time. The two
numbers aren't on identical claim sets (Phase 4 scored all 744 queries), but the gap is large. When
the model can see the image, it drops much of what the lower-ranked neighbours get wrong.

**Without the image, the model guesses.** Blind descriptions make twice as many contradicted
claims per description as full ones (0.60 vs 0.31). Copying isn't the main reason: blind claims
that were copied hold 42% of the time and the rest hold 49%. The model blends five captions that
disagree and commits to one answer. Two examples: for a breast invasive ductal carcinoma, the blind
description says "adenocarcinoma of the prostate". For a gross photo of lung with caseous necrosis,
it says "renal cell carcinoma". The full descriptions of both images get the organ right.

Blind descriptions are decided more often (27%) because they name an organ and a disease more
often. That's also why their floor (0.13) is close to the full system's (0.15) even though their
TCR is much lower.

## Label twins explain the multimodal gain

Phase 4 predicted that the image-blind control would do almost as well as the full system. That
holds for one group of queries and fails badly for the other:

| | queries | full | blind | random | multimodal gain | retrieval gain |
|---|---|---|---|---|---|---|
| label twin in corpus (`label_match`) | 82 | 0.70 | 0.69 | 0.62 | +0.01 [−0.12, 0.13] | +0.08 [−0.07, 0.22] |
| no label twin | 168 | 0.71 | **0.32** | 0.57 | **+0.39** [0.27, 0.49] | +0.14 [0.03, 0.26] |

When the corpus holds an image with the same short label (usually another picture from the same
textbook figure), the captions alone are enough, and seeing the image adds nothing measurable. For
the other two thirds of queries, blind descriptions are wrong more often than right. The full
system holds at 0.71 in both groups. The image is what keeps it there.

So the headline "+0.23 multimodal gain" is an average of two different situations. Phases 6 and 7
should report them separately.

## What retrieval adds

The `random` condition is roughly the model looking at the image on its own: it copies only 2.5%
of its claims and mostly ignores irrelevant captions. On top of that, the retrieved captions add
+0.12 TCR and about a quarter of a correct fact per description (0.75 vs 0.50 supported). They
barely change the number of wrong facts (0.31 vs 0.35). The gain is on visual claims (organ, tissue,
image type). On diagnostic claims it isn't significant, which fits Phase 4: only the top one or two
neighbours' diagnoses hold up.

## Copying

7% of full-system claims are copied from a caption. They hold slightly better than the rest (0.73
vs 0.70) and are decided far more often (45% vs 19%), because they are mostly the short organ and
disease labels that PathVQA answers state directly. Copying doesn't explain the full system's
score.

## Image kind

Multimodal gain by kind: micrographs +0.19 [0.08, 0.31] (130 queries), photos +0.27 [0.13, 0.41]
(95), diagrams +0.24 [−0.06, 0.56] (25). Photos gain the most, which fits Phase 3: their
neighbours are the least reliable, so a model that can't see the photo is misled most. The
intervals overlap, though.

## Same-case neighbours

Only 21 of the 250 queries were among the 50 reviewed by hand in Phase 3, and 4 of those have a
same-case sister in the top 5. That's too few to split on. The `label_match` split above is the
best proxy available here.

## How far to trust this

- **The generator and the judge are the same model.** The judge only sees the description's claims
  and the query's text answers, never the image, which limits self-preference. It is still the
  biggest weakness of this phase. Rescoring a sample with a judge from a different family would
  settle it.
- **The judge's agreement (κ = 0.82) was measured on caption claims, not generated ones.** Generated
  claims look the same (short, one fact each), but nobody has checked a sample of these verdicts.
- **About 80% of claims can't be decided.** PathVQA's answers say little, so TCR here is about
  organ, image type and the main diagnosis. Morphology and stain claims are mostly silent.
- **One generator, one prompt, temperature 0.** A stronger model could close the blind gap on the
  no-twin queries, or widen it.
- **The copy check is word overlap.** A paraphrased caption isn't counted as copied.

## What this means for Phases 6 and 7

- The demo should always give the generator the image, and show the neighbours with their ranks so
  a reader can see what was available.
- Report the label-twin split alongside every headline number. On PathVQA, much of what looks like
  caption transfer is the same textbook figure showing up twice.
- The most useful extra experiment for the write-up is a second judge on a sample of Phase 5
  verdicts, ideally with the same 50-item relabel as Phase 4.
