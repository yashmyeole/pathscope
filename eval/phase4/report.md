# Phase 4: claims and the transferable claim rate

All 744 test queries scored: 1,619 neighbour captions (16 had nothing usable) gave 5,413 claims,
which became 11,937 claim checks, since a claim is checked once for every query that retrieves its
caption. Numbers come from `tcr.json` (`pathscope tcr`).

## Pipeline

1. **Claims.** Each retrieved caption goes to `openai/gpt-oss-120b` on Groq with
   `prompts/extract.txt`: one fact per claim, each tagged A (visual), B (diagnostic) or C
   (case-specific).
2. **Check.** Every claim from a query's five neighbours is checked by `gemini-3.5-flash-lite`
   against that query's own PathVQA question/answer pairs (`prompts/verify.txt`): supported,
   contradicted or silent. PathVQA's "no" answers are left out of the evidence. The dataset makes
   them by pasting a random phrase from another caption into a true question, and in a first pass
   they caused false contradictions ("is hematoxylin-eosin seen in small arteries? no" on an H&E
   slide).
3. **TCR** = supported / (supported + contradicted). Silent claims are never counted as false.
   `decided` is the share of claims that could be checked at all. CIs resample whole queries.

Both models are on free tiers. The run took three days of Groq's daily token cap. Everything is
cached in `data/llm_cache/`, so rerunning `claims`, `verify` and `tcr` costs nothing.

## Is the extractor consistent?

Checked against hand-written claims for 30 captions taken from the dev queries' neighbours
(`extract_reference.csv`, 91 claims), after three prompt fixes (split joined facts, keep organ
system words, drop textbook background):

| recall | precision | tier agreement | mean claim-count difference |
|---|---|---|---|
| 0.85 | 0.87 | 0.97 | 0.4 per caption |

The reference was written by whoever tuned the prompt, so read this as an upper bound.

## Main result

| Tier | claims | supported | contradicted | silent | decided | TCR | 95% CI |
|---|---|---|---|---|---|---|---|
| A visual | 7,895 | 1,778 | 1,327 | 4,790 | 39% | **0.57** | 0.54–0.61 |
| B diagnostic | 3,572 | 393 | 248 | 2,931 | 18% | **0.61** | 0.56–0.67 |
| C case-specific | 470 | 12 | 12 | 446 | 5% | 0.50 | 0.29–0.71 |
| All | 11,937 | 2,183 | 1,587 | 8,167 | 32% | **0.58** | 0.55–0.61 |

Of the claims that can be checked, a little under 60% are true of the new image. Counting every
silent claim as a failure instead gives a floor of 0.18 overall (0.23 visual, 0.11 diagnostic).

**Tier C can't be measured on this data.** 95% of case-specific claims are silent, and the 24 that
could be checked are too few to say anything (CI 0.29–0.71). PathVQA records almost nothing about
the patient, so there is nothing to check age, sex or history against. This is a limit of the
evidence, not a finding that case-specific claims transfer half the time.

**Visual and diagnostic claims transfer at about the same rate.** We expected visual claims
to transfer clearly better. Here B is slightly higher than A and the intervals overlap. Two things
probably push A down. Most checkable A claims are about organ or organ system, the one thing a
PathVQA answer states directly, so they're the easiest claims to contradict. And B claims are only
decidable when the query's facts name the same disease, which is mostly when retrieval already
worked.

## Where it varies

**Rank.** Transfer falls steadily down the neighbour list:

| rank | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| A visual | 0.72 | 0.63 | 0.53 | 0.47 | 0.47 |
| B diagnostic | 0.86 | 0.69 | 0.45 | 0.46 | 0.42 |

The top neighbour's diagnostic claims hold 86% of the time. From rank 3 on it's worse than a coin
flip. Across queries, a neighbour's raw cosine score is only weakly related to how well it
transfers (Spearman ρ = 0.15 over 2,043 neighbour pairs with at least one decided claim). Rank
within a query is a much better signal than the similarity number itself.

**Image kind.**

| | micrograph | photo (gross or clinical) | diagram |
|---|---|---|---|
| A visual | 0.64 | 0.49 | 0.79 |
| B diagnostic | 0.72 | 0.47 | 0.50 (12 decided) |

Photos are the weak spot on both tiers. That matches Phase 3, where gross photos got matched on
background and specimen colour rather than organ.

**Neighbour source.** Claims from ARCH neighbours can almost never be checked (about 5%), and the
few that can are mostly contradicted (A 0.23–0.31, B 0.11–0.29). ARCH captions use a different
vocabulary from PathVQA's short answers, and they rarely come back for PathVQA queries anyway (12% of
neighbours). The overall numbers are essentially PathVQA-to-PathVQA transfer.

**label_match.** Queries that share an exact short label with some corpus image transfer better
(A 0.62 vs 0.53, B 0.70 vs 0.51) and are far more often decidable.

**Same-case neighbours** (the 50 queries reviewed in Phase 3). When a sister image of the same case
is in the top 5, diagnostic claims hold 91% of the time (21/23) against 50% (10/20) without one.
Visual: 0.61 vs 0.54. The samples are small, but they point the way Phase 3 warned: part of the
transfer we measure is the same case showing up twice, not a similar image being described
correctly.

## How far to trust the judge

`judge_sample.csv` has 50 judge decisions: 20 supported, 16 contradicted, 14 silent. The sample
oversamples the supported and contradicted calls, since a random draw would be almost all silent.
They were labelled again without the judge's answer, using the same rules and evidence the judge
gets. Medical terms were looked up where the call depended on them, for example whether the
appendix counts as colon (it is a diverticulum of the colon, so "the organ is the colon" stays
silent rather than contradicted), whether lipochrome is the same as lipid (it isn't), and whether
gliosis is specific to the nervous system (it is).

| | judge supported | judge contradicted | judge silent |
|---|---|---|---|
| reference supported | 20 | 0 | 0 |
| reference contradicted | 0 | 15 | 5 |
| reference silent | 0 | 1 | 9 |

Agreement is 44/50, **Cohen's κ = 0.82**. The judge never disagrees on "supported". All six
disagreements are about contradictions, and five of them go the same way: the facts name a
different organ inside a longer phrase ("adrenal, pheochromocytoma", "opened muscle probably
psoas", "duodenal biopsy", "distal esophagus", "the bone is expanded") and the judge calls the claim
silent instead of contradicted. So the judge under-counts contradictions a little, which makes the
TCR numbers above slightly optimistic, not pessimistic.

**These reference labels are not an expert's.** They were made by the same assistant that built the
pipeline, with web lookups for the medical terms, so κ = 0.82 is agreement between two careful
non-experts reading the same short facts. That's fine for the visual organ and organ-system claims
that make up most of the decided set, where the call is mostly about reading. It is not a substitute
for a pathologist on the diagnostic tier. If someone in the department can relabel the
`label_yash` column, the score is computed automatically next to this one.

## What this means for Phase 5

- Generation should lean on the top one or two neighbours. Claims from ranks 3–5 are as likely to
  be wrong as right.
- Expect the image-blind control to do almost as well as the full system on PathVQA queries. Most
  of what transfers is organ and disease labels from near-identical textbook figures.
- Keep reporting results split by image kind and with and without same-case neighbours.
