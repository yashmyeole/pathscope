# PathScope — Implementation Guide, Version 1 (Mid-Level Build)

Yash Manohar Yeole · University of Illinois at Chicago · 1 October 2026

> This is the cut-down build. Everything removed from v1 is listed in Part 5 with a reason — deferred, not lost.
> The canonical copy lives in the Claude project (`claude/PathScope_Implementation_Guide.md`); this file is a
> snapshot so the repo is self-contained.

## Part 1 — The project in one page

**Problem.** Given a new pathology image with no caption, retrieve the most visually similar images from a
captioned corpus and borrow their captions. *How much of what those borrowed captions say is actually true of
the new image?*

**Why pathology is the hard case.** Everything looks similar (pink/purple blobs); similar-looking does not mean
same disease; and there is almost no text attached to real archives — which is exactly why borrowing would be
valuable if it worked.

**Our idea.** Break every retrieved caption into atomic claims and sort them into tiers:

| Tier | Contains | Expected to transfer? |
|---|---|---|
| A — Visual | stain type, magnification, organ, tissue architecture, cell shape, colour | Yes — properties of the image |
| B — Diagnostic | disease name, pathological process | Unclear — the interesting case |
| C — Case-specific | grade, stage, measurements, patient age/sex, margins | No — properties of the patient |

**Headline metric.** Transferable Claim Rate (TCR): fraction of retrieved claims actually true of the query
image, overall and per tier. The per-tier breakdown is the contribution.

**Why it matters.** MMed-RAG (ICLR 2025) excluded pathology from description generation citing "brief and
insufficient descriptions" — asserted, not measured. FactMM-RAG (NAACL 2025) explicitly invites extension to
histology. Natural-image work shows bad retrieval can make captioning up to 50% worse than none.

## Part 2 — What v1 is

| Component | v1 | Why |
|---|---|---|
| Encoder | PLIP only | professor's pick; free, ungated |
| Corpus | ~18k images (PathVQA train + ARCH) | embeds in minutes, fits free tiers |
| Vector DB | Qdrant Cloud free | 18k × 512-d = 36 MB |
| Claim extraction | LLM, three tiers | |
| Verification | gold caption + LLM judge, 50 hand-checked | |
| Controls | image-blind, random-neighbour | |
| Demo | gallery of pre-embedded images | upload is v2 |
| Reranker / expert review | not in v1 | |

**Stack (all free, no runtime GPU):** Kaggle T4 for one-time embedding → Parquet; Qdrant Cloud free; Next.js
API routes + frontend on Vercel; Groq (gpt-oss-120b) for generation; Gemini Flash free tier as judge (different
model family on purpose); Kaggle/Colab for experiments.

**Keep-alive warning.** Qdrant free clusters suspend after 7 idle days and are DELETED after 28. A GitHub
Action pinging every 3 days is a week-1 task.

**Two paths.** Offline (once, GPU): download → de-duplicate → PLIP embed → Parquet → Qdrant. Live (per demo
use): gallery image → stored embedding → Qdrant top-5 → Groq description → claims colour-coded by tier. The
live path never runs a neural network.

## Part 3 — Phases (10 weeks)

### Phase 0 — Foundations (wk 1, ~8h)
Repo with `data/ notebooks/ src/ app/ eval/`; Qdrant cluster + API key as GitHub secret; keep-alive action;
load PathVQA and **look at twenty images yourself**; count unique images, train/test question repetition,
yes/no fraction; switch HF primary email to UIC and request CONCH/UNI access.
**Finish line:** a PathVQA image loads in a notebook and the cluster answers a ping from GitHub Actions.
*Gotcha:* PathVQA mixes gross specimens, microscopy and diagrams — know that before interpreting any number.

### Phase 1 — Data and leakage control (wk 2, ~10h)
Download ARCH (15,164 images, ~5 GB). Queries = PathVQA test; corpus = PathVQA train + ARCH. MD5 on decoded
pixels (exact dupes), pHash (re-encoded/cropped copies), caption overlap (PathVQA Qs were generated from
captions). Remove failures, record the three rates.
**Finish line:** clean manifest + leakage report. *Gotcha:* the query set shrinks a lot — that is a finding.

### Phase 2 — Embeddings and index (wk 3, ~8h)
PLIP (`vinid/plip`) on Kaggle T4; embed corpus (~5 min GPU); save embeddings+captions+ids to Parquet; Qdrant
collection 512-d cosine; batch upload (100/batch, verify counts); embed queries too; search function; confirm
an image retrieves itself at rank 1.
**Finish line:** image id → 5 neighbours with captions in <1 s.

### Phase 3 — Retrieval sanity check (wk 4, ~8h)
50 queries × 5 neighbours as a contact sheet, one sentence per case; same-organ / same-stain / same-magnification
rates for top-1; compare to published PLIP numbers (R@10 ≈ 0.067 on textbook/PubMed figures).
**Finish line:** an evidenced statement of whether PLIP retrieval here is good, mediocre or bad.

### Phase 4 — Claims and TCR (wk 5–6, ~20h)
Claim-extraction prompt; tier-classification prompt; test on 30 captions by hand; verification step
(supports / contradicts / silent against the gold caption); run on the query set; TCR overall + per tier;
hand-label 50 decisions and compute Cohen's κ vs the judge.
**Finish line:** TCR table + agreement figure. *Gotchas:* rigid output schema + few-shot examples; never count
"silent" as false.

### Phase 5 — Generation and controls (wk 7, ~10h)
Groq generation from retrieved captions; image-blind control; random-neighbour control; score all three with the
Phase 4 pipeline; multimodal gain; copy rate.
**Finish line:** how much the image contributes and how much is copying. Near-zero gain is a real result.

### Phase 6 — Demo (wk 8, ~12h)
Next.js shell on Vercel first; one API route (image id → neighbours); gallery of ~200 test images; detail view;
generated description with tier-coloured claims linked to source neighbour; "research project, not for clinical
use" banner; confirm keep-alive still green.
**Finish line:** public URL, <2 s load, complete story. *Gotcha:* no upload/filters/animations until the report
is written.

### Phase 7 — Analysis and write-up (wk 9–10, ~16h)
Every figure/table from a script; bootstrap CIs; report; clean-clone reproduction; README; slides; resume bullet.
**Finish line:** every number regenerates from one command.

## Part 4 — Honest expectations

Near-certain: working search over ~18k images; live URL; TCR overall + per tier; image-blind baseline;
defensible report. Likely: clean tier separation; similarity shown to be a weak proxy for transferability; judge
agreement good for tiers A and C. Might not happen: a publishable paper from v1; reliable Tier-B numbers without
a pathologist; an exciting positive result. Time sinks: claim-extraction prompting (budget double), data
wrangling, demo polish. Effort: ~90–100 h over 10 weeks.

## Part 5 — Upgrade path

1. In-browser PLIP via ONNX Runtime Web (image upload, ~1 wk)
2. KEEP + QuiltNet encoders via Qdrant named vectors (~1 wk)
3. Corpus to 200k+ with a BIOMEDICA pathology slice (~1 wk)
4. Transferability reranker trained on v1 TCR data (~2 wk)
5. Full control suite: blank, shuffled, no-retrieval, position (~1 wk)
6. Hybrid dense + sparse retrieval (~1 wk)
7. Pathologist validation — ask Ron early

## Immediate next steps

1. HF primary email → UIC; request gated models.
2. Qdrant cluster + keep-alive action.
3. Load PathVQA, look at twenty images.
4. Ask about pathologist access.
