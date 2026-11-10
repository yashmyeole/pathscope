# PathScope

**Do the captions of visually similar pathology images actually describe a new image?**

PathScope is a retrieval-augmented pathology image description system built to *measure* caption
transferability rather than assume it. About 11.6k pathology images are embedded with
[PLIP](https://huggingface.co/vinid/plip) and stored in Qdrant. For each test image, the five
nearest corpus images are retrieved, every one of their captions is broken into single claims in
three tiers, and each claim is checked against what PathVQA records about the test image:

| Tier | Contains | Expected to transfer? |
|---|---|---|
| **A, visual** | stain, magnification, organ, tissue architecture, cell morphology | yes |
| **B, diagnostic** | disease name, pathological process | unclear |
| **C, case-specific** | grade, stage, measurements, patient age/sex, margins | no |

The headline metric is the **transferable claim rate (TCR)**: supported / (supported +
contradicted), overall and per tier. Claims the evidence says nothing about are never counted as
wrong.

**Results:** [`docs/report.md`](docs/report.md). In short: 58% of checkable neighbour claims are
true of the new image, transfer drops below a coin flip after the second neighbour, and a
description model only does better than the captions alone when it can see the image and the
corpus has no copy of the same textbook figure.

> CS594 Vector Databases, University of Illinois Chicago, Fall 2026. Yash Manohar Yeole.
> Research project, not for clinical use.

## Status

| Phase | What | Output |
|---|---|---|
| 0 | Repo, Qdrant cluster and keep-alive, PathVQA statistics | `data/stats/pathvqa_stats.md` |
| 1 | Data split and leakage control (MD5, pHash, caption overlap) | `data/stats/phase1_leakage_report.md` |
| 2 | PLIP embeddings and Qdrant index | `pathscope embed / index / neighbors` |
| 3 | Retrieval sanity check | `eval/phase3/report.md` |
| 4 | Claim extraction, tiers, TCR, judge agreement | `eval/phase4/report.md` |
| 5 | Generation with image-blind and random-caption controls | `eval/phase5/report.md` |
| 6 | Next.js demo on Vercel | `app/` |
| 7 | Analysis and write-up | `docs/report.md` |

## Layout

```
pathscope/
├── src/pathscope/
│   ├── cli.py             # `pathscope <command>`, one per step below
│   ├── config.py          # paths and settings
│   ├── hashing.py         # pixel MD5 (image id) and pHash (near-duplicates)
│   ├── embed.py           # PLIP embeddings and the image-type classifier
│   ├── vectorstore.py     # Qdrant: create, upsert, search, neighbours
│   ├── retrieval.py       # retrieval checks on the stored vectors
│   ├── llm.py             # Groq / Gemini JSON clients with a response cache
│   ├── claims.py          # claim extraction, checking, TCR
│   ├── generate.py        # descriptions under the three conditions
│   ├── demo.py            # data and thumbnails for the demo
│   ├── report.py          # recomputes the results and renders docs/report.md
│   ├── prompts/           # extraction, checking and description prompts
│   └── data/              # PathVQA and ARCH loading, pseudo-captions, leakage checks, split
├── app/                   # Next.js demo
├── data/manifests/        # image manifests and split_v1 (committed)
├── data/stats/            # dataset statistics and the leakage report
├── eval/                  # results of phases 3 to 5
├── docs/                  # report, its template and figures
├── notebooks/             # first look at PathVQA
├── scripts/qdrant_ping.py # keep-alive ping used by the GitHub Action
└── tests/
```

## Running it

`uv run pathscope report` is enough to rebuild every number in the report from a clean clone.
The full pipeline, step by step (account setup is in `SETUP.md`):

```bash
# 1. Python env (uv: https://docs.astral.sh/uv/)
uv sync                      # core deps
uv sync --extra notebooks    # + jupyter
# torch and PLIP are only needed to re-embed (CPU is fine, about 20 min): uv sync --extra embed

# 2. Secrets
cp .env.example .env         # fill in QDRANT_URL / QDRANT_API_KEY

# 3. Phase 0
uv run pathscope manifest          # downloads PathVQA (~785 MB), hashes every image -> data/manifests/
uv run pathscope stats             # -> data/stats/pathvqa_stats.{json,md}
uv run pathscope contact-sheet     # -> data/stats/contact_sheets/*.png
uv run pathscope ping              # cluster responds?

# 4. Phase 1: put the two ARCH zips in data/raw/arch/ first (links below)
uv run pathscope arch-manifest     # hashes 7.6k ARCH images straight from the zips
uv run pathscope split             # leakage checks -> data/manifests/split_v1/
uv run pathscope leakage-report    # -> data/stats/phase1_leakage_report.md

# 5. Phase 2: needs the zips, the HF cache, torch, and QDRANT_URL / QDRANT_API_KEY in .env
uv run pathscope embed             # PLIP over corpus + test + dev -> data/embeddings/embeddings.parquet
uv run pathscope index             # upload (batches of 100), verify counts, self-retrieval check
uv run pathscope neighbors <image_id>   # 5 nearest corpus neighbours + captions

# 6. Phase 3 (works offline from embeddings.parquet, no cluster needed)
uv run pathscope retrieval-check --sheets   # -> eval/phase3/{metrics.json,neighbours.csv,sheets/}

# 7. Phase 4 (GROQ_API_KEY and GEMINI_API_KEY in .env; free tiers, rerun after a daily cap)
uv run pathscope claims --check   # extractor vs the 30 hand-written references
uv run pathscope claims           # all neighbour captions -> eval/phase4/claims.csv
uv run pathscope verify           # claims vs each query's Q/A -> eval/phase4/verdicts.csv
uv run pathscope tcr              # -> eval/phase4/tcr.json (+ judge_sample.csv to label)

# 8. Phase 5 (Gemini free tier: 500 requests/day, so this takes a few days; both steps resume)
uv run pathscope generate         # 250 queries x full / blind / random -> eval/phase5/descriptions.csv
uv run pathscope gen-score        # claims + checks -> eval/phase5/{verdicts.csv,results.json}

# 9. Phase 6 (ARCH images are read from the ARCH server if the zips aren't local)
uv run pathscope export-demo      # -> app/data/*.json and app/public/img/*.webp

# 10. Phase 7 (runs from a clean clone, no keys or downloads needed)
uv run pathscope report           # recompute every number, draw docs/figures/, render docs/report.md
```

The committed manifests mean Phases 2+ do **not** need the raw downloads to reproduce the split;
you only need the zips / HF cache to look at pixels (embedding, contact sheets).

See `SETUP.md` for the one-time account steps (Qdrant Cloud, GitHub secrets, HuggingFace email).

## Data

- **PathVQA** (`flaviagiammarino/path-vqa` on HuggingFace): 32,632 Q/A pairs over **4,288 unique images**
  (train 2,599 / val 832 / test 858). The HF version has no image ids and no captions, only `(image,
  question, answer)`, so images are identified by pixel MD5 and each gets a pseudo-caption from
  its non-yes/no answers. About half the images are micrographs and the rest are gross specimens, clinical
  photographs and diagrams. Statistics: `data/stats/pathvqa_stats.md`; observations:
  `data/stats/phase0_findings.md`.
- **ARCH** ([Gamper & Rajpoot, CVPR 2021](https://warwick.ac.uk/fac/cross_fac/tia/data/arch/)): the
  downloadable release has **7,579 images** (4,270 textbook figures with multi-panel "bags" + 3,309 PubMed
  figures), not the 15k in the paper; the download page notes the discrepancy. Real captions, median
  20 words. Corpus side only. CC BY-NC-SA 4.0. Download `books_set.zip` (5.3 GB) and `pubmed_set.zip`
  (0.5 GB) into `data/raw/arch/`.

### Split v1 (after leakage removal)

| Role | Source | Images |
|---|---|---|
| Queries (scored) | PathVQA test | 858 → **744** |
| Dev (prompt tuning only) | PathVQA validation | 832 → **699** |
| Corpus | PathVQA train 2,599 + ARCH 7,542 | **10,141** |

13% of test images had a near-duplicate or an identical caption in PathVQA train (none in ARCH).
Full method and sensitivity tables: `data/stats/phase1_leakage_report.md`.

## Demo

`app/` is a Next.js site with a gallery of the 250 Phase 5 test images. Each one shows its five
nearest corpus images and the three generated descriptions with every claim marked by tier and
by whether PathVQA supports it. Neighbours are fetched live from Qdrant through
`/api/neighbours/[id]`; if the cluster can't be reached, the page falls back to the saved ones.
Nothing calls an LLM at request time.

```bash
uv run pathscope export-demo      # data + thumbnails
cd app
cp .env.example .env.local        # QDRANT_URL / QDRANT_API_KEY
npm install
npm run dev                       # http://localhost:3000
```

Deploying to Vercel: see `SETUP.md`.

## Keep-alive

Qdrant Cloud free clusters suspend after 7 idle days and are deleted after 28.
`.github/workflows/qdrant-keepalive.yml` pings the cluster every 3 days and fails if it is
unreachable. It needs `QDRANT_URL` and `QDRANT_API_KEY` as repository secrets.

## License

MIT. Datasets retain their own licenses (PathVQA: MIT; ARCH: CC BY-NC-SA 4.0).
