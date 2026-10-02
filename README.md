# PathScope

**Do retrieved captions of visually similar pathology images actually describe a new image?**

PathScope is a retrieval-augmented pathology image description system built to *measure* caption
transferability rather than assume it. We embed ~18k pathology images with [PLIP](https://huggingface.co/vinid/plip),
store them in Qdrant, retrieve the five nearest neighbours for each query image, and then break every
retrieved caption into atomic claims sorted into three tiers:

| Tier | Contains | Expected to transfer? |
|---|---|---|
| **A — Visual** | stain, magnification, organ, tissue architecture, cell morphology | Yes |
| **B — Diagnostic** | disease name, pathological process | Unclear — the interesting case |
| **C — Case-specific** | grade, stage, measurements, patient age/sex, margins | No |

The headline metric is the **Transferable Claim Rate (TCR)**: the fraction of retrieved claims that are
actually true of the query image, reported overall and per tier.

> CS594 (Vector Databases), University of Illinois at Chicago · Fall 2026 · Yash Manohar Yeole
> **Research project — not for clinical use.**

## Status

| Phase | What | State |
|---|---|---|
| 0 | Foundations — repo, Qdrant cluster + keep-alive, PathVQA statistics | 🟡 in progress |
| 1 | Data & leakage control (MD5, pHash, caption overlap) | ⬜ |
| 2 | PLIP embeddings + Qdrant index | ⬜ |
| 3 | Retrieval sanity check (50 hand-inspected cases) | ⬜ |
| 4 | Claim extraction, tiering, TCR metric, judge agreement | ⬜ |
| 5 | Generation + image-blind / random-neighbour controls | ⬜ |
| 6 | Next.js demo on Vercel | ⬜ |
| 7 | Analysis & write-up | ⬜ |

Full plan: `docs/implementation_guide.md`.

## Layout

```
pathscope/
├── src/pathscope/         # importable package: data loading, hashing, vector store, CLI
│   ├── config.py          # all paths + env settings
│   ├── hashing.py         # pixel MD5 (image identity) and pHash (near-duplicates)
│   ├── vectorstore.py     # Qdrant wrapper: ping / create / upsert / search
│   ├── cli.py             # `pathscope manifest | stats | contact-sheet | ping`
│   └── data/              # pathvqa.py (manifest), stats.py, contact_sheet.py
├── scripts/               # standalone scripts (CI keep-alive ping)
├── notebooks/             # exploration + Kaggle GPU notebooks (embedding)
├── data/
│   ├── raw/               # HF cache, downloads   (git-ignored)
│   ├── manifests/         # small parquet tables  (committed)
│   └── stats/             # JSON/MD stats, contact sheets
├── eval/                  # Phase 4–5 pipelines and result tables
├── app/                   # Next.js demo (Phase 6)
└── .github/workflows/     # qdrant-keepalive.yml
```

## Setup

```bash
# 1. Python env (uv: https://docs.astral.sh/uv/)
uv sync                      # core deps
uv sync --extra notebooks    # + jupyter
# PLIP/torch are only needed on Kaggle: uv sync --extra embed

# 2. Secrets
cp .env.example .env         # fill in QDRANT_URL / QDRANT_API_KEY

# 3. Phase 0
uv run pathscope manifest          # downloads PathVQA (~785 MB), hashes every image -> data/manifests/
uv run pathscope stats             # -> data/stats/pathvqa_stats.{json,md}
uv run pathscope contact-sheet     # -> data/stats/contact_sheets/*.png  (look at these!)
uv run pathscope ping              # cluster responds?
```

See `SETUP.md` for the one-time account steps (Qdrant Cloud, GitHub secrets, HuggingFace email).

## Data

- **PathVQA** (`flaviagiammarino/path-vqa` on HuggingFace): 32,632 Q/A pairs over **4,288 unique images**
  (train 2,599 / val 832 / test 858). The HF version has no image ids and no captions — only `(image,
  question, answer)` — so we identify images by pixel MD5 and reconstruct a pseudo-caption per image from
  its non-yes/no answers. About half the images are micrographs; the rest are gross specimens, clinical
  photographs and diagrams. Statistics: `data/stats/pathvqa_stats.md`; observations:
  `data/stats/phase0_findings.md`.
- **ARCH** (Phase 1): 15k figure–caption pairs from textbooks and PubMed. Corpus side only.

## Keep-alive

Qdrant Cloud free clusters suspend after 7 idle days and are **deleted after 28**.
`.github/workflows/qdrant-keepalive.yml` pings the cluster every 3 days and fails loudly if it is
unreachable. Needs `QDRANT_URL` and `QDRANT_API_KEY` repository secrets.

## License

MIT. Datasets retain their own licenses (PathVQA: MIT; ARCH: CC BY-NC-SA 4.0).
