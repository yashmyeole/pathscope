"""Central paths and settings. Everything else imports from here so nothing hard-codes a path."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = two levels above src/pathscope/
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HF_CACHE_DIR = RAW_DIR / "hf_cache"
MANIFEST_DIR = DATA_DIR / "manifests"
STATS_DIR = DATA_DIR / "stats"
EMBED_DIR = DATA_DIR / "embeddings"

for _d in (RAW_DIR, MANIFEST_DIR, STATS_DIR, EMBED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Datasets
PATHVQA_HF_ID = "flaviagiammarino/path-vqa"
PATHVQA_SPLITS = ("train", "validation", "test")

# Encoder
PLIP_HF_ID = "vinid/plip"
EMBED_DIM = 512

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pathscope_plip_v1")

# Reproducibility
SEED = 594
