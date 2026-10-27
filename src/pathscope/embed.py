"""Phase 2: PLIP image embeddings for every split_v1 image, plus an image-type label.

Runs anywhere torch runs (CPU is fine: ~12k images). Chunks are checkpointed to disk, so an
interrupted run resumes instead of restarting. Output: data/embeddings/embeddings.parquet with
one row per image — role (corpus / test / dev), payload columns, and a 512-d unit vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from pathscope import config
from pathscope.data.pairs import ImageLoader
from pathscope.data.split import SPLIT_DIR

ROLE_FILES = {"corpus": "corpus", "test": "queries", "dev": "dev"}
CHUNK = 500

# Image type (Phase 3 stratifier, Phase 6 gallery filter): micrograph / photo (gross specimen or
# clinical photograph) / diagram (drawings, plots, X-rays). Zero-shot PLIP prompts failed (74% of
# the corpus came out "clinical"), so a logistic regression is fitted on the stored embeddings from
# 148 hand-labelled images (data/manifests/image_type_labels.csv). Stratified 6-fold CV accuracy
# is 94.6%; diagram recall is only 75%, so treat that class as approximate.
LABELS_CSV = config.MANIFEST_DIR / "image_type_labels.csv"


def _tensor(x):
    """transformers versions differ: feature getters return a tensor or a ModelOutput."""
    return x if hasattr(x, "norm") else x.pooler_output


def load_plip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    torch.manual_seed(config.SEED)
    kw = {"cache_dir": str(config.HF_CACHE_DIR)}
    model = CLIPModel.from_pretrained(config.PLIP_HF_ID, **kw).eval()
    proc = CLIPProcessor.from_pretrained(config.PLIP_HF_ID, **kw)
    return model, proc


def embed_images(model, proc, loader: ImageLoader, rows: pd.DataFrame, batch: int = 32):
    """Return unit-norm PLIP vectors [n, 512] for manifest `rows` the ImageLoader can open."""
    import torch

    vecs = []
    recs = rows.to_dict("records")
    for i in range(0, len(recs), batch):
        imgs = [loader.load(r) for r in recs[i : i + batch]]
        with torch.no_grad():
            px = proc(images=imgs, return_tensors="pt")["pixel_values"]
            f = _tensor(model.get_image_features(pixel_values=px))
        vecs.append(torch.nn.functional.normalize(f, dim=-1).numpy())
    return np.concatenate(vecs).astype("float32")


def label_image_types(df: pd.DataFrame, labels_csv=LABELS_CSV):
    """Fit on the hand-labelled subset of `df` (needs `embedding`, `image_id`); predict for all.
    Returns (labels, top-1 probability)."""
    from sklearn.linear_model import LogisticRegression

    lab = pd.read_csv(labels_csv).set_index("image_id").label
    X = np.stack(df.embedding.to_numpy())
    m = df.image_id.isin(lab.index).to_numpy()
    clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=config.SEED)
    clf.fit(X[m], lab.loc[df.image_id[m]].to_numpy())
    p = clf.predict_proba(X)
    return clf.classes_[p.argmax(1)].tolist(), p.max(1).tolist()


def run_embedding(limit: int | None = None, batch: int = 32) -> pd.DataFrame:
    model, proc = load_plip()
    loader = ImageLoader()
    ckpt = config.EMBED_DIR / "_chunks"
    ckpt.mkdir(exist_ok=True)
    parts = []
    for role, fname in ROLE_FILES.items():
        df = pd.read_parquet(SPLIT_DIR / f"{fname}.parquet")
        if limit:
            df = df.head(limit)
        outs = []
        for i in tqdm(range(0, len(df), CHUNK), desc=role):
            f = ckpt / f"{role}_{i // CHUNK:04d}.parquet"
            if f.exists() and not limit:
                outs.append(pd.read_parquet(f))
                continue
            rows = df.iloc[i : i + CHUNK]
            v = embed_images(model, proc, loader, rows, batch)
            out = pd.DataFrame({"image_id": rows.image_id.to_numpy()})
            out["embedding"] = list(v)
            if not limit:
                out.to_parquet(f, index=False)
            outs.append(out)
        emb = pd.concat(outs, ignore_index=True)
        meta = df[["image_id", "source", "text"]].copy()
        meta["figure_id"] = df["figure_id"] if "figure_id" in df else None
        meta["label_match"] = df["label_match"] if "label_match" in df else None
        meta["role"] = role
        parts.append(meta.merge(emb, on="image_id", how="left", validate="1:1"))
    full = pd.concat(parts, ignore_index=True)
    assert not full.embedding.isna().any(), "missing embeddings"
    full["image_type"], full["image_type_p"] = label_image_types(full)
    out_path = config.EMBED_DIR / ("embeddings_sample.parquet" if limit else "embeddings.parquet")
    full.to_parquet(out_path, index=False)
    return full
