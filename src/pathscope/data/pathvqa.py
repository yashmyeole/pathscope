"""Load PathVQA from HuggingFace and build a flat per-question manifest with stable image ids.

Why a manifest: the HF dataset stores the image bytes inline in every Q/A row. We decode each
image once, hash it, and write a lightweight table (image_id, split, question, answer, ...) so
every downstream step works from a small parquet instead of re-decoding 5 GB of images.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset
from PIL import Image
from tqdm import tqdm

from pathscope import config
from pathscope.hashing import phash, pixel_md5

YES_NO = {"yes", "no"}


def load_pathvqa(cache_dir: Path | None = None) -> DatasetDict:
    """Download (or reuse cached) PathVQA. ~785 MB compressed."""
    return load_dataset(config.PATHVQA_HF_ID, cache_dir=str(cache_dir or config.HF_CACHE_DIR))


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation/whitespace — for overlap counts, not for display."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s)


def answer_type(answer: str) -> str:
    a = normalize_text(answer)
    if a in YES_NO:
        return "yes_no"
    if len(a.split()) <= 2:
        return "short"
    return "open"


def _to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict) and "bytes" in x:
        return Image.open(io.BytesIO(x["bytes"]))
    raise TypeError(f"Unexpected image type: {type(x)}")


def build_manifest(
    ds: DatasetDict,
    out_path: Path | None = None,
    splits: tuple[str, ...] = config.PATHVQA_SPLITS,
    compute_phash: bool = True,
    limit_per_split: int | None = None,
) -> pd.DataFrame:
    """One row per (image, question, answer). Images are identified by pixel md5.

    Columns: split, row_idx, image_id, phash, width, height, mode, question, answer,
             answer_type, q_norm, a_norm
    """
    rows: list[dict] = []
    for split in splits:
        d = ds[split]
        n = len(d) if limit_per_split is None else min(limit_per_split, len(d))
        # md5 is cheap next to JPEG decode, so compute it every row; cache the pHash per
        # image_id since consecutive rows usually share an image.
        phash_cache: dict[str, str | None] = {}
        for i in tqdm(range(n), desc=f"manifest:{split}", unit="row"):
            ex = d[i]
            img = _to_pil(ex["image"])
            image_id = pixel_md5(img)
            if image_id not in phash_cache:
                phash_cache[image_id] = phash(img) if compute_phash else None
            q, a = ex["question"], ex["answer"]
            rows.append(
                {
                    "split": split,
                    "row_idx": i,
                    "image_id": image_id,
                    "phash": phash_cache[image_id],
                    "width": img.size[0],
                    "height": img.size[1],
                    "mode": img.mode,
                    "question": q,
                    "answer": a,
                    "answer_type": answer_type(a),
                    "q_norm": normalize_text(q),
                    "a_norm": normalize_text(a),
                }
            )
    df = pd.DataFrame(rows)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
    return df


def export_unique_images(
    ds: DatasetDict,
    manifest: pd.DataFrame,
    out_dir: Path,
    splits: tuple[str, ...] = config.PATHVQA_SPLITS,
    fmt: str = "png",
) -> int:
    """Write each unique image once to out_dir/<split>/<image_id>.<fmt>. Returns count written.
    Used in Phase 2 so the embedding notebook iterates over files, not Q/A rows."""
    written = 0
    for split in splits:
        sub = manifest[manifest.split == split].drop_duplicates("image_id")
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        d = ds[split]
        for row_idx, image_id in tqdm(
            zip(sub.row_idx, sub.image_id, strict=True), total=len(sub), desc=f"export:{split}"
        ):
            p = out_dir / split / f"{image_id}.{fmt}"
            if p.exists():
                continue
            _to_pil(d[int(row_idx)]["image"]).convert("RGB").save(p)
            written += 1
    return written
