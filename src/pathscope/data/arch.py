"""ARCH (Gamper & Rajpoot, CVPR 2021): figure–caption pairs from pathology textbooks and PubMed.

Distributed as two zips (books_set.zip ~5.3 GB, pubmed_set.zip ~0.5 GB) from
https://warwick.ac.uk/fac/cross_fac/tia/data/arch/ — CC BY-NC-SA 4.0, research use only.

Each zip holds `<set>/captions.json` (a dict of records with at least `caption` and `uuid`;
the books set also has `figure_id` + `letter` for multi-panel figures — a "bag" in the paper's
terminology, one caption shared by all panels) and `<set>/images/<uuid>.{jpg,png}`.

We never extract the zips: images are read straight from the archive, hashed, and only the
small per-image manifest is kept. Phase 2 re-reads from the zip on Kaggle in the same way.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from pathscope import config
from pathscope.hashing import phash, pixel_md5

ARCH_DIR = config.RAW_DIR / "arch"
ARCH_SETS = {"books": "books_set", "pubmed": "pubmed_set"}
ARCH_URLS = {
    "books": "https://warwick.ac.uk/fac/cross_fac/tia/data/arch/books_set.zip",
    "pubmed": "https://warwick.ac.uk/fac/cross_fac/tia/data/arch/pubmed_set.zip",
}


def zip_path(subset: str) -> Path:
    return ARCH_DIR / f"{ARCH_SETS[subset]}.zip"


def load_captions(zf: zipfile.ZipFile, subset: str) -> list[dict]:
    """Flatten captions.json into one dict per image with normalised keys:
    uuid, caption, figure_id (None for pubmed), letter (None for pubmed)."""
    prefix = ARCH_SETS[subset]
    raw = json.loads(zf.read(f"{prefix}/captions.json"))
    records = raw.values() if isinstance(raw, dict) else raw
    out = []
    for r in records:
        out.append(
            {
                "uuid": r["uuid"],
                "caption": (r.get("caption") or "").strip(),
                "figure_id": r.get("figure_id"),
                "letter": r.get("letter"),
            }
        )
    return out


def _image_index(zf: zipfile.ZipFile, subset: str) -> dict[str, str]:
    """uuid -> member name, for whatever extension the image has."""
    prefix = f"{ARCH_SETS[subset]}/images/"
    idx: dict[str, str] = {}
    for n in zf.namelist():
        if n.startswith(prefix) and not n.endswith("/") and "__MACOSX" not in n:
            stem = Path(n).stem
            idx[stem] = n
    return idx


def open_image(zf: zipfile.ZipFile, member: str) -> Image.Image:
    return Image.open(io.BytesIO(zf.read(member)))


def build_arch_manifest(
    out_path: Path | None = None,
    subsets: tuple[str, ...] = ("pubmed", "books"),
    compute_phash: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """One row per ARCH image.

    Columns: source ('arch_books' | 'arch_pubmed'), uuid, member, figure_id, letter,
             caption, image_id (pixel md5), phash, width, height, mode, decode_error
    """
    rows: list[dict] = []
    for subset in subsets:
        zp = zip_path(subset)
        if not zp.exists():
            raise FileNotFoundError(f"{zp} missing — download from {ARCH_URLS[subset]}")
        with zipfile.ZipFile(zp) as zf:
            caps = load_captions(zf, subset)
            idx = _image_index(zf, subset)
            if limit:
                caps = caps[:limit]
            missing = 0
            for r in tqdm(caps, desc=f"arch:{subset}", unit="img"):
                member = idx.get(r["uuid"])
                row = {
                    "source": f"arch_{subset}",
                    "uuid": r["uuid"],
                    "member": member,
                    "figure_id": r["figure_id"],
                    "letter": r["letter"],
                    "caption": r["caption"],
                    "image_id": None,
                    "phash": None,
                    "width": None,
                    "height": None,
                    "mode": None,
                    "decode_error": None,
                }
                if member is None:
                    missing += 1
                    row["decode_error"] = "no image for uuid"
                else:
                    try:
                        img = open_image(zf, member)
                        img.load()
                        row.update(
                            image_id=pixel_md5(img),
                            phash=phash(img) if compute_phash else None,
                            width=img.size[0],
                            height=img.size[1],
                            mode=img.mode,
                        )
                    except Exception as e:  # noqa: BLE001 — record, don't crash a 15k-image pass
                        row["decode_error"] = f"{type(e).__name__}: {e}"[:200]
                rows.append(row)
            if missing:
                print(f"[arch:{subset}] {missing} caption records had no image file")
    df = pd.DataFrame(rows)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
    return df
