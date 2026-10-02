"""Generates notebooks/00_pathvqa_explore.ipynb. Kept as a script so the notebook is reproducible
and diff-able; run `python notebooks/build_notebook_00.py` after editing."""

import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# 00 · PathVQA exploration (Phase 0)

Goal of this notebook, straight from the implementation guide:

1. Load PathVQA from HuggingFace.
2. **Look at twenty images and their captions with your own eyes.**
3. Count the basics: unique images, question repetition between train and test, yes/no fraction.

Runs on Kaggle / Colab / locally. On Kaggle, turn Internet **on** in the notebook settings.
""")

code("""
# --- environment ---------------------------------------------------------------------------
import sys, subprocess, os
IN_KAGGLE = os.path.exists("/kaggle")
IN_COLAB = "google.colab" in sys.modules

if IN_KAGGLE or IN_COLAB:
    # fresh machine: grab the repo so `pathscope` is importable
    if not os.path.exists("pathscope"):
        subprocess.run(["git", "clone", "-q", "https://github.com/yashmyeole/pathscope.git"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "pathscope"], check=True)
    os.chdir("pathscope")
else:
    # local: assume the notebook lives in <repo>/notebooks/
    os.chdir(os.path.join(os.path.dirname(os.path.abspath("__file__")), ".."))
print("cwd:", os.getcwd())
""")

code("""
import pandas as pd
from pathscope import config
from pathscope.data.pathvqa import load_pathvqa, build_manifest
from pathscope.data.stats import compute_stats, stats_markdown
from pathscope.data.contact_sheet import sample_images, render_contact_sheet
from IPython.display import Image as IPyImage, Markdown, display

ds = load_pathvqa()
ds
""")

md("""
## 1. What does one row look like?

Note what is **not** here: no image id, no caption, no source. Just `(image, question, answer)`.
PathVQA was generated from textbook captions, but the HF release only ships the derived Q/A pairs.
""")

code("""
ex = ds["train"][0]
print("question:", ex["question"])
print("answer:  ", ex["answer"])
print("image:   ", ex["image"].size, ex["image"].mode)
ex["image"]
""")

md("""
## 2. Build the manifest

Decode every image once, hash the pixels (MD5 = identity, pHash = near-duplicate), and write one
flat parquet row per Q/A pair. ~6 minutes for all 32.6k rows on a laptop CPU.
Set `LIMIT` to e.g. 500 for a quick first pass.
""")

code("""
LIMIT = None   # rows per split; None = everything
manifest_path = config.MANIFEST_DIR / "pathvqa_manifest.parquet"

if manifest_path.exists() and LIMIT is None:
    m = pd.read_parquet(manifest_path)
    print("loaded cached manifest:", len(m), "rows")
else:
    m = build_manifest(ds, out_path=manifest_path if LIMIT is None else None, limit_per_split=LIMIT)
m.head()
""")

md("""
## 3. Look at twenty images

Do not skip this. PathVQA mixes H&E microscopy, gross specimen photographs, radiographs, and
diagrams. Change the seed and run again a few times.
""")

code("""
SEED = config.SEED
sample = sample_images(m, split="train", n=20, seed=SEED)
out = render_contact_sheet(
    ds, sample, config.STATS_DIR / "contact_sheets" / f"pathvqa_train_20_seed{SEED}.png",
    title=f"PathVQA train — 20 random images (seed {SEED})",
)
display(IPyImage(filename=str(out)))
""")

code("""
# Same, but from the TEST split — these are our future query images.
sample_te = sample_images(m, split="test", n=20, seed=SEED)
out_te = render_contact_sheet(
    ds, sample_te, config.STATS_DIR / "contact_sheets" / f"pathvqa_test_20_seed{SEED}.png",
    title=f"PathVQA test — 20 random images (seed {SEED})",
)
display(IPyImage(filename=str(out_te)))
""")

md("""
### Notes from eyeballing (fill in)

- Fraction that are H&E microscopy vs gross specimens vs other: …
- Image quality / resolution range: …
- Do yes/no questions carry any descriptive content? …
- Anything that would confuse PLIP (diagrams, text overlays, arrows)? …
""")

md("""
## 4. Count the basics
""")

code("""
stats = compute_stats(m)
display(Markdown(stats_markdown(stats)))
""")

code("""
# Per-image question count — how many Q/A pairs does a typical image carry?
qpi = m.groupby(["split", "image_id"]).size().rename("n_q").reset_index()
qpi.groupby("split").n_q.describe()
""")

code("""
# How templated are the questions? Top 20 question strings.
m.q_norm.value_counts().head(20)
""")

code("""
# Open-ended answers are our only caption-like text. Length distribution in words:
open_ans = m[m.answer_type == "open"].answer.str.split().str.len()
ax = open_ans.clip(upper=30).hist(bins=30, figsize=(7, 3))
ax.set_title("Open-ended answer length (words, clipped at 30)")
ax.set_xlabel("words"); ax.set_ylabel("count");
""")

md("""
## 5. Leakage preview (full treatment in Phase 1)

Exact-duplicate images across splits, found by pixel MD5. Phase 1 adds pHash near-duplicates
and caption/answer overlap.
""")

code("""
by = {s: set(g.image_id) for s, g in m.groupby("split")}
for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]:
    inter = by[a] & by[b]
    print(f"{a:>10} ∩ {b:<10}: {len(inter):4d} exact-duplicate images "
          f"({100*len(inter)/len(by[b]):.1f}% of {b})")
""")

md("""
## 6. What this means for the build

Write 3–5 sentences here once the numbers are in:

- Query set size after removing duplicates: …
- Corpus side from PathVQA train: … images; ARCH adds ~15k.
- Where the "gold caption" for verification will come from (open-ended answers? ARCH captions?): …
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).with_name("00_pathvqa_explore.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
