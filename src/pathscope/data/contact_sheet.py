"""Render a grid of sample images with their Q/A text — the 'look at twenty images yourself'
step from Phase 0, and reused in Phase 3 for query-vs-neighbour sheets."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import DatasetDict  # noqa: E402

from pathscope.data.pathvqa import _to_pil  # noqa: E402


def sample_images(manifest: pd.DataFrame, split: str, n: int, seed: int) -> pd.DataFrame:
    """n distinct images from a split, each with all of its Q/A pairs attached."""
    ids = (
        manifest[manifest.split == split]
        .drop_duplicates("image_id")
        .sample(n=n, random_state=seed)
        .image_id.tolist()
    )
    sub = manifest[(manifest.split == split) & manifest.image_id.isin(ids)]
    return sub


def render_contact_sheet(
    ds: DatasetDict,
    sample: pd.DataFrame,
    out_path: Path,
    cols: int = 4,
    max_qa: int = 3,
    title: str | None = None,
) -> Path:
    groups = list(sample.groupby("image_id", sort=False))
    n = len(groups)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.6))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (_image_id, g) in zip(axes, groups, strict=False):
        split, row_idx = g.iloc[0].split, int(g.iloc[0].row_idx)
        img = _to_pil(ds[split][row_idx]["image"]).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        lines = [f"{split} · {img.size[0]}×{img.size[1]} · {len(g)} Q"]
        for _, r in g.head(max_qa).iterrows():
            lines.append(textwrap.shorten(f"Q: {r.question}", 60))
            lines.append(textwrap.shorten(f"A: {r.answer}", 60))
        ax.set_title("\n".join(lines), fontsize=7, loc="left", family="monospace")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path
