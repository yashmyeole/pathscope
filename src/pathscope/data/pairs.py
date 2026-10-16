"""Render query/corpus image pairs side by side — used to eyeball pHash near-duplicates when
choosing the Phase 1 threshold, and reusable in Phase 3 for query/neighbour sheets."""

from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from pathscope.data.arch import open_image, zip_path  # noqa: E402
from pathscope.data.pathvqa import _to_pil  # noqa: E402


class ImageLoader:
    """Open any image referenced by a split-manifest row, whatever its source."""

    def __init__(self, pathvqa_ds=None):
        self.ds = pathvqa_ds
        self._zips: dict[str, zipfile.ZipFile] = {}

    def _zip(self, subset: str) -> zipfile.ZipFile:
        if subset not in self._zips:
            self._zips[subset] = zipfile.ZipFile(zip_path(subset))
        return self._zips[subset]

    def load(self, row) -> Image.Image:
        src = row["source"] if isinstance(row, dict) else row.source
        if src.startswith("pathvqa_"):
            if self.ds is None:
                from pathscope.data.pathvqa import load_pathvqa

                self.ds = load_pathvqa()
            split = src.split("_", 1)[1]
            row_idx = int(row["row_idx"] if isinstance(row, dict) else row.row_idx)
            return _to_pil(self.ds[split][row_idx]["image"]).convert("RGB")
        if src.startswith("arch_"):
            subset = src.split("_", 1)[1]
            member = row["member"] if isinstance(row, dict) else row.member
            return open_image(self._zip(subset), member).convert("RGB")
        raise ValueError(f"unknown source {src}")


def render_pairs(
    pairs: list[tuple[pd.Series, pd.Series, str]],
    out_path: Path,
    loader: ImageLoader,
    title: str | None = None,
    per_row: int = 3,
) -> Path:
    """pairs: list of (left_row, right_row, label). Draws left/right tiles, `per_row` per row."""
    n = len(pairs)
    rows = -(-n // per_row)
    fig, axes = plt.subplots(rows, per_row * 2, figsize=(per_row * 2 * 3.2, rows * 3.6))
    axes = axes.reshape(rows, per_row * 2)
    for k, (a, b, label) in enumerate(pairs):
        r, c = divmod(k, per_row)
        for j, (row, who) in enumerate(((a, "Q"), (b, "C"))):
            ax = axes[r, 2 * c + j]
            ax.imshow(loader.load(row))
            ax.set_xticks([])
            ax.set_yticks([])
            txt = str(row.get("text", "") or "")
            ax.set_title(
                f"{who} {row['source']}\n"
                + textwrap.shorten(txt, 70)
                + (f"\n{label}" if j == 0 else ""),
                fontsize=6.5,
                loc="left",
                family="monospace",
            )
    for k in range(n, rows * per_row):
        r, c = divmod(k, per_row)
        axes[r, 2 * c].axis("off")
        axes[r, 2 * c + 1].axis("off")
    if title:
        fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path
