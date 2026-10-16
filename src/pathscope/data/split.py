"""Phase 1 split: queries vs corpus, with leakage removed and documented.

    queries  = PathVQA test images                      (what we describe and score)
    dev      = PathVQA validation images                (prompt tuning in Phase 4; never scored)
    corpus   = PathVQA train images + ARCH books + ARCH pubmed   (what we retrieve from)

Every image carries a `text` column: the ARCH caption, or the PathVQA pseudo-caption.
Writes data/manifests/split_v1/{queries,dev,corpus}.parquet and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pathscope import config
from pathscope.data.captions import build_pseudo_captions
from pathscope.data.leakage import LeakageConfig, run_leakage, within_corpus_duplicates

SPLIT_DIR = config.MANIFEST_DIR / "split_v1"


def _pathvqa_images(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """One row per unique image in a PathVQA split, with its pseudo-caption as `text`."""
    pc = build_pseudo_captions(manifest[manifest.split == split])
    first = (
        manifest[manifest.split == split]
        .drop_duplicates("image_id")[["image_id", "phash", "width", "height", "mode", "row_idx"]]
        .set_index("image_id")
    )
    df = pc.set_index("image_id").join(first, how="left").reset_index()
    df["source"] = f"pathvqa_{split}"
    df["uuid"] = None
    df["text"] = df.gold_text
    return df[
        [
            "source", "image_id", "uuid", "row_idx", "phash", "width", "height", "mode",
            "text", "gold_fragments", "neg_fragments", "n_qa",
        ]
    ]  # fmt: skip


def _arch_images(arch: pd.DataFrame) -> pd.DataFrame:
    ok = arch[arch.decode_error.isna()].copy()
    ok["text"] = ok.caption
    ok["row_idx"] = None
    ok["gold_fragments"] = None
    ok["neg_fragments"] = None
    ok["n_qa"] = None
    return ok[
        [
            "source", "image_id", "uuid", "row_idx", "phash", "width", "height", "mode",
            "text", "gold_fragments", "neg_fragments", "n_qa", "member", "figure_id", "letter",
        ]
    ]  # fmt: skip


def _dup_source_pairs(df: pd.DataFrame) -> dict:
    """Which sources share exact-duplicate images, e.g. {'arch_books|arch_books': 30}."""
    out: dict[str, int] = {}
    for _, g in df.groupby("image_id"):
        if len(g) < 2:
            continue
        srcs = sorted(g.source.tolist())
        key = "|".join(srcs[:2]) if len(set(srcs)) > 1 else f"{srcs[0]}|{srcs[0]}"
        out[key] = out.get(key, 0) + len(g) - 1
    return out


def build_split(
    pathvqa_manifest: pd.DataFrame,
    arch_manifest: pd.DataFrame,
    cfg: LeakageConfig | None = None,
    out_dir: Path = SPLIT_DIR,
) -> dict:
    cfg = cfg or LeakageConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _pathvqa_images(pathvqa_manifest, "train")
    dev = _pathvqa_images(pathvqa_manifest, "validation")
    test = _pathvqa_images(pathvqa_manifest, "test")
    arch = _arch_images(arch_manifest)

    # --- corpus: concat, drop exact duplicates (keep first = PathVQA train before ARCH so the
    # richer per-image Q/A stays attached), report within-corpus near-dups.
    corpus_raw = pd.concat([train, arch], ignore_index=True)
    n_raw = len(corpus_raw)
    corpus = corpus_raw.drop_duplicates("image_id").reset_index(drop=True)
    corpus["corpus_idx"] = range(len(corpus))
    within = within_corpus_duplicates(corpus, cfg.phash_threshold)
    within["exact_duplicate_rows_removed"] = n_raw - len(corpus)
    within["exact_duplicates_by_source_pair"] = _dup_source_pairs(corpus_raw)
    within["rows_before"] = n_raw
    within["rows_after"] = len(corpus)

    # Manual grey-zone adjudications, if the reviewer file exists.
    manual_path = out_dir / "manual_near_dups.csv"
    if manual_path.exists() and cfg.manual_near_dups is None:
        cfg.manual_near_dups = pd.read_csv(manual_path)

    # --- leakage of each query-like set against the corpus
    res_test = run_leakage(test, corpus, cfg, q_text_col="text", c_text_col="text")
    res_dev = run_leakage(dev, corpus, cfg, q_text_col="text", c_text_col="text")

    # --- dev vs test: dev is used for prompt tuning, so it must not contain test figures.
    # We drop the sibling from DEV (cheap: dev is only for tuning) and keep the test query.
    res_dev_vs_test = run_leakage(dev, test, cfg, q_text_col="text", c_text_col="text")

    queries = test.merge(res_test.flags, on="image_id", how="left")
    dev_out = dev.merge(res_dev.flags, on="image_id", how="left")
    dev_out["leaked_vs_test"] = res_dev_vs_test.flags.leaked.to_numpy()

    clean_q = queries[~queries.leaked].reset_index(drop=True)
    clean_dev = dev_out[~dev_out.leaked & ~dev_out.leaked_vs_test].reset_index(drop=True)

    # Write everything: full (with flags) and clean subsets.
    queries.to_parquet(out_dir / "queries_all_flags.parquet", index=False)
    dev_out.to_parquet(out_dir / "dev_all_flags.parquet", index=False)
    clean_q.to_parquet(out_dir / "queries.parquet", index=False)
    clean_dev.to_parquet(out_dir / "dev.parquet", index=False)
    corpus.to_parquet(out_dir / "corpus.parquet", index=False)

    summary = {
        "corpus": {
            "rows": int(len(corpus)),
            "by_source": corpus.source.value_counts().to_dict(),
            "within_corpus": within,
        },
        "queries_test": {
            **res_test.summary,
            "clean": int(len(clean_q)),
            "shrinkage_pct": round(100 * (1 - len(clean_q) / len(queries)), 2),
        },
        "dev_validation": {
            **res_dev.summary,
            "leaked_vs_test": int(dev_out.leaked_vs_test.sum()),
            "leaked_vs_test_only": int((dev_out.leaked_vs_test & ~dev_out.leaked).sum()),
            "clean": int(len(clean_dev)),
            "shrinkage_pct": round(100 * (1 - len(clean_dev) / len(dev_out)), 2),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
