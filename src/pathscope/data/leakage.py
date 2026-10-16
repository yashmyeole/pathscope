"""Phase 1 leakage checks between a query set and a corpus.

Three independent signals, each reported as its own rate (guide, Phase 1):

  1. exact      — identical decoded pixels (pixel MD5)
  2. near       — perceptual-hash near-duplicates (re-encoded / rescaled / lightly cropped copies)
  3. text       — the query image's own gold text appears (near-)verbatim as a corpus caption,
                  which means the corpus holds the same textbook figure or sentence and a
                  retrieval+verify loop would be grading against the answer key

A query image flagged by any check is removed from the evaluation set. We never silently drop:
every flag is written out with the matching corpus row so the decision can be audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# ----------------------------------------------------------------------------- pHash distance


def phash_matrix(hashes: pd.Series) -> np.ndarray:
    """Hex pHash strings -> (n, 32) uint8 array. Missing/None -> all-zero row (never matches)."""
    out = np.zeros((len(hashes), 32), dtype=np.uint8)
    for i, h in enumerate(hashes.tolist()):
        if isinstance(h, str) and len(h) == 64:
            out[i] = np.frombuffer(bytes.fromhex(h), dtype=np.uint8)
    return out


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_all_pairs(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Hamming distances between every row of a (n,32) and b (m,32) -> (n, m) uint16."""
    n, m = len(a), len(b)
    out = np.empty((n, m), dtype=np.uint16)
    for i in range(0, n, chunk):
        x = a[i : i + chunk, None, :] ^ b[None, :, :]  # (c, m, 32)
        out[i : i + chunk] = _POPCOUNT[x].sum(axis=2)
    return out


def nearest_phash(
    q_hashes: pd.Series, c_hashes: pd.Series, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """For each query, the k smallest Hamming distances and the corpus indices they came from."""
    D = hamming_all_pairs(phash_matrix(q_hashes), phash_matrix(c_hashes))
    # Rows with an all-zero (missing) query hash would match other missing rows at distance 0;
    # mark distances from missing hashes as max so they never flag.
    q_missing = ~q_hashes.map(lambda h: isinstance(h, str) and len(h) == 64).to_numpy()
    c_missing = ~c_hashes.map(lambda h: isinstance(h, str) and len(h) == 64).to_numpy()
    D[q_missing, :] = 256
    D[:, c_missing] = 256
    idx = np.argpartition(D, kth=min(k, D.shape[1] - 1), axis=1)[:, :k]
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1)
    return np.take_along_axis(dist, order, axis=1), np.take_along_axis(idx, order, axis=1)


# ----------------------------------------------------------------------------- text overlap


def longest_fragment(fragments, fallback: str | None) -> str:
    """The most caption-like piece of text we have for an image: the longest gold fragment for
    PathVQA (so organ-system labels like 'endocrine; endocrine system' do not inflate scores),
    or the caption itself for ARCH."""
    if fragments is not None and len(fragments):
        return max(fragments, key=lambda s: len(s.split()))
    return fallback or ""


def _norm_key(s: str) -> str:
    return " ".join(str(s).lower().replace("'", "").split())


def text_overlap(
    q_texts: pd.Series,
    c_texts: pd.Series,
    min_words: int = 6,
    scorer=fuzz.token_sort_ratio,
) -> pd.DataFrame:
    """Two text signals per query, both computed on `longest_fragment` text.

    * score / c_idx  — best fuzzy match (token_sort_ratio: order-insensitive but NOT subset-
      tolerant, unlike token_set_ratio which scores 100 whenever one side is a subset of the
      other) among texts with >= min_words on both sides. Short texts are ineligible: a
      five-word diagnosis label ("skin kaposis sarcoma") is a class name, not a caption.
    * label_match    — the query's short text (< min_words) is *identical* (normalised) to a
      corpus text. Reported, not removed: it marks queries whose exact diagnosis label has an
      exemplar in the corpus, which is a stratification variable for TCR, not leakage.
    """
    c_list = c_texts.fillna("").tolist()
    c_ok = [i for i, t in enumerate(c_list) if len(t.split()) >= min_words]
    choices = [c_list[i] for i in c_ok]
    short_index: dict[str, int] = {}
    for i, t in enumerate(c_list):
        k = _norm_key(t)
        if k and k not in short_index:
            short_index[k] = i
    rows = []
    for qi, qt in enumerate(q_texts.fillna("").tolist()):
        n_words = len(qt.split())
        if n_words < min_words or not choices:
            k = _norm_key(qt)
            # one-word labels ("liver", "its") are too generic to mean anything
            li = short_index.get(k, -1) if (k and n_words >= 2) else -1
            rows.append(
                {
                    "q_idx": qi,
                    "score": 0.0,
                    "c_idx": -1,
                    "eligible": False,
                    "label_match": li >= 0,
                    "label_c_idx": li,
                }
            )
            continue
        best = process.extractOne(qt, choices, scorer=scorer)
        rows.append(
            {
                "q_idx": qi,
                "score": float(best[1]),
                "c_idx": int(c_ok[best[2]]),
                "eligible": True,
                "label_match": False,
                "label_c_idx": -1,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- orchestration


@dataclass
class LeakageConfig:
    # pHash Hamming distance (of 256 bits). Chosen by inspecting every query/corpus pair up to
    # d=86 by eye (data/stats/contact_sheets/phase1_phash_pairs_*.png): all 49 pairs with d<=58
    # were the same photograph; 60..86 is a grey zone (12 same / 37 different) handled by the
    # manual adjudication file; random 256-bit hashes never come closer than ~86.
    phash_threshold: int = 58
    phash_grey_zone_max: int = 86
    text_threshold: float = 85.0  # rapidfuzz token_sort_ratio on the longest fragment
    text_min_words: int = 6
    extra_phash_report: tuple[int, ...] = field(default=(12, 24, 48, 58, 70, 86))
    extra_text_report: tuple[int, ...] = field(default=(80, 85, 90, 95, 100))
    manual_near_dups: pd.DataFrame | None = None  # columns: image_id, corpus_image_id, verdict


@dataclass
class LeakageResult:
    flags: pd.DataFrame  # one row per query image with all signals
    summary: dict


def run_leakage(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    cfg: LeakageConfig | None = None,
    q_text_col: str = "gold_text",
    c_text_col: str = "caption",
) -> LeakageResult:
    """queries / corpus: one row per image with columns image_id, phash, <text col>, plus any
    bookkeeping columns (source, uuid, split...). Returns per-query flags and a summary dict."""
    cfg = cfg or LeakageConfig()
    q = queries.reset_index(drop=True)
    c = corpus.reset_index(drop=True)

    # 1. exact
    c_by_md5 = c.groupby("image_id").indices
    exact_idx = q.image_id.map(lambda m: int(c_by_md5[m][0]) if m in c_by_md5 else -1)

    # 2. near
    dist, idx = nearest_phash(q.phash, c.phash, k=3)

    # 3. text — on the longest fragment, not the joined pseudo-caption
    q_key = pd.Series(
        [longest_fragment(f, t) for f, t in zip(q.get("gold_fragments", [None] * len(q)),
                                                q[q_text_col], strict=True)]
    )  # fmt: skip
    c_key = pd.Series(
        [longest_fragment(f, t) for f, t in zip(c.get("gold_fragments", [None] * len(c)),
                                                c[c_text_col], strict=True)]
    )  # fmt: skip
    tx = text_overlap(q_key, c_key, min_words=cfg.text_min_words)

    flags = pd.DataFrame(
        {
            "image_id": q.image_id,
            "text_key": q_key,
            "text_key_words": q_key.str.split().str.len(),
            "exact_dup": exact_idx >= 0,
            "exact_corpus_idx": exact_idx,
            "phash_dist": dist[:, 0],
            "phash_corpus_idx": idx[:, 0],
            "phash_dist_2nd": dist[:, 1] if dist.shape[1] > 1 else 256,
            "text_score": tx.score,
            "text_corpus_idx": tx.c_idx,
            "text_eligible": tx.eligible,
            "label_match": tx.label_match,
            "label_corpus_idx": tx.label_c_idx,
        }
    )
    flags["near_dup"] = flags.phash_dist <= cfg.phash_threshold
    # Grey zone: pairs a human has looked at. 'same' promotes to near_dup; anything in the zone
    # without a verdict is reported so the review can be extended.
    flags["grey_zone"] = (flags.phash_dist > cfg.phash_threshold) & (
        flags.phash_dist <= cfg.phash_grey_zone_max
    )
    flags["manual_verdict"] = None
    if cfg.manual_near_dups is not None and len(cfg.manual_near_dups):
        mv = cfg.manual_near_dups.set_index("image_id").verdict
        flags["manual_verdict"] = flags.image_id.map(mv)
        flags.loc[flags.grey_zone & (flags.manual_verdict == "same"), "near_dup"] = True
    flags["text_dup"] = flags.text_eligible & (flags.text_score >= cfg.text_threshold)
    flags["leaked"] = flags.exact_dup | flags.near_dup | flags.text_dup

    def _desc(i: int) -> dict:
        if i < 0:
            return {}
        r = c.iloc[i]
        return {
            "source": r.get("source"),
            "corpus_image_id": r.image_id,
            "corpus_text": str(c_key.iloc[i])[:160],
        }

    flags["phash_match"] = [_desc(int(i)) for i in flags.phash_corpus_idx]
    flags["text_match"] = [_desc(int(i)) for i in flags.text_corpus_idx]
    flags["label_match_desc"] = [_desc(int(i)) for i in flags.label_corpus_idx]

    n = len(flags)
    summary = {
        "n_queries": int(n),
        "n_corpus": int(len(c)),
        "config": {
            "phash_threshold": cfg.phash_threshold,
            "text_threshold": cfg.text_threshold,
            "text_min_words": cfg.text_min_words,
        },
        "exact": {"n": int(flags.exact_dup.sum()), "pct": round(100 * flags.exact_dup.mean(), 2)},
        "near": {
            "n": int(flags.near_dup.sum()),
            "pct": round(100 * flags.near_dup.mean(), 2),
            "n_excluding_exact": int((flags.near_dup & ~flags.exact_dup).sum()),
            "n_auto_at_threshold": int((flags.phash_dist <= cfg.phash_threshold).sum()),
            "n_manual_same": int((flags.grey_zone & (flags.manual_verdict == "same")).sum()),
            "grey_zone": {
                "range": [cfg.phash_threshold + 1, cfg.phash_grey_zone_max],
                "n_pairs": int(flags.grey_zone.sum()),
                "n_reviewed": int((flags.grey_zone & flags.manual_verdict.notna()).sum()),
                "n_same": int((flags.grey_zone & (flags.manual_verdict == "same")).sum()),
                "n_different": int((flags.grey_zone & (flags.manual_verdict == "different")).sum()),
            },
            "n_at_or_below": {
                str(t): int((flags.phash_dist <= t).sum()) for t in cfg.extra_phash_report
            },
        },
        "text": {
            "n": int(flags.text_dup.sum()),
            "pct": round(100 * flags.text_dup.mean(), 2),
            "n_eligible": int(flags.text_eligible.sum()),
            "pct_of_eligible": round(
                100 * flags.text_dup.sum() / max(1, flags.text_eligible.sum()), 2
            ),
            "n_at_or_above": {
                str(t): int((flags.text_eligible & (flags.text_score >= t)).sum())
                for t in cfg.extra_text_report
            },
            "sensitivity_min_words": {
                str(w): int(
                    ((flags.text_key_words >= w) & (flags.text_score >= cfg.text_threshold)).sum()
                )
                for w in (cfg.text_min_words, 8, 10)
            },
        },
        "label_match_kept": {
            "n": int(flags.label_match.sum()),
            "pct": round(100 * flags.label_match.mean(), 2),
            "note": "short diagnosis label identical to a corpus text; kept, used to stratify",
        },
        "any": {"n": int(flags.leaked.sum()), "pct": round(100 * flags.leaked.mean(), 2)},
        "clean_queries": int((~flags.leaked).sum()),
    }
    return LeakageResult(flags=flags, summary=summary)


def within_corpus_duplicates(corpus: pd.DataFrame, phash_threshold: int = 12) -> dict:
    """Exact and near duplicates *inside* the corpus (e.g. the same figure in ARCH books and
    PathVQA train). Exact ones are dropped by the split step; near ones are reported only —
    multi-panel ARCH figures legitimately look alike."""
    c = corpus.reset_index(drop=True)
    exact_groups = c.groupby("image_id").size()
    n_exact_extra = int((exact_groups - 1).clip(lower=0).sum())

    H = phash_matrix(c.phash)
    valid = c.phash.map(lambda h: isinstance(h, str) and len(h) == 64).to_numpy()
    near_pairs = 0
    chunk = 256
    for i in range(0, len(c), chunk):
        D = hamming_all_pairs(H[i : i + chunk], H)
        # only count j > i to avoid double counting / self
        rows = np.arange(i, min(i + chunk, len(c)))
        for r_local, r in enumerate(rows):
            if not valid[r]:
                continue
            d = D[r_local, r + 1 :]
            v = valid[r + 1 :]
            near_pairs += int(((d <= phash_threshold) & v).sum())
    return {
        "exact_duplicate_rows_removed": n_exact_extra,
        "near_duplicate_pairs_remaining": near_pairs,
        "phash_threshold": phash_threshold,
    }
