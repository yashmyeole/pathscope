"""Phase 0 dataset statistics for PathVQA, computed from the manifest.

The guide asks three questions explicitly — unique images, question repetition between
train and test, yes/no fraction — plus a few extras that matter for later phases
(images shared across splits = leakage preview; answer-length distribution = how much
'caption-like' text we actually have to verify claims against).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def compute_stats(m: pd.DataFrame) -> dict:
    out: dict = {"source": "flaviagiammarino/path-vqa", "splits": {}}

    for split, g in m.groupby("split", sort=False):
        n_rows = len(g)
        n_img = g.image_id.nunique()
        at = g.answer_type.value_counts()
        alen = g.answer.str.split().str.len()
        out["splits"][split] = {
            "qa_rows": int(n_rows),
            "unique_images": int(n_img),
            "questions_per_image_mean": round(n_rows / n_img, 2) if n_img else None,
            "unique_questions_text": int(g.q_norm.nunique()),
            "unique_answers_text": int(g.a_norm.nunique()),
            "answer_type_counts": {k: int(v) for k, v in at.items()},
            "yes_no_pct": _pct(int(at.get("yes_no", 0)), n_rows),
            "answer_len_words": {
                "mean": round(float(alen.mean()), 2),
                "median": float(alen.median()),
                "p90": float(alen.quantile(0.9)),
                "max": int(alen.max()),
            },
            "image_size_px": {
                "width_median": float(g.width.median()),
                "height_median": float(g.height.median()),
                "width_min": int(g.width.min()),
                "width_max": int(g.width.max()),
            },
        }

    # --- cross-split overlap -------------------------------------------------------------
    by_split = {s: g for s, g in m.groupby("split")}
    tr, te = by_split.get("train"), by_split.get("test")
    va = by_split.get("validation")

    def q_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
        qa, qb = set(a.q_norm), set(b.q_norm)
        inter = qa & qb
        b_rows_hit = int(b.q_norm.isin(inter).sum())
        return {
            "shared_question_strings": len(inter),
            "pct_of_b_unique_questions_seen_in_a": _pct(len(inter), len(qb)),
            "pct_of_b_rows_with_question_seen_in_a": _pct(b_rows_hit, len(b)),
        }

    def qa_pair_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
        pa = set(zip(a.q_norm, a.a_norm, strict=True))
        pb = set(zip(b.q_norm, b.a_norm, strict=True))
        inter = pa & pb
        return {
            "shared_qa_pairs": len(inter),
            "pct_of_b_unique_pairs_seen_in_a": _pct(len(inter), len(pb)),
        }

    def img_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
        ia, ib = set(a.image_id), set(b.image_id)
        inter = ia & ib
        return {
            "shared_exact_images": len(inter),
            "pct_of_b_images_also_in_a": _pct(len(inter), len(ib)),
        }

    out["overlap"] = {}
    if tr is not None and te is not None:
        out["overlap"]["train_vs_test"] = {
            "questions": q_overlap(tr, te),
            "qa_pairs": qa_pair_overlap(tr, te),
            "images_exact_md5": img_overlap(tr, te),
        }
    if tr is not None and va is not None:
        out["overlap"]["train_vs_validation"] = {
            "questions": q_overlap(tr, va),
            "images_exact_md5": img_overlap(tr, va),
        }
    if va is not None and te is not None:
        out["overlap"]["validation_vs_test"] = {
            "images_exact_md5": img_overlap(va, te),
        }

    # --- whole-dataset view ---------------------------------------------------------------
    out["total"] = {
        "qa_rows": int(len(m)),
        "unique_images_all_splits": int(m.image_id.nunique()),
        "yes_no_pct": _pct(int((m.answer_type == "yes_no").sum()), len(m)),
    }

    # Most common questions overall — tells you how templated the dataset is.
    top_q = m.q_norm.value_counts().head(15)
    out["top_questions"] = [{"question": q, "count": int(c)} for q, c in top_q.items()]

    # Open-ended answers are our only 'caption-like' text. How many images have at least one?
    open_imgs = m[m.answer_type == "open"].groupby("split").image_id.nunique()
    out["images_with_open_answer"] = {
        s: {"count": int(open_imgs.get(s, 0)), "pct": _pct(int(open_imgs.get(s, 0)), int(n))}
        for s, n in m.groupby("split").image_id.nunique().items()
    }
    return out


def write_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def stats_markdown(stats: dict) -> str:
    """Human-readable summary for the README / report."""
    L = ["# PathVQA — Phase 0 dataset statistics", "", f"Source: `{stats['source']}`", ""]
    L += [
        "| Split | Q/A rows | Unique images | Q per image | Yes/No % | Median answer (words) |",
        "|---|---|---|---|---|---|",
    ]
    for s, v in stats["splits"].items():
        L.append(
            f"| {s} | {v['qa_rows']:,} | {v['unique_images']:,} | {v['questions_per_image_mean']} "
            f"| {v['yes_no_pct']} | {v['answer_len_words']['median']} |"
        )
    t = stats["total"]
    L += [
        "",
        f"**Total:** {t['qa_rows']:,} Q/A rows over {t['unique_images_all_splits']:,} "
        f"unique images; {t['yes_no_pct']}% yes/no.",
        "",
    ]
    if "train_vs_test" in stats["overlap"]:
        o = stats["overlap"]["train_vs_test"]
        L += [
            "## Train vs test overlap",
            "",
            f"- Question strings shared: {o['questions']['shared_question_strings']:,} "
            f"({o['questions']['pct_of_b_unique_questions_seen_in_a']}% of unique test "
            f"questions; {o['questions']['pct_of_b_rows_with_question_seen_in_a']}% "
            f"of test rows)",
            f"- Exact (question, answer) pairs shared: {o['qa_pairs']['shared_qa_pairs']:,} "
            f"({o['qa_pairs']['pct_of_b_unique_pairs_seen_in_a']}% of unique test pairs)",
            f"- Exact duplicate images (pixel MD5): "
            f"{o['images_exact_md5']['shared_exact_images']:,} "
            f"({o['images_exact_md5']['pct_of_b_images_also_in_a']}% of test images)",
            "",
        ]
    L += ["## Most common questions", ""]
    L += [f"- `{q['question']}` × {q['count']:,}" for q in stats["top_questions"][:10]]
    L += ["", "## Images with at least one open-ended answer", ""]
    L += [
        f"- {s}: {v['count']:,} ({v['pct']}%)" for s, v in stats["images_with_open_answer"].items()
    ]
    return "\n".join(L) + "\n"
