"""Phase 7: recompute the headline numbers from the committed verdict tables, draw the figures,
and fill in docs/report.template.md, so the numbers in the report can't drift from the data.

Everything here runs from a clean clone: the only inputs are files under eval/ and
data/manifests/. Image type and label_match per image come from eval/image_meta.csv, which is
written from embeddings.parquet whenever that file is present."""

import json
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from pathscope import claims, config, generate

DOCS = config.ROOT / "docs"
FIG_DIR = DOCS / "figures"
TEMPLATE = DOCS / "report.template.md"
REPORT = DOCS / "report.md"
META = config.ROOT / "eval" / "image_meta.csv"
EVAL = config.ROOT / "eval"

# Quilt-1M paper, Table 2, PLIP on ARCH (percent; the paper has no R@10)
QUILT_PLIP = {
    "image_to_text": {"R@1": 0.74, "R@50": 13.75, "R@200": 29.46},
    "text_to_image": {"R@1": 0.56, "R@50": 13.10, "R@200": 29.85},
}


def write_meta():
    path = config.EMBED_DIR / "embeddings.parquet"
    if not path.exists():
        return False
    cols = ["image_id", "role", "source", "image_type", "label_match"]
    pd.read_parquet(path, columns=cols).to_csv(META, index=False)
    return True


def load_meta():
    return pd.read_csv(META)


def recompute(meta):
    """Phase 4 and 5 summaries rebuilt from the verdict tables."""
    v4 = pd.read_csv(EVAL / "phase4" / "verdicts.csv")
    v5 = pd.read_csv(EVAL / "phase5" / "verdicts.csv")
    return claims.tcr(v4, meta), generate.summary(v5, meta)


def differences(new, old, path=""):
    """Paths where a recomputed value differs from the committed one (keys only in one side
    are ignored, floats compared to 3 decimals)."""
    out = []
    if isinstance(new, dict) and isinstance(old, dict):
        for k in new.keys() & old.keys():
            out += differences(new[k], old[k], f"{path}.{k}" if path else str(k))
    elif isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
        for i, (a, b) in enumerate(zip(new, old, strict=True)):
            out += differences(a, b, f"{path}[{i}]")
    elif isinstance(new, float) or isinstance(old, float):
        if new is None or old is None or round(float(new), 3) != round(float(old), 3):
            out.append(path)
    elif new != old:
        out.append(path)
    return out


def _json(path):
    return json.loads(path.read_text())


def context(p4, p5):
    committed4 = _json(EVAL / "phase4" / "tcr.json")
    # judge agreement and the same-case split need the hand labels; take them as committed
    for k in ("judge_kappa_vs_label_ref", "judge_kappa_vs_label_yash", "reviewed_50_by_same_case"):
        if k in committed4:
            p4.setdefault(k, committed4[k])
    for r in p4.get("reviewed_50_by_same_case", {}).values():
        r["decided_n"] = r["supported"] + r["contradicted"]
    split = _json(config.MANIFEST_DIR / "split_v1" / "summary.json")
    return {
        "p0": _json(config.STATS_DIR / "pathvqa_stats.json"),
        "p1": split,
        "p3": _json(EVAL / "phase3" / "metrics.json"),
        "p4": p4,
        "x": _json(EVAL / "phase4" / "extract_check.json"),
        "p5": p5,
    }


def _get(ctx, path):
    cur = ctx
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def _round(x, places):
    # half up, so 0.585 prints as 0.59 everywhere instead of Python's banker's rounding
    q = Decimal(1).scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def fmt(x, how):
    if x is None:
        return "–"
    if how == "2f":
        return f"{_round(x, 2):.2f}".replace("-", "−")
    if how == "1f":
        return f"{_round(x, 1):.1f}"
    if how == "pct":
        return f"{_round(100 * x, 0):.0f}%"
    if how == "pct1":
        return f"{_round(100 * x, 1):.1f}%"
    if how == "int":
        return f"{int(x):,}"
    if how == "ci":
        sep = " to " if x[0] < 0 else "–"
        return f"{fmt(x[0], '2f')}{sep}{fmt(x[1], '2f')}"
    if how == "signed":
        return ("+" if x >= 0 else "−") + f"{_round(abs(x), 2):.2f}"
    return str(x)


def _rate_row(name, r, ci=True):
    cells = [
        name,
        fmt(r["claims"], "int"),
        fmt(r["supported"], "int"),
        fmt(r["contradicted"], "int"),
        fmt(r["decided"], "pct"),
        f"**{fmt(r['tcr'], '2f')}**",
    ]
    if ci:
        cells.append(fmt(r.get("ci95"), "ci"))
    return cells


def _md(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def tables(ctx):
    p3, p4, p5 = ctx["p3"], ctx["p4"], ctx["p5"]
    head = ["", "claims", "supported", "contradicted", "checkable", "TCR", "95% CI"]
    tiers = {"A": "A, visual", "B": "B, diagnostic", "C": "C, case-specific"}
    t = {}

    t["tiers"] = _md(
        head,
        [_rate_row(name, p4["by_tier"][k]) for k, name in tiers.items()]
        + [_rate_row("All", p4["overall"] | {"ci95": p4["overall"].get("ci95")})],
    )

    rank = p4["by_tier_and_rank"]
    t["rank"] = _md(
        ["neighbour rank", "1", "2", "3", "4", "5"],
        [
            [f"tier {k}"] + [fmt(rank.get(f"{k}|{r}", {}).get("tcr"), "2f") for r in range(1, 6)]
            for k in ("A", "B")
        ],
    )

    def with_n(r):
        return f"{fmt(r['tcr'], '2f')} ({r['supported'] + r['contradicted']})"

    kinds = p4["by_tier_and_image_type"]
    t["kinds"] = _md(
        ["", "micrograph", "photo", "diagram"],
        [
            [f"tier {k}"]
            + [with_n(kinds[f"{k}|{kind}"]) for kind in ("micrograph", "photo", "diagram")]
            for k in ("A", "B")
        ],
    )

    src = p4["by_tier_and_source"]
    names = {
        "pathvqa_train": "PathVQA train",
        "arch_books": "ARCH books",
        "arch_pubmed": "ARCH PubMed",
    }
    t["sources"] = _md(
        [
            "neighbour source",
            "share of top 5",
            "tier A TCR",
            "A checkable",
            "tier B TCR",
            "B checkable",
        ],
        [
            [
                name,
                fmt(p3["automatic"]["neighbour_source_share"][s]["top5"], "pct"),
                fmt(src[f"A|{s}"]["tcr"], "2f"),
                fmt(src[f"A|{s}"]["decided"], "pct"),
                fmt(src[f"B|{s}"]["tcr"], "2f"),
                fmt(src[f"B|{s}"]["decided"], "pct"),
            ]
            for s, name in names.items()
        ],
    )

    cm = p3["arch_crossmodal"]
    t["crossmodal"] = _md(
        ["", "R@1", "R@10", "R@50", "R@200"],
        [
            [label, *[fmt(cm[d].get(k), "2f") for k in ("R@1", "R@10", "R@50", "R@200")]]
            for d, label in (
                ("image_to_text", "image → text, ours"),
                ("text_to_image", "text → image, ours"),
            )
        ]
        + [
            [
                label,
                *[
                    fmt(QUILT_PLIP[d].get(k), "2f") if k in QUILT_PLIP[d] else "n/a"
                    for k in ("R@1", "R@10", "R@50", "R@200")
                ],
            ]
            for d, label in (
                ("image_to_text", "image → text, Quilt-1M paper"),
                ("text_to_image", "text → image, Quilt-1M paper"),
            )
        ],
    )

    conds = {
        "full": "image + retrieved captions",
        "random": "image + random captions",
        "blind": "retrieved captions, no image",
    }
    t["conditions"] = _md(
        [
            "condition",
            "TCR",
            "95% CI",
            "checkable",
            "supported / description",
            "contradicted / description",
            "copied",
        ],
        [
            [
                name,
                f"**{fmt(p5[c]['tcr'], '2f')}**",
                fmt(p5[c]["ci95"], "ci"),
                fmt(p5[c]["decided"], "pct"),
                fmt(p5[c]["supported_per_description"], "2f"),
                fmt(p5[c]["contradicted_per_description"], "2f"),
                fmt(p5[c]["copy_rate"], "pct1"),
            ]
            for c, name in conds.items()
        ],
    )

    def gain(g):
        return f"{fmt(g['gain'], 'signed')} [{fmt(g['ci95'], 'ci')}]"

    t["gains"] = _md(
        ["", "all claims", "tier A", "tier B"],
        [
            [
                "image gain (full − captions only)",
                *[gain(p5["multimodal_gain"][k]) for k in ("all", "A", "B")],
            ],
            [
                "retrieval gain (full − random)",
                *[gain(p5["retrieval_gain"][k]) for k in ("all", "A", "B")],
            ],
        ],
    )

    lm = p5["by_label_match"]
    t["twins"] = _md(
        ["", "images", "full", "captions only", "random", "image gain", "retrieval gain"],
        [
            [
                name,
                lm[k]["queries"],
                fmt(lm[k]["tcr"]["full"], "2f"),
                fmt(lm[k]["tcr"]["blind"], "2f"),
                fmt(lm[k]["tcr"]["random"], "2f"),
                gain(lm[k]["multimodal_gain"]),
                gain(lm[k]["retrieval_gain"]),
            ]
            for k, name in (("true", "label twin in corpus"), ("false", "no label twin"))
        ],
    )
    return t


def render(template, ctx, tabs):
    def sub(m):
        key = m.group(1).strip()
        if key.startswith("table:"):
            return tabs[key[6:]]
        # stats keys like "B|y" contain a bar too, so the format is whatever follows the last one
        path, _, how = key.rpartition("|")
        return fmt(_get(ctx, path.strip()), how.strip())

    return re.sub(r"\{\{(.+?)\}\}", sub, template)


INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, labels, vals, cis, color, x=None, width=0.6, label=None):
    x = np.arange(len(labels)) if x is None else x
    lo = [v - c[0] for v, c in zip(vals, cis, strict=True)]
    hi = [c[1] - v for v, c in zip(vals, cis, strict=True)]
    ax.bar(x, vals, width, color=color, edgecolor=SURFACE, linewidth=2, label=label, zorder=2)
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
    for xi, v, c in zip(x, vals, cis, strict=True):
        ax.text(xi, c[1] + 0.02, fmt(v, "2f"), ha="center", va="bottom", fontsize=9, color=INK)
    return x


def figures(ctx):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p4, p5 = ctx["p4"], ctx["p5"]
    paths = []

    def save(fig, name):
        path = FIG_DIR / name
        fig.savefig(
            path, dpi=160, bbox_inches="tight", facecolor=SURFACE, metadata={"Software": None}
        )
        plt.close(fig)
        paths.append(path)

    # 1. TCR by tier, raw neighbour captions
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _axes(ax)
    tiers = [("A", "A visual"), ("B", "B diagnostic"), ("C", "C case-specific"), ("all", "All")]
    rows = [p4["overall"] if k == "all" else p4["by_tier"][k] for k, _ in tiers]
    _bars(ax, [n for _, n in tiers], [r["tcr"] for r in rows], [r["ci95"] for r in rows], BLUE)
    ax.set_xticks(
        range(len(tiers)),
        [f"{n}\n{r['decided']:.0%} checkable" for (_, n), r in zip(tiers, rows, strict=True)],
    )
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_ylabel("TCR (95% CI)", color=INK2, fontsize=9)
    ax.set_title(
        "Neighbour-caption claims that hold for the query image", loc="left", fontsize=11, color=INK
    )
    save(fig, "tcr_by_tier.png")

    # 2. TCR by neighbour rank
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _axes(ax)
    rank = p4["by_tier_and_rank"]
    for k, color, name in (("A", BLUE, "A visual"), ("B", ORANGE, "B diagnostic")):
        ys = [rank[f"{k}|{r}"]["tcr"] for r in range(1, 6)]
        ax.plot(
            range(1, 6),
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=name,
            zorder=3,
        )
        ax.text(5.12, ys[-1], name, color=INK, fontsize=9, va="center")
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xlim(0.8, 5.9)
    ax.set_ylim(0.3, 0.95)
    ax.set_xticks(range(1, 6), [f"#{r}" for r in range(1, 6)])
    ax.set_xlabel("neighbour rank", color=INK2, fontsize=9)
    ax.set_ylabel("TCR", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.86, 1))
    ax.set_title("Transfer falls with rank", loc="left", fontsize=11, color=INK)
    save(fig, "tcr_by_rank.png")

    # 3. three conditions
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _axes(ax)
    conds = [
        ("full", "image +\nretrieved captions"),
        ("random", "image +\nrandom captions"),
        ("blind", "retrieved captions,\nno image"),
    ]
    _bars(
        ax,
        [n for _, n in conds],
        [p5[c]["tcr"] for c, _ in conds],
        [p5[c]["ci95"] for c, _ in conds],
        BLUE,
    )
    ax.set_xticks(range(3), [n for _, n in conds])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_ylabel("TCR of generated claims (95% CI)", color=INK2, fontsize=9)
    ax.set_title(
        "Generated descriptions under the two controls", loc="left", fontsize=11, color=INK
    )
    save(fig, "conditions.png")

    # 4. label twins
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _axes(ax)
    lm = p5["by_label_match"]
    groups = [
        ("true", f"label twin in corpus\n({lm['true']['queries']} images)"),
        ("false", f"no label twin\n({lm['false']['queries']} images)"),
    ]
    x = np.arange(2)
    w = 0.34
    for off, cond, color, name in (
        (-w / 2, "full", BLUE, "image + captions"),
        (w / 2, "blind", ORANGE, "captions only"),
    ):
        vals = [lm[k]["tcr"][cond] for k, _ in groups]
        ax.bar(x + off, vals, w, color=color, edgecolor=SURFACE, linewidth=2, label=name, zorder=2)
        for xi, v in zip(x + off, vals, strict=True):
            ax.text(xi, v + 0.02, fmt(v, "2f"), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [n for _, n in groups])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_ylabel("TCR", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.set_title(
        "The image matters when there is no label twin", loc="left", fontsize=11, color=INK
    )
    save(fig, "label_twins.png")
    return paths


def build():
    """Recompute, check against the committed results, draw figures, render the report."""
    refreshed = write_meta()
    meta = load_meta()
    p4, p5 = recompute(meta)
    diff4 = differences(p4, _json(EVAL / "phase4" / "tcr.json"))
    diff5 = differences(p5, _json(EVAL / "phase5" / "results.json"))
    ctx = context(p4, p5)
    figs = figures(ctx)
    body = re.sub(r"\A<!--.*?-->\s*", "", TEMPLATE.read_text(), flags=re.S)
    note = (
        "<!-- Generated by `pathscope report` from docs/report.template.md. Edit that file. -->\n\n"
    )
    REPORT.write_text(note + render(body, ctx, tables(ctx)))
    return {
        "image_meta": "rewritten from embeddings.parquet"
        if refreshed
        else "read from eval/image_meta.csv",
        "phase4_mismatches": diff4,
        "phase5_mismatches": diff5,
        "figures": [str(p.relative_to(config.ROOT)) for p in figs],
        "report": str(REPORT.relative_to(config.ROOT)),
    }
