"""Phase 3 retrieval checks. Everything here works off embeddings.parquet with plain numpy
(exact cosine), so it doesn't need the Qdrant cluster to be up."""

import json

import numpy as np
import pandas as pd

from pathscope import config
from pathscope.data.split import SPLIT_DIR
from pathscope.organs import organs

OUT_DIR = config.ROOT / "eval" / "phase3"


def load_embeddings(path=None):
    df = pd.read_parquet(path or config.EMBED_DIR / "embeddings.parquet")
    X = np.stack(df.embedding.to_numpy()).astype("float32")
    return df.drop(columns="embedding"), X


def topk(Q, C, k):
    """Indices and scores of the k most similar rows of C for each row of Q, best first."""
    S = Q @ C.T
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-s, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(s, order, axis=1)


def neighbour_table(df, X, role="test", k=5):
    q = df[df.role == role]
    c = df[df.role == "corpus"]
    idx, sc = topk(X[q.index], X[c.index], k)
    c = c.reset_index(drop=True)
    rows = []
    for qid, nbr, s in zip(q.image_id, idx, sc, strict=True):
        for rank, (j, score) in enumerate(zip(nbr, s, strict=True), 1):
            rows.append(
                (qid, rank, c.image_id[j], round(float(score), 4), c.source[j], c.image_type[j])
            )
    cols = ["query_id", "rank", "image_id", "score", "source", "image_type"]
    return pd.DataFrame(rows, columns=cols)


def _label_twin_ranks(df, X, queries):
    """Rank (1-based, among the whole corpus) of the corpus image that has the exact same short
    label as the query. Only defined for label_match queries."""
    corpus_ids = pd.read_parquet(SPLIT_DIR / "corpus.parquet", columns=["image_id", "corpus_idx"])
    twin_of = queries[queries.label_match].set_index("image_id").label_corpus_idx
    twin_of = twin_of.astype(int).map(corpus_ids.set_index("corpus_idx").image_id)

    c = df[df.role == "corpus"]
    pos = pd.Series(np.arange(len(c)), index=c.image_id)
    qpos = pd.Series(df.index, index=df.image_id)
    C = X[c.index]
    ranks = {}
    for qid, tid in twin_of.items():
        s = C @ X[qpos[qid]]
        ranks[qid] = int((s > s[pos[tid]]).sum()) + 1
    return pd.Series(ranks)


def _rate(hits):
    hits = list(hits)
    return {"n": len(hits), "rate": round(float(np.mean(hits)), 3) if hits else None}


def retrieval_stats(df, X, nbrs):
    corpus = df[df.role == "corpus"]
    queries = pd.read_parquet(SPLIT_DIR / "queries.parquet")
    q = df[df.role == "test"]
    top1 = nbrs[nbrs["rank"] == 1].set_index("query_id")
    out = {"n_queries": len(q), "n_corpus": len(corpus)}

    out["top1_score"] = top1.score.describe()[["mean", "50%", "min", "max"]].round(3).to_dict()
    share = corpus.source.value_counts(normalize=True)
    got = nbrs.source.value_counts(normalize=True)
    out["neighbour_source_share"] = {
        s: {"top5": round(float(got.get(s, 0)), 3), "corpus": round(float(share[s]), 3)}
        for s in share.index
    }

    # image type agreement, with the chance rate you'd get from a random corpus image
    q_type = q.set_index("image_id").image_type
    c_type = corpus.image_type.value_counts(normalize=True)
    same = top1.image_type == q_type.loc[top1.index]
    # optimistic: the type labels come from a classifier trained on these same embeddings
    out["type_top1"] = _rate(same)
    out["type_chance"] = round(float(q_type.map(c_type).fillna(0).mean()), 3)
    out["type_top1_by_query_type"] = {
        t: _rate(same[q_type.loc[same.index] == t]) for t in sorted(q_type.unique())
    }

    # organ agreement from captions
    c_org = corpus.set_index("image_id").text.map(organs)
    tagged = c_org[c_org.map(len) > 0]
    nb = nbrs.assign(org=nbrs.image_id.map(c_org))
    per_q = []
    for qid, qtext, qtype, lm in zip(q.image_id, q.text, q.image_type, q.label_match, strict=True):
        qo = organs(qtext)
        if not qo:
            continue
        g = nb[nb.query_id == qid]
        first = g.org.iloc[0]
        others = [o for o in g.org if o]
        per_q.append(
            {
                "query_id": qid,
                "query_type": qtype,
                "label_match": bool(lm),
                "top1_source": g.source.iloc[0],
                "top1": bool(qo & first) if first else None,
                "top5": any(qo & o for o in others) if others else None,
                "chance": float(np.mean([bool(qo & o) for o in tagged])),
            }
        )
    org = pd.DataFrame(per_q)
    out["organ_queries_tagged"] = f"{len(org)} of {len(q)}"
    out["organ_top1"] = _rate(org.top1.dropna())
    out["organ_any_of_top5"] = _rate(org.top5.dropna())
    out["organ_chance_one_random"] = round(float(org.chance.mean()), 3)
    for col in ("query_type", "top1_source", "label_match"):
        out[f"organ_top1_by_{col}"] = {str(k): _rate(g.top1.dropna()) for k, g in org.groupby(col)}

    ranks = _label_twin_ranks(df, X, queries)
    out["label_twin"] = {
        "n": len(ranks),
        "median_rank": int(ranks.median()),
        "in_top5": round(float((ranks <= 5).mean()), 3),
        "in_top50": round(float((ranks <= 50).mean()), 3),
        "chance_median_rank": len(corpus) // 2,
    }
    return out


def arch_crossmodal(df, X, text_vecs, captions, ns=(1, 10, 50, 200)):
    """Image->text and text->image recall on the ARCH part of the corpus, same idea as Table 2 of
    the Quilt-1M paper. Book figures share one caption across panels, so a text query counts as
    a hit if any of its panels comes back."""
    arch = df[df.source.str.startswith("arch_")]
    cap_id = arch.text.map({c: i for i, c in enumerate(captions)}).to_numpy()
    imgs = X[arch.index]
    S = imgs @ text_vecs.T  # images x captions

    # rank of the right caption for each image
    right = S[np.arange(len(imgs)), cap_id]
    i2t = (S > right[:, None]).sum(1) + 1

    # rank of the best matching panel for each caption
    St = S.T
    t2i = np.empty(len(captions), dtype=int)
    for t in range(len(captions)):
        mine = St[t, cap_id == t]
        t2i[t] = (St[t] > mine.max()).sum() + 1

    res = {"images": len(imgs), "captions": len(captions)}
    for name, r in (("image_to_text", i2t), ("text_to_image", t2i)):
        res[name] = {f"R@{n}": round(100 * float((r <= n).mean()), 2) for n in ns}
    return res


def save_json(obj, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(obj, indent=2))


def sample_queries(df, n=50, seed=config.SEED):
    return df[df.role == "test"].sample(n, random_state=seed).image_id.tolist()


def render_sheets(query_ids, nbrs, df, per_sheet=5, out_dir=None):
    """One row per query: the query on the left, its 5 neighbours to the right."""
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pathscope.data.pairs import ImageLoader

    out_dir = out_dir or OUT_DIR / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.concat(
        [pd.read_parquet(SPLIT_DIR / f"{f}.parquet") for f in ("queries", "corpus")]
    ).set_index("image_id")
    meta = df.set_index("image_id")
    loader = ImageLoader()

    def tile(ax, iid, head):
        ax.imshow(loader.load(manifest.loc[iid].to_dict() | {"image_id": iid}))
        ax.set_xticks([])
        ax.set_yticks([])
        cap = textwrap.fill(textwrap.shorten(str(meta.text[iid]), 110), 34)
        ax.set_title(f"{head}\n{cap}", fontsize=6.5, loc="left", family="monospace")

    paths = []
    for s in range(0, len(query_ids), per_sheet):
        chunk = query_ids[s : s + per_sheet]
        fig, axes = plt.subplots(len(chunk), 6, figsize=(17, 3.9 * len(chunk)))
        axes = np.atleast_2d(axes)
        for r, qid in enumerate(chunk):
            tile(axes[r, 0], qid, f"Q{s + r + 1:02d} {meta.image_type[qid]}")
            g = nbrs[nbrs.query_id == qid].sort_values("rank")
            for c, n in enumerate(g.itertuples(), 1):
                tile(axes[r, c], n.image_id, f"#{n.rank} {n.score:.3f} {n.source}")
        fig.tight_layout()
        path = out_dir / f"queries_{s // per_sheet + 1:02d}.png"
        fig.savefig(path, dpi=75)
        plt.close(fig)
        paths.append(path)
    return paths


def review_summary(path=None):
    rv = pd.read_csv(path or OUT_DIR / "review.csv")

    def yes(col):
        v = rv[col][rv[col].isin(["y", "n"])]
        return {"n": len(v), "rate": round(float((v == "y").mean()), 3)}

    group = rv.kind.replace({"gross": "photo", "clinical": "photo", "xray": "diagram"})
    return {
        "cases": len(rv),
        "verdict": rv.verdict.value_counts().to_dict(),
        "verdict_by_kind": {k: g.value_counts().to_dict() for k, g in rv.verdict.groupby(group)},
        "top1_same_kind": yes("top1_same_kind"),
        "top1_same_organ": yes("top1_same_organ"),
        "top1_same_stain": yes("top1_same_stain"),
        "top1_similar_mag": yes("top1_similar_mag"),
        "mean_related_of_5": round(float(rv.n_related_of_5.mean()), 2),
        "same_case_in_top5": yes("same_case_in_top5"),
    }


def arch_text_vectors():
    """PLIP text vectors for the unique ARCH captions, cached next to the image embeddings."""
    path = config.EMBED_DIR / "arch_text.parquet"
    if path.exists():
        t = pd.read_parquet(path)
        return t.text.tolist(), np.stack(t.embedding.to_numpy())

    from pathscope.embed import embed_texts, load_plip

    df = pd.read_parquet(config.EMBED_DIR / "embeddings.parquet", columns=["source", "text"])
    caps = sorted(df[df.source.str.startswith("arch_")].text.unique())
    model, proc = load_plip()
    T = embed_texts(model, proc, caps)
    pd.DataFrame({"text": caps, "embedding": list(T)}).to_parquet(path, index=False)
    return caps, T
