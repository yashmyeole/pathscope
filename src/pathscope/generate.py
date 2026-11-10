"""Phase 5: describe each query image from its retrieved captions, plus two controls.

full     the image and its 5 retrieved captions
blind    the same 5 captions, no image
random   the image and 5 random corpus captions, each from the same source as the neighbour it
         replaces, so caption style stays the same and only relevance changes
"""

import io
import json
import re

import numpy as np
import pandas as pd

from pathscope import claims, config, llm

OUT_DIR = config.ROOT / "eval" / "phase5"
CONDITIONS = ("full", "blind", "random")
SETUP = {
    "full": "The image is attached.",
    "random": "The image is attached.",
    "blind": "You cannot see the image. You only have the captions below.",
}


def sample_queries(nbrs, n=250, seed=config.SEED):
    qids = sorted(nbrs.query_id.unique())
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(qids, size=min(n, len(qids)), replace=False).tolist())


def caption_sets(qids, nbrs, df, seed=config.SEED):
    """{(query_id, condition): [5 captions]} for every query and condition."""
    text_of = df.set_index("image_id").text
    corpus = df[(df.role == "corpus") & df.text.str.strip().astype(bool)]
    pools = {s: g.image_id.to_numpy() for s, g in corpus.groupby("source")}
    rng = np.random.default_rng(seed)
    out = {}
    for qid in qids:
        g = nbrs[nbrs.query_id == qid].sort_values("rank")
        mine = set(g.image_id)
        retrieved = [text_of[i] for i in g.image_id]
        rand = []
        for src in g.source:
            pick = rng.choice(pools[src])
            while pick in mine:
                pick = rng.choice(pools[src])
            rand.append(text_of[pick])
        out[(qid, "full")] = out[(qid, "blind")] = retrieved
        out[(qid, "random")] = rand
    return out


def jpeg(img, size=512):
    img = img.convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def generate(qids, sets, load_image, conditions=CONDITIONS, limit=None, call=None):
    """Descriptions for every (query, condition), cached. load_image(qid) -> PIL image."""
    call = call or llm.gemini_json
    template = (claims.PROMPTS / "describe.txt").read_text()
    version = claims.text_id(template)
    cache = llm.Cache.open("generate")
    model = llm.GEMINI_MODEL if call is llm.gemini_json else getattr(call, "__name__", "fake")

    asked, rows = 0, []
    for qid in qids:
        for cond in conditions:
            caps = sets[(qid, cond)]
            listing = "\n".join(f"{i}. {c}" for i, c in enumerate(caps, 1))
            key = cache.key(model, version, cond, qid, listing)
            desc = cache.get(key)
            if desc is None:
                if limit is not None and asked >= limit:
                    continue
                prompt = template.replace("{setup}", SETUP[cond]).replace("{captions}", listing)
                image = None if cond == "blind" else jpeg(load_image(qid))
                desc = str(call(prompt, image=image).get("description", "")).strip()
                asked += 1
                if not desc:
                    continue
                cache.put(key, desc)
                if asked % 50 == 0:
                    print(f"  generated {asked}")
            rows.append((qid, cond, desc, json.dumps(caps)))
    return pd.DataFrame(rows, columns=["query_id", "condition", "description", "captions"])


STOP = set(
    "a an the of in and or with is are was were be been this that these those it its to on at "
    "by for from as there image images shows show shown seen specimen tissue organ which into "
    "has have also some".split()
)


def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def copied(claim, captions, share=0.8):
    """True if most of the claim's content words appear together in one input caption."""
    w = _words(claim)
    return bool(w) and any(len(w & _words(c)) >= share * len(w) for c in captions)


def score(gen, manifest, extract_limit=None, verify_limit=None):
    """Claims from each description, checked against the query's facts. One row per claim."""
    texts = {claims.text_id(d): d for d in gen.description}
    try:
        claims.extract(texts, limit=extract_limit)
    except llm.QuotaExhausted as e:
        print(f"  extraction stopped: {str(e)[:120]}")
    cl = claims.extract(texts, limit=0)
    done = set(cl.text_id) | claims.extracted_ids(texts)

    asked, rows = 0, []
    for r in gen.itertuples():
        tid = claims.text_id(r.description)
        if tid not in done:
            continue
        c = cl[cl.text_id == tid]
        if c.empty:
            continue
        ask = verify_limit is None or asked < verify_limit
        try:
            labels, a = claims.check(claims.facts_for(r.query_id, manifest), list(c.claim), ask=ask)
        except llm.QuotaExhausted as e:
            print(f"  verification stopped: {str(e)[:120]}")
            verify_limit, labels, a = asked, None, False
        asked += a
        if labels is None:
            continue
        caps = json.loads(r.captions)
        rows.append(
            c.assign(
                query_id=r.query_id,
                condition=r.condition,
                label=labels,
                copied=[copied(x, caps) for x in c.claim],
            )
        )
    cols = ["query_id", "condition", "claim", "tier", "label", "copied"]
    return pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)


def _tcr(v):
    s, c = (v.label == "supported").sum(), (v.label == "contradicted").sum()
    return s / (s + c) if s + c else np.nan


def paired_gain(v, a, b, tier=None, n_boot=1000, seed=config.SEED):
    """TCR(a) - TCR(b) on queries scored in both conditions, with a query-level bootstrap CI."""
    if tier:
        v = v[v.tier == tier]
    both = set(v[v.condition == a].query_id) & set(v[v.condition == b].query_id)
    v = v[v.query_id.isin(both)]
    per = v.assign(s=v.label.eq("supported"), c=v.label.eq("contradicted"))
    per = per.groupby(["query_id", "condition"])[["s", "c"]].sum().unstack("condition").fillna(0)
    if per.empty:
        return None
    qs = per.index.to_numpy()

    def diff(idx):
        x = per.loc[qs[idx]]
        ta = x["s"][a].sum() / max(x["s"][a].sum() + x["c"][a].sum(), 1)
        tb = x["s"][b].sum() / max(x["s"][b].sum() + x["c"][b].sum(), 1)
        return ta - tb

    rng = np.random.default_rng(seed)
    boots = [diff(rng.integers(0, len(qs), len(qs))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "queries": len(qs),
        "gain": round(float(diff(np.arange(len(qs)))), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
    }


def summary(v, df):
    out = {"queries": int(v.query_id.nunique())}
    for cond in CONDITIONS:
        x = v[v.condition == cond]
        if x.empty:
            continue
        per_desc = x.groupby("query_id")
        r = claims._rates(x)
        r["ci95"] = claims._boot_ci(x)
        r["claims_per_description"] = round(float(per_desc.size().mean()), 2)
        for lab in ("supported", "contradicted"):
            r[f"{lab}_per_description"] = round(
                float(per_desc.label.apply(lambda s, lab=lab: (s == lab).sum()).mean()), 2
            )
        r["copy_rate"] = round(float(x.copied.mean()), 3)
        r["by_tier"] = {t: claims._rates(x[x.tier == t]) for t in claims.TIERS}
        out[cond] = r

    out["multimodal_gain"] = {
        t or "all": paired_gain(v, "full", "blind", t) for t in (None, "A", "B")
    }
    out["retrieval_gain"] = {
        t or "all": paired_gain(v, "full", "random", t) for t in (None, "A", "B")
    }
    full = v[v.condition == "full"]
    out["full_copied_vs_not"] = {
        "copied": claims._rates(full[full.copied]),
        "not_copied": claims._rates(full[~full.copied]),
    }
    kinds = df.set_index("image_id").image_type
    out["full_minus_blind_by_image_type"] = {
        k: paired_gain(v[v.query_id.map(kinds) == k], "full", "blind")
        for k in sorted(kinds[v.query_id.unique()].unique())
    }

    # queries with an exact label twin in the corpus can be described from the captions alone
    twin = df.set_index("image_id").label_match
    out["by_label_match"] = {}
    for k in (True, False):
        x = v[v.query_id.map(twin) == k]
        out["by_label_match"][str(k).lower()] = {
            "queries": int(x.query_id.nunique()),
            "tcr": {c: claims._rates(x[x.condition == c])["tcr"] for c in CONDITIONS},
            "multimodal_gain": paired_gain(x, "full", "blind"),
            "retrieval_gain": paired_gain(x, "full", "random"),
        }
    return out
