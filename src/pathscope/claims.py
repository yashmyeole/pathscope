"""Phase 4: split neighbour captions into tiered claims (Groq), check each claim against the
query's own Q/A pairs (Gemini, a different model family), and compute TCR. Silent claims are
reported separately, never counted as false."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pathscope import config, llm

PROMPTS = Path(__file__).parent / "prompts"
OUT_DIR = config.ROOT / "eval" / "phase4"
TIERS = ("A", "B", "C")
LABELS = ("supported", "contradicted", "silent")


def _prompt(name):
    text = (PROMPTS / f"{name}.txt").read_text()
    return text, hashlib.sha1(text.encode()).hexdigest()[:8]


def text_id(text):
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def neighbour_texts(nbrs, df):
    """Unique caption texts of the retrieved neighbours, keyed by text_id."""
    text_of = df.set_index("image_id").text
    texts = nbrs.image_id.map(text_of).dropna().unique()
    return {text_id(t): t for t in texts}


def _check_items(items, ids):
    got = [it.get("id") for it in items]
    if got != ids:
        raise ValueError(f"ids came back as {got}, expected {ids}")
    for it in items:
        for c in it["claims"]:
            if c.get("tier") not in TIERS or not str(c.get("claim", "")).strip():
                raise ValueError(f"bad claim {c}")


def extract(texts, batch=12, limit=None, call=None):
    """texts: {text_id: caption}. Returns one row per claim. Cached per caption, so it is safe to
    stop (or run out of quota) and start again."""
    call = call or llm.groq_json
    system, version = _prompt("extract")
    cache = llm.Cache("extract")
    model = llm.GROQ_MODEL if call is llm.groq_json else getattr(call, "__name__", "fake")
    key = {tid: cache.key(model, version, t) for tid, t in texts.items()}

    for tid, t in texts.items():
        if not t.strip() and cache.get(key[tid]) is None:
            cache.put(key[tid], [])  # a few PathVQA images have no usable caption
    todo = [tid for tid in texts if cache.get(key[tid]) is None]
    if limit is not None:
        todo = todo[:limit]
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        payload = [{"id": n, "text": texts[tid]} for n, tid in enumerate(chunk, 1)]
        try:
            items = call(system, json.dumps(payload, ensure_ascii=False))["items"]
            _check_items(items, list(range(1, len(chunk) + 1)))
        except (ValueError, KeyError, TypeError) as e:
            # one retry with a smaller batch usually fixes a malformed answer
            if len(chunk) > 1:
                print(f"  batch {i // batch}: {e}; retrying one caption at a time")
                extract({t: texts[t] for t in chunk}, batch=1, call=call)
                continue
            print(f"  skipping {chunk[0]}: {e}")
            continue
        for tid, it in zip(chunk, items, strict=True):
            cache.put(key[tid], it["claims"])
        if (i // batch) % 10 == 9 or i + batch >= len(todo):
            print(f"  extracted {min(i + batch, len(todo))}/{len(todo)}")

    rows = []
    for tid, t in texts.items():
        claims = cache.get(key[tid])
        if claims is None:
            continue
        for n, c in enumerate(claims, 1):
            rows.append((tid, n, c["claim"].strip(), c["tier"], t))
    return pd.DataFrame(rows, columns=["text_id", "claim_no", "claim", "tier", "text"])


def facts_for(image_id, manifest):
    qa = manifest[manifest.image_id == image_id][["question", "answer"]].drop_duplicates()
    return "\n".join(f"Q: {q.strip()}  A: {a.strip()}" for q, a in qa.itertuples(index=False))


def claims_for_query(qid, nbrs, claims, df):
    text_of = df.set_index("image_id").text
    g = nbrs[nbrs.query_id == qid].sort_values("rank")
    parts = []
    for n in g.itertuples():
        c = claims[claims.text_id == text_id(text_of[n.image_id])]
        parts.append(c.assign(rank=n.rank, neighbour_id=n.image_id, score=n.score, source=n.source))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def verify(query_ids, nbrs, claims, df, manifest, limit=None, call=None):
    call = call or llm.gemini_json
    template, version = _prompt("verify")
    cache = llm.Cache("verify")
    model = llm.GEMINI_MODEL if call is llm.gemini_json else getattr(call, "__name__", "fake")

    done, rows = 0, []
    for qid in query_ids:
        qc = claims_for_query(qid, nbrs, claims, df)
        if qc.empty:
            continue
        facts = facts_for(qid, manifest)
        listing = "\n".join(f"{i}. {c}" for i, c in enumerate(qc.claim, 1))
        key = cache.key(model, version, facts, listing)
        labels = cache.get(key)
        if labels is None:
            if limit is not None and done >= limit:
                continue
            prompt = template.replace("{facts}", facts).replace("{claims}", listing)
            out = call(prompt)["verdicts"]
            labels = [v.get("label") for v in out]
            if len(labels) != len(qc) or any(lab not in LABELS for lab in labels):
                print(f"  {qid}: got {len(labels)} verdicts for {len(qc)} claims, skipped")
                continue
            cache.put(key, labels)
            done += 1
            if done % 25 == 0:
                print(f"  verified {done} queries")
        rows.append(qc.assign(query_id=qid, label=labels))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _rates(v):
    n = len(v)
    s = int((v.label == "supported").sum())
    c = int((v.label == "contradicted").sum())
    return {
        "claims": n,
        "supported": s,
        "contradicted": c,
        "silent": n - s - c,
        "tcr": round(s / (s + c), 3) if s + c else None,  # among claims that could be decided
        "decided": round((s + c) / n, 3) if n else None,
        "tcr_lower": round(s / n, 3) if n else None,  # silence counted as not transferred
    }


def _boot_ci(v, n_boot=1000, seed=config.SEED):
    """95% CI for TCR, resampling whole queries (claims from one query aren't independent)."""
    per_q = v.assign(s=v.label.eq("supported"), c=v.label.eq("contradicted"))
    per_q = per_q.groupby("query_id")[["s", "c"]].sum()
    if per_q.empty or (per_q.s + per_q.c).sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    s, c = per_q.s.to_numpy(), per_q.c.to_numpy()
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        d = s[i].sum() + c[i].sum()
        if d:
            vals.append(s[i].sum() / d)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def tcr(verdicts, df):
    v = verdicts.merge(
        df[["image_id", "image_type", "label_match"]].rename(columns={"image_id": "query_id"}),
        on="query_id",
        how="left",
    )
    out = {"queries": int(v.query_id.nunique()), "overall": _rates(v)}
    out["overall"]["ci95"] = _boot_ci(v)
    out["by_tier"] = {}
    for t in TIERS:
        r = _rates(v[v.tier == t])
        r["ci95"] = _boot_ci(v[v.tier == t])
        out["by_tier"][t] = r
    for col in ("image_type", "source", "rank", "label_match"):
        out[f"by_tier_and_{col}"] = {
            f"{t}|{k}": _rates(g) for (t, k), g in v.groupby(["tier", col]) if len(g) >= 20
        }

    # is a closer neighbour more transferable? one point per (query, neighbour)
    pn = v.assign(s=v.label.eq("supported"), c=v.label.eq("contradicted"))
    pn = pn.groupby(["query_id", "neighbour_id"]).agg(
        score=("score", "first"), s=("s", "sum"), c=("c", "sum")
    )
    pn = pn[(pn.s + pn.c) > 0]
    if len(pn) > 10:
        rate = pn.s / (pn.s + pn.c)
        out["similarity_vs_tcr_spearman"] = {
            "pairs": len(pn),
            "rho": round(float(pn.score.rank().corr(rate.rank())), 3),
        }
    return out


def kappa(a, b):
    """Cohen's kappa for two label lists."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == k).mean()) * float((b == k).mean()) for k in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def compare_extraction(model, ref, threshold=70):
    """How close the extractor gets to the hand-written reference on the same captions. Claims
    are paired greedily by fuzzy text match (rapidfuzz token_set_ratio)."""
    from rapidfuzz import fuzz

    matched, tier_ok, per_caption = 0, 0, []
    for tid, r in ref.groupby("text_id"):
        m = model[model.text_id == tid]
        left = list(m.itertuples())
        hit = 0
        for rc in r.itertuples():
            scores = [fuzz.token_set_ratio(rc.claim.lower(), mc.claim.lower()) for mc in left]
            if scores and max(scores) >= threshold:
                best = left.pop(int(np.argmax(scores)))
                hit += 1
                tier_ok += best.tier == rc.tier
        matched += hit
        per_caption.append((len(r), len(m)))
    n_ref = len(ref)
    n_model = int(model.text_id.isin(ref.text_id).sum())
    counts = np.array(per_caption)
    return {
        "captions": len(per_caption),
        "ref_claims": n_ref,
        "model_claims": n_model,
        "recall": round(matched / n_ref, 3) if n_ref else None,
        "precision": round(matched / n_model, 3) if n_model else None,
        "tier_agreement_on_matched": round(tier_ok / matched, 3) if matched else None,
        "mean_abs_count_diff": round(float(np.abs(counts[:, 0] - counts[:, 1]).mean()), 2),
        "model_tiers": model[model.text_id.isin(ref.text_id)].tier.value_counts().to_dict(),
        "ref_tiers": ref.tier.value_counts().to_dict(),
    }


def judge_sample(verdicts, manifest, n=50, seed=config.SEED):
    """Decisions for a person to label blind. Over-samples supported/contradicted, since a
    random draw would be almost all 'silent' and say nothing about the hard calls."""
    per = {"supported": n * 2 // 5, "contradicted": n * 2 // 5}
    per["silent"] = n - sum(per.values())
    parts = []
    for lab, k in per.items():
        g = verdicts[verdicts.label == lab]
        parts.append(g.sample(min(k, len(g)), random_state=seed))
    s = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    s["facts"] = s.query_id.map(lambda q: facts_for(q, manifest))
    s.insert(0, "item", range(1, len(s) + 1))
    return s[["item", "query_id", "neighbour_id", "tier", "claim", "facts", "label"]].rename(
        columns={"label": "judge"}
    )
