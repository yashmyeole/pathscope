"""Phase 6: export what the Next.js demo needs. One JSON file per Phase 5 query (neighbours,
the three descriptions with their checked claims, the query's own PathVQA facts), a gallery
index, a summary of the headline numbers, and small WebP thumbnails of every image shown."""

import json
import shutil

import numpy as np
import pandas as pd

from pathscope import claims, config, generate

APP_DIR = config.ROOT / "app"
DATA_DIR = APP_DIR / "data"
IMG_DIR = APP_DIR / "public" / "img"
QUERY_PX, NEIGHBOUR_PX = 512, 256


def source_of(claim, captions, share=0.5):
    """1-based index of the caption sharing the most content words with the claim, if at least
    `share` of them; else None. Used to link a generated claim back to a neighbour."""
    w = generate._words(claim)
    if not w:
        return None
    overlap = [len(w & generate._words(c)) / len(w) for c in captions]
    best = int(np.argmax(overlap))
    return best + 1 if overlap[best] >= share else None


def _facts(qid, manifest):
    qa = manifest[manifest.image_id == qid][["question", "answer"]].drop_duplicates()
    qa = qa[qa.answer.str.strip().str.lower() != "no"]
    return [{"q": q.strip(), "a": a.strip()} for q, a in qa.itertuples(index=False)]


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def query_record(qid, nbrs, df, gen, verdicts, manifest):
    meta = df.set_index("image_id")
    g = nbrs[nbrs.query_id == qid].sort_values("rank")
    neighbours = [
        {
            "rank": int(n.rank),
            "id": n.image_id,
            "score": round(float(n.score), 3),
            "source": n.source,
            "kind": meta.image_type[n.image_id],
            "caption": str(meta.text[n.image_id]),
        }
        for n in g.itertuples()
    ]
    conditions = {}
    for cond in generate.CONDITIONS:
        row = gen[(gen.query_id == qid) & (gen.condition == cond)]
        if row.empty:
            continue
        caps = json.loads(row.captions.iloc[0])
        v = verdicts[(verdicts.query_id == qid) & (verdicts.condition == cond)]
        conditions[cond] = {
            "description": row.description.iloc[0],
            "captions": caps if cond == "random" else None,
            "claims": [
                {
                    "text": c.claim,
                    "tier": c.tier,
                    "label": c.label,
                    "from": source_of(c.claim, caps),
                    "copied": bool(c.copied),
                }
                for c in v.itertuples()
            ],
        }
    return {
        "id": qid,
        "kind": meta.image_type[qid],
        "labelMatch": bool(meta.label_match[qid]),
        "facts": _facts(qid, manifest),
        "neighbours": neighbours,
        "conditions": conditions,
    }


def summary(tcr4, res5):
    keep = ("claims", "supported", "contradicted", "silent", "tcr", "decided")
    by_rank = tcr4["by_tier_and_rank"]

    def pick(r):
        return {k: _num(r.get(k)) for k in keep} | {"ci95": r.get("ci95")}

    return {
        "phase4": {
            "queries": tcr4["queries"],
            "overall": pick(tcr4["overall"]),
            "byTier": {t: pick(tcr4["by_tier"][t]) for t in claims.TIERS},
            "byRank": {
                t: [_num(by_rank.get(f"{t}|{r}", {}).get("tcr")) for r in range(1, 6)]
                for t in ("A", "B")
            },
            "spearman": tcr4.get("similarity_vs_tcr_spearman"),
        },
        "phase5": {
            "queries": res5["queries"],
            "conditions": {
                c: pick(res5[c])
                | {
                    "copyRate": res5[c]["copy_rate"],
                    "supportedPerDescription": res5[c]["supported_per_description"],
                    "contradictedPerDescription": res5[c].get("contradicted_per_description"),
                }
                for c in generate.CONDITIONS
            },
            "multimodalGain": res5["multimodal_gain"]["all"],
            "retrievalGain": res5["retrieval_gain"]["all"],
            "byLabelMatch": res5.get("by_label_match"),
        },
    }


def thumbnail(img, size):
    from PIL import Image

    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return img


def export(df, nbrs, manifest, rows, loader=None):
    """rows: split-manifest rows (queries + corpus) indexed by image_id, for loading pixels.
    With loader=None only the JSON is written."""
    gen = pd.read_csv(generate.OUT_DIR / "descriptions.csv")
    verdicts = pd.read_csv(generate.OUT_DIR / "verdicts.csv")
    qids = sorted(gen.query_id.unique())
    meta = df.set_index("image_id")

    if (DATA_DIR / "queries").exists():
        shutil.rmtree(DATA_DIR / "queries")
    (DATA_DIR / "queries").mkdir(parents=True)
    for qid in qids:
        rec = query_record(qid, nbrs, df, gen, verdicts, manifest)
        (DATA_DIR / "queries" / f"{qid}.json").write_text(json.dumps(rec, ensure_ascii=False))

    order = {"micrograph": 0, "photo": 1, "diagram": 2}
    gallery = sorted(
        ({"id": q, "kind": meta.image_type[q]} for q in qids),
        key=lambda x: (order.get(x["kind"], 3), x["id"]),
    )
    (DATA_DIR / "gallery.json").write_text(json.dumps(gallery))

    tcr4 = json.loads((claims.OUT_DIR / "tcr.json").read_text())
    res5 = json.loads((generate.OUT_DIR / "results.json").read_text())
    (DATA_DIR / "summary.json").write_text(json.dumps(summary(tcr4, res5), indent=2))

    if loader is None:
        return {"queries": len(qids), "images": 0}
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {q: QUERY_PX for q in qids}
    for i in nbrs[nbrs.query_id.isin(qids)].image_id:
        wanted.setdefault(i, NEIGHBOUR_PX)
    written, failed = 0, []
    for iid, px in wanted.items():
        path = IMG_DIR / f"{iid}.webp"
        if path.exists():
            continue
        r = rows.loc[iid].to_dict() | {"image_id": iid}
        try:
            img = loader.load(r)
        except Exception as e:  # one unreadable image shouldn't stop the export
            print(f"  skipped {iid} ({r['source']}): {e}")
            failed.append(iid)
            continue
        thumbnail(img, px).save(path, "WEBP", quality=80, method=6)
        written += 1
        if written % 200 == 0:
            print(f"  {written} thumbnails")
    return {"queries": len(qids), "images": len(wanted), "new_images": written, "failed": failed}
