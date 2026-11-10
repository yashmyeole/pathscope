"""`pathscope` command line. Each phase gets a subcommand so results regenerate from one
command (guide, Phase 7) instead of from notebook state."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from pathscope import config


def cmd_manifest(args: argparse.Namespace) -> None:
    from pathscope.data.pathvqa import build_manifest, load_pathvqa

    ds = load_pathvqa()
    out = config.MANIFEST_DIR / "pathvqa_manifest.parquet"
    df = build_manifest(ds, out_path=out, limit_per_split=args.limit)
    print(f"wrote {len(df):,} rows -> {out}")


def cmd_stats(args: argparse.Namespace) -> None:
    from pathscope.data.stats import compute_stats, stats_markdown, write_stats

    m = pd.read_parquet(config.MANIFEST_DIR / "pathvqa_manifest.parquet")
    stats = compute_stats(m)
    write_stats(stats, config.STATS_DIR / "pathvqa_stats.json")
    (config.STATS_DIR / "pathvqa_stats.md").write_text(stats_markdown(stats))
    print(stats_markdown(stats))


def cmd_contact_sheet(args: argparse.Namespace) -> None:
    from pathscope.data.contact_sheet import render_contact_sheet, sample_images
    from pathscope.data.pathvqa import load_pathvqa

    ds = load_pathvqa()
    m = pd.read_parquet(config.MANIFEST_DIR / "pathvqa_manifest.parquet")
    sample = sample_images(m, args.split, args.n, args.seed)
    out = config.STATS_DIR / "contact_sheets" / f"pathvqa_{args.split}_{args.n}_seed{args.seed}.png"
    render_contact_sheet(ds, sample, out, title=f"PathVQA {args.split} — {args.n} random images")
    print(f"wrote {out}")


def cmd_ping(args: argparse.Namespace) -> None:
    from pathscope.vectorstore import get_client, ping

    print(json.dumps(ping(get_client()), indent=2))


def cmd_arch_manifest(args: argparse.Namespace) -> None:
    from pathscope.data.arch import build_arch_manifest

    out = config.MANIFEST_DIR / "arch_manifest.parquet"
    df = build_arch_manifest(out_path=out, subsets=tuple(args.subsets), limit=args.limit)
    errs = df.decode_error.notna().sum()
    print(f"wrote {len(df):,} rows ({errs} decode errors) -> {out}")


def cmd_split(args: argparse.Namespace) -> None:
    from pathscope.data.leakage import LeakageConfig
    from pathscope.data.split import SPLIT_DIR, build_split

    pv = pd.read_parquet(config.MANIFEST_DIR / "pathvqa_manifest.parquet")
    arch = pd.read_parquet(config.MANIFEST_DIR / "arch_manifest.parquet")
    cfg = LeakageConfig(phash_threshold=args.phash, text_threshold=args.text)
    summary = build_split(pv, arch, cfg)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote split manifests -> {SPLIT_DIR}")


def cmd_leakage_report(args: argparse.Namespace) -> None:
    from pathscope.data.leakage_report import render_report

    md = render_report()
    out = config.STATS_DIR / "phase1_leakage_report.md"
    out.write_text(md)
    print(md)
    print(f"\nwrote {out}")


def cmd_embed(args: argparse.Namespace) -> None:
    from pathscope.embed import run_embedding

    df = run_embedding(limit=args.limit, batch=args.batch)
    print(df.groupby("role").image_type.value_counts().to_string())
    print(f"embedded {len(df):,} images -> {config.EMBED_DIR}")


def cmd_label_types(args: argparse.Namespace) -> None:
    from pathscope.embed import label_image_types

    path = config.EMBED_DIR / "embeddings.parquet"
    df = pd.read_parquet(path)
    df["image_type"], df["image_type_p"] = label_image_types(df)
    df.to_parquet(path, index=False)
    print(pd.crosstab(df.source, df.image_type, normalize="index").round(2).to_string())


def cmd_index(args: argparse.Namespace) -> None:
    from pathscope.vectorstore import get_client, index_embeddings, self_retrieval_rate

    df = pd.read_parquet(config.EMBED_DIR / "embeddings.parquet")
    client = get_client(timeout=120)
    print(json.dumps(index_embeddings(client, df), indent=2))
    rate = self_retrieval_rate(client, df)
    print(f"self-retrieval (200 corpus images, top-1 is itself): {rate:.1%}")


def cmd_neighbors(args: argparse.Namespace) -> None:
    import time

    from pathscope.vectorstore import get_client, neighbors

    client = get_client()
    t = time.time()
    own, hits = neighbors(client, args.image_id, k=args.k)
    dt = time.time() - t
    print(f"[{own['role']}] {own['source']}: {own['text'][:150]}\n")
    for h in hits:
        p = h.payload
        print(f"{h.score:.3f}  [{p['source']}, {p['image_type']}]  {p['text'][:120]}")
    print(f"\n{dt * 1000:.0f} ms")


def cmd_retrieval_check(args: argparse.Namespace) -> None:
    from pathscope import retrieval as r

    df, X = r.load_embeddings()
    nbrs = r.neighbour_table(df, X, k=5)
    nbrs.to_csv(r.OUT_DIR / "neighbours.csv", index=False)

    metrics = {"automatic": r.retrieval_stats(df, X, nbrs)}
    caps, T = r.arch_text_vectors()
    metrics["arch_crossmodal"] = r.arch_crossmodal(df, X, T, caps)
    if (r.OUT_DIR / "review.csv").exists():
        metrics["hand_review"] = r.review_summary()
    r.save_json(metrics, "metrics.json")
    print(json.dumps(metrics, indent=2))

    if args.sheets:
        for p in r.render_sheets(r.sample_queries(df, args.n), nbrs, df):
            print("wrote", p)


def _phase4_inputs():
    from pathscope import retrieval as r

    df, _ = r.load_embeddings()
    nbrs = pd.read_csv(r.OUT_DIR / "neighbours.csv")
    return df, nbrs


def cmd_claims(args: argparse.Namespace) -> None:
    from pathscope import claims as c

    c.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.check:
        ref = pd.read_csv(c.OUT_DIR / "extract_reference.csv")
        texts = dict(ref.drop_duplicates("text_id")[["text_id", "text"]].to_numpy())
        got = c.extract(texts, batch=args.batch)
        got.to_csv(c.OUT_DIR / "extract_check.csv", index=False)
        res = c.compare_extraction(got, ref)
        (c.OUT_DIR / "extract_check.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    df, nbrs = _phase4_inputs()
    texts = c.neighbour_texts(nbrs, df)
    print(f"{len(texts)} unique neighbour captions")
    try:
        c.extract(texts, batch=args.batch, limit=args.limit)
    finally:
        out = c.extract(texts, limit=0)  # whatever is cached so far
        out.to_csv(c.OUT_DIR / "claims.csv", index=False)
        print(f"{out.text_id.nunique()}/{len(texts)} captions done, {len(out)} claims")


def cmd_verify(args: argparse.Namespace) -> None:
    from pathscope import claims as c

    df, nbrs = _phase4_inputs()
    cl = pd.read_csv(c.OUT_DIR / "claims.csv")
    manifest = pd.read_parquet(config.MANIFEST_DIR / "pathvqa_manifest.parquet")
    # only queries whose five neighbour captions have all been through extraction
    done = c.extracted_ids(c.neighbour_texts(nbrs, df))
    tid = nbrs.image_id.map(df.set_index("image_id").text).map(c.text_id)
    ready = nbrs.assign(ok=tid.isin(done)).groupby("query_id").ok.all()
    qids = [q for q in dict.fromkeys(nbrs.query_id) if ready[q]]
    print(f"{len(qids)}/{len(ready)} queries have all neighbour claims")
    try:
        c.verify(qids, nbrs, cl, df, manifest, limit=args.limit)
    finally:
        v = c.verify(qids, nbrs, cl, df, manifest, limit=0)
        v.drop(columns=["text"]).to_csv(c.OUT_DIR / "verdicts.csv", index=False)
        print(f"{v.query_id.nunique()}/{len(ready)} queries verified, {len(v)} claim checks")


def cmd_tcr(args: argparse.Namespace) -> None:
    from pathscope import claims as c

    df, _ = _phase4_inputs()
    v = pd.read_csv(c.OUT_DIR / "verdicts.csv")
    res = c.tcr(v, df)

    review = config.ROOT / "eval" / "phase3" / "review.csv"
    if review.exists():
        rv = pd.read_csv(review).set_index("query_id").same_case_in_top5
        sub = v[v.query_id.isin(rv.index)]
        res["reviewed_50_by_same_case"] = {
            f"{t}|{flag}": c._rates(g)
            for (t, flag), g in sub.assign(flag=sub.query_id.map(rv)).groupby(["tier", "flag"])
        }

    labels = c.OUT_DIR / "judge_sample.csv"
    if labels.exists():
        # refresh the judge column from the latest verdicts, keep the hand labels
        js = pd.read_csv(labels)
        keys = ["query_id", "neighbour_id", "claim"]
        latest = v.drop_duplicates(keys).set_index(keys).label
        js["judge"] = [
            latest.get(tuple(k), j) for k, j in zip(js[keys].values, js.judge, strict=True)
        ]
        js.to_csv(labels, index=False)
        for col in ("label_yash", "label_ref"):
            if col in js and js[col].notna().all():
                res[f"judge_kappa_vs_{col}"] = {
                    "n": len(js),
                    "kappa": c.kappa(js.judge, js[col]),
                    "agreement": round(float((js.judge == js[col]).mean()), 3),
                }
    else:
        manifest = pd.read_parquet(config.MANIFEST_DIR / "pathvqa_manifest.parquet")
        c.judge_sample(v, manifest).assign(label_yash=None).to_csv(labels, index=False)
        print(f"wrote {labels}: label it blind (fill label_yash), then rerun")

    (c.OUT_DIR / "tcr.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def _phase5_sets(n):
    from pathscope import generate as g

    df, nbrs = _phase4_inputs()
    qids = g.sample_queries(nbrs, n)
    return df, qids, g.caption_sets(qids, nbrs, df)


def cmd_generate(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    from pathscope import generate as g
    from pathscope.data.pairs import ImageLoader

    df, qids, sets = _phase5_sets(args.n)
    test = load_dataset(config.PATHVQA_HF_ID, split="test", cache_dir=str(config.HF_CACHE_DIR))
    loader = ImageLoader({"test": test})
    rows = pd.read_parquet(config.MANIFEST_DIR / "split_v1" / "queries.parquet")
    rows = rows.set_index("image_id")

    def load_image(qid):
        return loader.load(rows.loc[qid].to_dict())

    g.OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        g.generate(qids, sets, load_image, limit=args.limit)
    finally:
        out = g.generate(qids, sets, load_image, limit=0)
        out.to_csv(g.OUT_DIR / "descriptions.csv", index=False)
        print(f"{len(out)}/{len(qids) * len(g.CONDITIONS)} descriptions")


def cmd_gen_score(args: argparse.Namespace) -> None:
    from pathscope import generate as g

    df, _ = _phase4_inputs()
    gen = pd.read_csv(g.OUT_DIR / "descriptions.csv")
    manifest = pd.read_parquet(config.MANIFEST_DIR / "pathvqa_manifest.parquet")
    v = g.score(gen, manifest, args.extract_limit, args.verify_limit)
    v.to_csv(g.OUT_DIR / "verdicts.csv", index=False)
    if v.empty:
        print("nothing verified yet")
        return
    res = g.summary(v, df)
    (g.OUT_DIR / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pathscope")
    sp = p.add_subparsers(dest="cmd", required=True)

    a = sp.add_parser("manifest", help="Phase 0: decode PathVQA, hash images, write manifest")
    a.add_argument("--limit", type=int, default=None, help="rows per split (for quick tests)")
    a.set_defaults(fn=cmd_manifest)

    a = sp.add_parser("stats", help="Phase 0: dataset statistics from the manifest")
    a.set_defaults(fn=cmd_stats)

    a = sp.add_parser("contact-sheet", help="Phase 0: render N random images with their Q/A")
    a.add_argument("--split", default="train", choices=config.PATHVQA_SPLITS)
    a.add_argument("--n", type=int, default=20)
    a.add_argument("--seed", type=int, default=config.SEED)
    a.set_defaults(fn=cmd_contact_sheet)

    a = sp.add_parser("ping", help="Check the Qdrant cluster responds")
    a.set_defaults(fn=cmd_ping)

    a = sp.add_parser("arch-manifest", help="Phase 1: hash ARCH images straight from the zips")
    a.add_argument("--subsets", nargs="+", default=["pubmed", "books"], choices=["pubmed", "books"])
    a.add_argument("--limit", type=int, default=None, help="images per subset (quick tests)")
    a.set_defaults(fn=cmd_arch_manifest)

    a = sp.add_parser("split", help="Phase 1: leakage checks + clean queries/dev/corpus manifests")
    a.add_argument("--phash", type=int, default=58, help="pHash Hamming threshold (of 256 bits)")
    a.add_argument("--text", type=float, default=85.0, help="token_sort_ratio threshold")
    a.set_defaults(fn=cmd_split)

    a = sp.add_parser("leakage-report", help="Phase 1: render the leakage report from split_v1")
    a.set_defaults(fn=cmd_leakage_report)

    a = sp.add_parser("embed", help="Phase 2: PLIP-embed corpus/test/dev -> embeddings.parquet")
    a.add_argument("--limit", type=int, default=None, help="images per role (quick test)")
    a.add_argument("--batch", type=int, default=32)
    a.set_defaults(fn=cmd_embed)

    a = sp.add_parser("label-types", help="Phase 2: re-fit image-type labels on embeddings.parquet")
    a.set_defaults(fn=cmd_label_types)

    a = sp.add_parser("index", help="Phase 2: upload embeddings.parquet to Qdrant and verify")
    a.set_defaults(fn=cmd_index)

    a = sp.add_parser("neighbors", help="Phase 2: nearest corpus neighbours of an image id")
    a.add_argument("image_id")
    a.add_argument("-k", type=int, default=5)
    a.set_defaults(fn=cmd_neighbors)

    a = sp.add_parser("retrieval-check", help="Phase 3: neighbour stats, ARCH recall, review")
    a.add_argument("--sheets", action="store_true", help="also render the query/neighbour sheets")
    a.add_argument("--n", type=int, default=50, help="queries on the sheets")
    a.set_defaults(fn=cmd_retrieval_check)

    a = sp.add_parser("claims", help="Phase 4: neighbour captions -> tiered claims (Groq)")
    a.add_argument("--check", action="store_true", help="only the 30 hand-referenced captions")
    a.add_argument("--limit", type=int, default=None, help="max captions to send this run")
    a.add_argument("--batch", type=int, default=12)
    a.set_defaults(fn=cmd_claims)

    a = sp.add_parser("verify", help="Phase 4: check claims against each query's Q/A (Gemini)")
    a.add_argument("--limit", type=int, default=None, help="max new queries this run")
    a.set_defaults(fn=cmd_verify)

    a = sp.add_parser("tcr", help="Phase 4: TCR tables + judge agreement")
    a.set_defaults(fn=cmd_tcr)

    a = sp.add_parser("generate", help="Phase 5: descriptions for full / blind / random (Gemini)")
    a.add_argument("--n", type=int, default=250, help="number of test queries")
    a.add_argument("--limit", type=int, default=None, help="max new descriptions this run")
    a.set_defaults(fn=cmd_generate)

    a = sp.add_parser("gen-score", help="Phase 5: claims + checks for the descriptions, gains")
    a.add_argument("--extract-limit", type=int, default=None)
    a.add_argument("--verify-limit", type=int, default=None)
    a.set_defaults(fn=cmd_gen_score)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
