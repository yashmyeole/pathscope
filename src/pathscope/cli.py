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

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
