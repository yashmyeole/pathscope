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

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
