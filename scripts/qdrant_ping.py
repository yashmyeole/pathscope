#!/usr/bin/env python3
"""Keep-alive ping for the Qdrant Cloud free tier.

Free clusters suspend after 7 idle days and are deleted after 28. GitHub Actions runs this
every 3 days (.github/workflows/qdrant-keepalive.yml). Exit code is non-zero on any failure so
a broken cluster shows up as a red run, not a silent one.

Standalone on purpose: uses only `httpx` (no package install needed in CI) and reads
QDRANT_URL / QDRANT_API_KEY from the environment.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import httpx


def main() -> int:
    url = os.environ.get("QDRANT_URL", "").rstrip("/")
    key = os.environ.get("QDRANT_API_KEY", "")
    collection = os.environ.get("QDRANT_COLLECTION", "pathscope_plip_v1")
    if not url:
        print("QDRANT_URL not set", file=sys.stderr)
        return 2

    headers = {"api-key": key} if key else {}
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with httpx.Client(base_url=url, headers=headers, timeout=30) as c:
            r = c.get("/collections")
            r.raise_for_status()
            names = [x["name"] for x in r.json()["result"]["collections"]]
            report = {"ts": ts, "ok": True, "collections": names}
            if collection in names:
                r2 = c.post(f"/collections/{collection}/points/count", json={"exact": True})
                r2.raise_for_status()
                report["points"] = r2.json()["result"]["count"]
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ts": ts, "ok": False, "error": repr(e)}))
        return 1

    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
