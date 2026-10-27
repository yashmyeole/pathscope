"""Thin Qdrant wrapper: ping, collection create, batched upsert, search, and the Phase 2 helpers
(index a parquet of embeddings, self-retrieval check, neighbours of an image id)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from pathscope import config


def get_client(
    url: str | None = None, api_key: str | None = None, timeout: int = 30
) -> QdrantClient:
    url = url or config.QDRANT_URL
    api_key = api_key if api_key is not None else config.QDRANT_API_KEY
    if not url:
        raise RuntimeError("QDRANT_URL is not set (see .env.example)")
    return QdrantClient(url=url, api_key=api_key or None, timeout=timeout)


def ping(client: QdrantClient) -> dict:
    """Cheap, real traffic: list collections and (if present) count our collection.
    Returning counts rather than just 'ok' means the keep-alive log doubles as a sanity check."""
    cols = [c.name for c in client.get_collections().collections]
    info: dict = {"collections": cols}
    for name in (config.QDRANT_COLLECTION, config.QDRANT_QUERY_COLLECTION):
        if name in cols:
            info[name] = {"points": client.count(name, exact=True).count}
    return info


def ensure_collection(
    client: QdrantClient, name: str = config.QDRANT_COLLECTION, dim: int = config.EMBED_DIM
) -> None:
    if name in [c.name for c in client.get_collections().collections]:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


def upsert_batches(
    client: QdrantClient,
    points: list[qm.PointStruct],
    name: str = config.QDRANT_COLLECTION,
    batch: int = 100,
) -> int:
    """Upload in small batches and return the final exact count so the caller can assert it
    matches the parquet row count (guide, Phase 2: 'do not trust the upload until checked')."""
    for i in range(0, len(points), batch):
        client.upsert(collection_name=name, points=points[i : i + batch], wait=True)
    return client.count(name, exact=True).count


def search(
    client: QdrantClient,
    vector: list[float],
    k: int = 5,
    name: str = config.QDRANT_COLLECTION,
    exclude_ids: list | None = None,
):
    flt = None
    if exclude_ids:
        flt = qm.Filter(must_not=[qm.HasIdCondition(has_id=exclude_ids)])
    return client.query_points(
        collection_name=name, query=vector, limit=k, query_filter=flt, with_payload=True
    ).points


# ---------------------------------------------------------------- Phase 2: index + neighbours
PAYLOAD_COLS = ["role", "source", "image_id", "text", "image_type", "figure_id", "label_match"]


def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def to_points(df: pd.DataFrame) -> list[qm.PointStruct]:
    """The point id IS the image's pixel-MD5 (a valid 32-hex UUID), so any image id can be
    looked up directly and ids are stable across re-indexing."""
    cols = [c for c in PAYLOAD_COLS if c in df.columns]
    return [
        qm.PointStruct(
            id=r["image_id"],
            vector=np.asarray(r["embedding"], dtype="float32").tolist(),
            payload={c: _clean(r[c]) for c in cols},
        )
        for r in df.to_dict("records")
    ]


def index_embeddings(client: QdrantClient, df: pd.DataFrame) -> dict:
    """Corpus rows -> corpus collection; test/dev rows -> query collection. Returns the exact
    server-side counts next to the expected ones; raises if any point went missing."""
    out = {}
    for name, part in (
        (config.QDRANT_COLLECTION, df[df.role == "corpus"]),
        (config.QDRANT_QUERY_COLLECTION, df[df.role != "corpus"]),
    ):
        ensure_collection(client, name)
        got = upsert_batches(client, to_points(part), name)
        out[name] = {"expected": len(part), "in_qdrant": got}
        if got != len(part):
            raise RuntimeError(f"{name}: uploaded {len(part)} but Qdrant holds {got}")
    return out


def self_retrieval_rate(client: QdrantClient, df: pd.DataFrame, n: int = 200, seed: int = 594):
    """Fraction of sampled corpus images whose own vector returns itself as the top hit."""
    s = df[df.role == "corpus"].sample(min(n, (df.role == "corpus").sum()), random_state=seed)
    hits = 0
    for r in s.to_dict("records"):
        top = search(client, np.asarray(r["embedding"]).tolist(), k=1)[0]
        hits += str(top.id).replace("-", "") == r["image_id"]
    return hits / len(s)


def neighbors(client: QdrantClient, image_id: str, k: int = 5):
    """Five nearest corpus neighbours of a stored image (query or corpus). A corpus image never
    returns itself. Returns (the image's own payload, hits)."""
    for name in (config.QDRANT_QUERY_COLLECTION, config.QDRANT_COLLECTION):
        got = client.retrieve(name, ids=[image_id], with_vectors=True, with_payload=True)
        if got:
            ex = [image_id] if name == config.QDRANT_COLLECTION else None
            return got[0].payload, search(client, got[0].vector, k=k, exclude_ids=ex)
    raise KeyError(f"image id {image_id} not found in either collection")
