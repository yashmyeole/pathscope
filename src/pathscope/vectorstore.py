"""Thin Qdrant wrapper. Phase 0 only needs `ping`; collection create/upsert/search are here so
Phase 2 has a home and so the ping can report collection state once it exists."""

from __future__ import annotations

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
    if config.QDRANT_COLLECTION in cols:
        cnt = client.count(config.QDRANT_COLLECTION, exact=True).count
        info[config.QDRANT_COLLECTION] = {"points": cnt}
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
