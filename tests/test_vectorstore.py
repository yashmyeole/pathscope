"""Index / self-retrieval / neighbours against an in-memory Qdrant (no cluster needed)."""

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient

from pathscope import config
from pathscope import vectorstore as vs


def _df(n_corpus=30, n_query=5, dim=config.EMBED_DIM, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_corpus + n_query, dim)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return pd.DataFrame(
        {
            "role": ["corpus"] * n_corpus + ["test"] * n_query,
            "image_id": [f"{i:032x}" for i in range(n_corpus + n_query)],
            "source": "pathvqa_train",
            "text": "a caption",
            "image_type": "micrograph",
            "figure_id": None,
            "label_match": None,
            "embedding": list(v),
        }
    )


@pytest.fixture
def indexed():
    client, df = QdrantClient(":memory:"), _df()
    return client, df, vs.index_embeddings(client, df)


def test_counts_match_and_split_into_two_collections(indexed):
    _, _, counts = indexed
    assert counts[config.QDRANT_COLLECTION] == {"expected": 30, "in_qdrant": 30}
    assert counts[config.QDRANT_QUERY_COLLECTION] == {"expected": 5, "in_qdrant": 5}


def test_reindexing_is_idempotent(indexed):
    client, df, _ = indexed
    assert vs.index_embeddings(client, df)[config.QDRANT_COLLECTION]["in_qdrant"] == 30


def test_self_retrieval(indexed):
    client, df, _ = indexed
    assert vs.self_retrieval_rate(client, df, n=30) == 1.0


def test_query_neighbors_are_corpus_only(indexed):
    client, df, _ = indexed
    own, hits = vs.neighbors(client, df.image_id.iloc[-1], k=5)
    assert own["role"] == "test" and len(hits) == 5
    assert all(h.payload["role"] == "corpus" for h in hits)


def test_corpus_image_never_returns_itself(indexed):
    client, df, _ = indexed
    _, hits = vs.neighbors(client, df.image_id.iloc[0], k=5)
    assert df.image_id.iloc[0] not in {str(h.id).replace("-", "") for h in hits}


def test_unknown_id_raises(indexed):
    client, _, _ = indexed
    with pytest.raises(KeyError):
        vs.neighbors(client, "f" * 32)
