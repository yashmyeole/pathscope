import pandas as pd
import pytest

from pathscope import claims, llm


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)


def fake_extract(system, user):
    import json

    items = json.loads(user)
    return {
        "items": [
            {"id": it["id"], "claims": [{"claim": f"The organ is {it['text']}.", "tier": "A"}]}
            for it in items
        ]
    }


def fake_verify(prompt):
    n = prompt.split("Claims:")[1].strip().count("\n") + 1
    # first claim supported, second contradicted, rest silent
    labels = ["supported", "contradicted"] + ["silent"] * (n - 2)
    return {"verdicts": [{"id": i + 1, "label": lab} for i, lab in enumerate(labels[:n])]}


def setup_data():
    df = pd.DataFrame(
        {
            "image_id": ["q1", "c1", "c2", "c3"],
            "role": ["test", "corpus", "corpus", "corpus"],
            "text": ["liver", "liver", "kidney", "liver"],
            "image_type": ["photo"] * 4,
            "label_match": [False] * 4,
        }
    )
    nbrs = pd.DataFrame(
        {
            "query_id": ["q1"] * 3,
            "rank": [1, 2, 3],
            "image_id": ["c1", "c2", "c3"],
            "score": [0.9, 0.8, 0.7],
            "source": ["pathvqa_train"] * 3,
        }
    )
    manifest = pd.DataFrame(
        {"image_id": ["q1"], "question": ["what organ is this?"], "answer": ["liver"]}
    )
    return df, nbrs, manifest


def test_extract_dedupes_and_caches():
    df, nbrs, _ = setup_data()
    texts = claims.neighbour_texts(nbrs, df)
    assert len(texts) == 2  # c1 and c3 share a caption
    calls = []

    def counting(system, user):
        calls.append(user)
        return fake_extract(system, user)

    out = claims.extract(texts, call=counting)
    assert len(out) == 2 and set(out.tier) == {"A"}
    claims.extract(texts, call=counting)
    assert len(calls) == 1  # second run is all cache


def test_extract_rejects_bad_tier_and_retries():
    def bad(system, user):
        import json

        items = json.loads(user)
        return {
            "items": [{"id": it["id"], "claims": [{"claim": "x", "tier": "D"}]} for it in items]
        }

    out = claims.extract({"a": "one", "b": "two"}, call=bad)
    assert out.empty


def test_verify_and_tcr():
    df, nbrs, manifest = setup_data()
    cl = claims.extract(claims.neighbour_texts(nbrs, df), call=fake_extract)
    v = claims.verify(["q1"], nbrs, cl, df, manifest, call=fake_verify)
    assert list(v.label) == ["supported", "contradicted", "silent"]
    t = claims.tcr(v, df)
    assert t["overall"]["tcr"] == 0.5
    assert t["overall"]["decided"] == round(2 / 3, 3)
    assert t["overall"]["silent"] == 1


def test_kappa():
    assert claims.kappa("aab", "aab") == 1.0
    assert claims.kappa(["s", "s", "c", "c"], ["s", "c", "s", "c"]) == 0.0
