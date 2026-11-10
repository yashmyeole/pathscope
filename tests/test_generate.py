import json

import pandas as pd
import pytest
from PIL import Image

from pathscope import generate as g
from pathscope import llm


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)


def data():
    ids = [f"c{i}" for i in range(12)]
    df = pd.DataFrame(
        {
            "image_id": ["q1", *ids],
            "role": ["test"] + ["corpus"] * 12,
            "source": ["pathvqa_test"] + ["pathvqa_train"] * 6 + ["arch_books"] * 6,
            "text": ["q"] + [f"caption {i}" for i in range(12)],
            "image_type": ["photo"] * 13,
        }
    )
    nbrs = pd.DataFrame(
        {
            "query_id": ["q1"] * 5,
            "rank": [1, 2, 3, 4, 5],
            "image_id": ["c0", "c1", "c2", "c6", "c7"],
            "source": ["pathvqa_train"] * 3 + ["arch_books"] * 2,
        }
    )
    return df, nbrs


def test_random_captions_keep_source_and_skip_neighbours():
    df, nbrs = data()
    sets = g.caption_sets(["q1"], nbrs, df)
    assert sets[("q1", "full")] == sets[("q1", "blind")]
    rand = sets[("q1", "random")]
    assert len(rand) == 5
    src = df.set_index("text").source
    assert [src[c] for c in rand] == list(nbrs.source)
    assert not set(rand) & set(sets[("q1", "full")])


def test_generate_sends_image_only_when_it_should():
    df, nbrs = data()
    sets = g.caption_sets(["q1"], nbrs, df)
    seen = []

    def fake(prompt, image=None):
        seen.append(image is not None)
        return {"description": "The organ is the liver."}

    out = g.generate(["q1"], sets, lambda q: Image.new("RGB", (8, 8)), call=fake)
    assert list(out.condition) == ["full", "blind", "random"]
    assert seen == [True, False, True]
    g.generate(["q1"], sets, lambda q: Image.new("RGB", (8, 8)), call=fake)
    assert len(seen) == 3  # cached


def test_copied():
    caps = ["Liver with marked fatty change, H&E"]
    assert g.copied("The liver shows marked fatty change.", caps)
    assert not g.copied("The stain is Congo red.", caps)


def test_paired_gain():
    rows = []
    for q in range(4):
        rows += [(f"q{q}", "full", "A", "supported", False)] * 3
        rows += [
            (f"q{q}", "blind", "A", "supported", False),
            (f"q{q}", "blind", "A", "contradicted", False),
        ]
    v = pd.DataFrame(rows, columns=["query_id", "condition", "tier", "label", "copied"])
    res = g.paired_gain(v, "full", "blind")
    assert res["gain"] == 0.5 and res["queries"] == 4
    assert json.dumps(res)
