import numpy as np
import pandas as pd

from pathscope.data.captions import build_pseudo_captions, yesno_parse
from pathscope.data.leakage import (
    LeakageConfig,
    hamming_all_pairs,
    longest_fragment,
    nearest_phash,
    phash_matrix,
    run_leakage,
    text_overlap,
)

Z = "0" * 64
F = "f" * 64


def test_hamming_basic():
    a = pd.Series([Z, F, "0" * 63 + "1"])
    b = pd.Series([Z, F])
    D = hamming_all_pairs(phash_matrix(a), phash_matrix(b))
    assert D.tolist() == [[0, 256], [256, 0], [1, 255]]


def test_nearest_phash_ignores_missing_hashes():
    q = pd.Series([None, Z])
    c = pd.Series([None, Z, F])
    dist, idx = nearest_phash(q, c, k=2)
    assert dist[0, 0] == 256  # missing query never matches
    assert dist[1, 0] == 0 and idx[1, 0] == 1  # real match found, missing corpus row skipped


def test_yesno_parse_templates():
    assert yesno_parse("does this image show thyroid follicular adenoma") == (
        "thyroid follicular adenoma",
        None,
    )
    assert yesno_parse("does peritoneum show thyroid follicular adenoma") == (
        "thyroid follicular adenoma",
        "peritoneum",
    )
    assert yesno_parse("is cardiovascular present") == ("cardiovascular", None)
    assert yesno_parse("is there necrosis") == ("necrosis", None)
    assert yesno_parse("was talc used") == ("talc used", None)
    assert yesno_parse("what is present") == (None, None)


def test_pseudo_caption_excludes_swapped_negatives():
    m = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "image_id": ["a"] * 4,
            "question": ["q"] * 4,
            "q_norm": [
                "what does this image show",
                "does this image show thyroid follicular adenoma",
                "does peritoneum show thyroid follicular adenoma",
                "is cardiovascular present",
            ],
            "answer": ["thyroid follicular adenoma", "yes", "no", "no"],
            "a_norm": ["thyroid follicular adenoma", "yes", "no", "no"],
            "answer_type": ["open", "yes_no", "yes_no", "yes_no"],
        }
    )
    pc = build_pseudo_captions(m).iloc[0]
    assert pc.gold_fragments == ["thyroid follicular adenoma"]  # deduped
    assert set(pc.neg_fragments) == {"peritoneum", "cardiovascular"}


def test_text_overlap_two_tiers():
    q = pd.Series(["skin kaposi sarcoma", "heart old myocardial infarction with fibrosis he"])
    c = pd.Series(["skin kaposi sarcoma", "heart old myocardial infarction with fibrosis he", "x"])
    r = text_overlap(q, c, min_words=6)
    assert not r.eligible[0] and r.label_match[0] and r.label_c_idx[0] == 0
    assert r.eligible[1] and r.score[1] == 100 and r.c_idx[1] == 1 and not r.label_match[1]


def test_text_overlap_is_not_subset_tolerant():
    # token_set_ratio would give 100 here; token_sort_ratio must not
    q = pd.Series(["liver cirrhosis portal hypertension ascites varices"])
    c = pd.Series(
        ["liver cirrhosis portal hypertension ascites varices plus many other words here now"]
    )
    r = text_overlap(q, c, min_words=5)
    assert r.score[0] < 85


def test_longest_fragment():
    assert longest_fragment(["liver", "liver early chronic passive congestion"], "x") == (
        "liver early chronic passive congestion"
    )
    assert longest_fragment(None, "arch caption") == "arch caption"
    assert longest_fragment([], None) == ""


def _rows(ids, hashes, texts, frags=None, src="s"):
    return pd.DataFrame(
        {
            "image_id": ids,
            "phash": hashes,
            "text": texts,
            "gold_fragments": frags if frags is not None else [None] * len(ids),
            "source": src,
        }
    )


def test_run_leakage_flags_and_manual_override():
    rng = np.random.default_rng(1)
    rand = lambda: rng.bytes(32).hex()  # noqa: E731
    # 72 bits from Z and 64 bits from the "0"*62+"ff" corpus row -> nearest is 64: grey zone
    near = "0" * 46 + "f" * 18
    q = _rows(
        ["q_exact", "q_near", "q_grey", "q_text", "q_clean"],
        [Z, "0" * 60 + "000f", near, rand(), rand()],
        ["a", "b", "c", "heart old myocardial infarction with fibrosis he", "e"],
        src="pathvqa_test",
    )
    c = _rows(
        ["q_exact", "c2", "c3", "c4"],
        [Z, "0" * 62 + "ff", rand(), rand()],
        ["a", "b", "heart old myocardial infarction with fibrosis he", "d"],
        src="pathvqa_train",
    )
    cfg = LeakageConfig(
        manual_near_dups=pd.DataFrame({"image_id": ["q_grey"], "verdict": ["same"]})
    )
    res = run_leakage(q, c, cfg, q_text_col="text", c_text_col="text")
    f = res.flags.set_index("image_id")
    assert f.loc["q_exact", "exact_dup"] and f.loc["q_exact", "leaked"]
    assert f.loc["q_near", "near_dup"] and f.loc["q_near", "phash_dist"] <= 58
    assert f.loc["q_grey", "grey_zone"] and f.loc["q_grey", "near_dup"]  # manual promote
    assert f.loc["q_text", "text_dup"]
    assert not f.loc["q_clean", "leaked"]
    assert res.summary["any"]["n"] == 4 and res.summary["clean_queries"] == 1
