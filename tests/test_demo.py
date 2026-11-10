from pathscope import demo


def test_source_of_picks_the_caption_with_most_shared_words():
    caps = ["lung; respiratory; sarcoidosis", "liver; hepatobiliary; acute rejection"]
    assert demo.source_of("The organ is the liver with acute rejection.", caps) == 2


def test_source_of_returns_none_below_threshold():
    caps = ["lung; respiratory", "kidney; urinary"]
    assert demo.source_of("The stain is hematoxylin and eosin.", caps) is None
    assert demo.source_of("", caps) is None


def test_summary_keeps_headline_numbers():
    rates = {
        "claims": 10,
        "supported": 6,
        "contradicted": 2,
        "silent": 2,
        "tcr": 0.75,
        "decided": 0.8,
        "ci95": [0.5, 0.9],
    }
    tcr4 = {
        "queries": 3,
        "overall": rates,
        "by_tier": {t: rates for t in "ABC"},
        "by_tier_and_rank": {"A|1": rates},
    }
    cond = rates | {
        "copy_rate": 0.1,
        "supported_per_description": 1.0,
        "contradicted_per_description": 0.2,
    }
    gain = {"all": {"queries": 3, "gain": 0.1, "ci95": [0.0, 0.2]}}
    res5 = {
        "queries": 3,
        "full": cond,
        "blind": cond,
        "random": cond,
        "multimodal_gain": gain,
        "retrieval_gain": gain,
    }
    out = demo.summary(tcr4, res5)
    assert out["phase4"]["byRank"]["A"] == [0.75, None, None, None, None]
    assert out["phase5"]["conditions"]["blind"]["copyRate"] == 0.1
    assert out["phase5"]["multimodalGain"]["gain"] == 0.1
