from pathscope.data.pathvqa import answer_type, normalize_text


def test_normalize():
    assert normalize_text("  Is this H&E?  ") == "is this he"


def test_answer_type():
    assert answer_type("Yes") == "yes_no"
    assert answer_type("no.") == "yes_no"
    assert answer_type("adenocarcinoma") == "short"
    assert answer_type("in the canals of hering") == "open"
