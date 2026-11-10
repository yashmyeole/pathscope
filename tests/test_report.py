from pathscope import report


def test_fmt_rounds_half_up():
    assert report.fmt(0.585, "2f") == "0.59"
    assert report.fmt(0.265, "pct") == "27%"
    assert report.fmt(-0.078, "2f") == "−0.08"
    assert report.fmt(0.121, "signed") == "+0.12"
    assert report.fmt([0.5, 0.67], "ci") == "0.50–0.67"
    assert report.fmt([-0.08, 0.23], "ci") == "−0.08 to 0.23"
    assert report.fmt(None, "2f") == "–"


def test_render_fills_paths_and_tables():
    ctx = {"p4": {"by_rank": {"B|1": {"tcr": 0.857}}}, "n": {"rows": 10141}}
    out = report.render(
        "{{p4.by_rank.B|1.tcr|pct}} of {{n.rows|int}}\n{{table:t}}", ctx, {"t": "|a|"}
    )
    assert out == "86% of 10,141\n|a|"


def test_differences_ignores_extra_keys_and_float_noise():
    new = {"a": 0.5001, "b": {"c": 1}, "only_new": 2}
    old = {"a": 0.5, "b": {"c": 2}, "only_old": 3}
    assert report.differences(new, old) == ["b.c"]
