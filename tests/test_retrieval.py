import numpy as np
import pandas as pd

from pathscope.organs import organs
from pathscope.retrieval import arch_crossmodal, topk


def unit(rows):
    a = np.array(rows, dtype="float32")
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_topk_order():
    C = unit([[1, 0], [0.9, 0.1], [0, 1]])
    idx, s = topk(unit([[1, 0]]), C, 2)
    assert idx.tolist() == [[0, 1]]
    assert s[0, 0] >= s[0, 1]


def test_organs():
    assert organs("Glomerulus in a renal biopsy") == {"kidney"}
    assert organs("bone marrow aspirate") == {"bone_marrow"}
    assert "bladder" not in organs("opened gallbladder with stones")
    assert organs("lymphocytes and smooth muscle") == set()
    assert organs(None) == set()


def test_crossmodal_perfect_and_shared_caption():
    # three images, the first two share one caption (a two-panel figure)
    X = unit([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]])
    df = pd.DataFrame({"source": ["arch_books"] * 3, "text": ["a", "a", "b"]})
    T = unit([[1, 0.05, 0], [0, 0, 1]])
    res = arch_crossmodal(df, X, T, ["a", "b"], ns=(1,))
    assert res["image_to_text"]["R@1"] == 100.0
    assert res["text_to_image"]["R@1"] == 100.0
