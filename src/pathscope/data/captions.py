"""Reconstruct a per-image "pseudo-caption" for PathVQA.

PathVQA's HF release has no captions. But every image's Q/A cluster was generated from one
textbook caption, and the pieces are recoverable:

  * open-ended answers ("what does this image show?" -> the caption fragment)
  * short answers (organ / system: "where is this?" -> "oral")
  * the subject of each *yes* question ("does this image show thyroid, follicular adenoma?" -> yes)

The subject of a *no* question is a random distractor swapped in by the generator
("does peritoneum show thyroid, follicular adenoma?" -> no), so those are deliberately excluded
from the caption — but kept separately as known-false statements (useful hard negatives later).

Output per image_id: gold_text (joined, de-duplicated), gold_fragments (list), neg_fragments (list).
"""

from __future__ import annotations

import re

import pandas as pd

_IMAGE_WORDS = r"(?:this|the) (?:image|picture|slide|section|photo|photograph|figure|specimen)"

# "does this image show X"  -> content X
_P_IMAGE_SHOW = re.compile(rf"^does {_IMAGE_WORDS} (?:show|reveal|demonstrate|display) (.+)$")
_P_IMAGE_IS = re.compile(rf"^is {_IMAGE_WORDS} (?:showing|of|an? example of) (.+)$")
# "does <subject> show X" — the generator builds negatives by swapping <subject> for a random
# phrase, so <subject> is what is false in a "no" row, while X is usually still true.
_P_SUBJ_SHOW = re.compile(r"^(?:does|do) (.+?) (?:show|shows|reveal|demonstrate|display) (.+)$")
_P_THERE = re.compile(r"^(?:is|are) there (.+)$")
_P_PRESENT = re.compile(r"^(?:is|are) (.+?) present$")
_P_IS_THIS = re.compile(r"^is (?:this|it) (?:an? )?(.+)$")
# Last resort: strip the auxiliary verb and take the rest as the statement.
_P_AUX = re.compile(r"^(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would) (.+)$")


def yesno_parse(q_norm: str) -> tuple[str | None, str | None]:
    """Parse a normalised yes/no question into (content, swapped_subject).

    content         — the phrase asserted about the image ("thyroid follicular adenoma")
    swapped_subject — for "does <X> show <content>" templates where <X> is not the image itself,
                      the <X> phrase (what the generator swapped in to make a negative)
    Returns (None, None) only for questions no template recognises.
    """
    q = q_norm.strip()
    if m := _P_IMAGE_SHOW.match(q):
        return m.group(1).strip(), None
    if m := _P_IMAGE_IS.match(q):
        return m.group(1).strip(), None
    if m := _P_SUBJ_SHOW.match(q):
        return m.group(2).strip(), m.group(1).strip()
    if m := _P_THERE.match(q):
        return m.group(1).strip(), None
    if m := _P_PRESENT.match(q):
        return m.group(1).strip(), None
    if m := _P_IS_THIS.match(q):
        return m.group(1).strip(), None
    if m := _P_AUX.match(q):
        return m.group(1).strip(), None
    return None, None


def yesno_subject(q_norm: str) -> str | None:
    """Back-compat helper: the asserted content phrase, or None."""
    return yesno_parse(q_norm)[0]


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


def build_pseudo_captions(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per (split, image_id).

    Columns: split, image_id, n_qa, gold_fragments, neg_fragments, gold_text, n_open, n_yes, n_no
    """
    rows = []
    for (split, image_id), g in manifest.groupby(["split", "image_id"], sort=False):
        gold: list[str] = []
        neg: list[str] = []
        n_open = n_yes = n_no = 0
        # Longest open answers first so gold_text starts with the most caption-like fragment.
        g_sorted = g.assign(_len=g.answer.str.len()).sort_values("_len", ascending=False)
        for r in g_sorted.itertuples(index=False):
            if r.answer_type == "yes_no":
                content, swapped = yesno_parse(r.q_norm)
                if r.a_norm == "yes":
                    n_yes += 1
                    if content:
                        gold.append(content)
                    if swapped:  # "does lesion show X? yes" -> "lesion" is also asserted
                        gold.append(swapped)
                else:
                    n_no += 1
                    # In a swapped-subject negative the subject is the false part; the content
                    # is usually true but we do not promote it to gold on the strength of a "no".
                    if swapped:
                        neg.append(swapped)
                    elif content:
                        neg.append(content)
            else:
                n_open += 1
                gold.append(r.answer)
        gold = _dedup_keep_order(gold)
        neg = _dedup_keep_order(neg)
        rows.append(
            {
                "split": split,
                "image_id": image_id,
                "n_qa": len(g),
                "gold_fragments": gold,
                "neg_fragments": neg,
                "gold_text": "; ".join(gold),
                "n_open": n_open,
                "n_yes": n_yes,
                "n_no": n_no,
            }
        )
    return pd.DataFrame(rows)
