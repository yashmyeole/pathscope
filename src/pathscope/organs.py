"""Very rough organ tagging from caption text.

Used only to check whether a retrieved neighbour is about the same organ as the query. Words that
show up in many organs (lymphocytes, smooth muscle, vessels, tubules, villi) are left out on
purpose, so a lot of captions get no tag at all. That's fine, we only score pairs where both
sides have one.
"""

import re

ORGAN_PATTERNS = {
    "liver": r"liver|hepat|cirrho|portal tract",
    "kidney": r"kidney|renal|glomerul|nephr",
    "lung": r"lung|pulmonar|alveol|bronch|pleura|pneum",
    "heart": r"heart|cardiac|myocard|endocard|pericard|coronary",
    "cns": r"brain|cerebr|cerebell|spinal cord|mening|glio|astrocyt",
    "skin": r"skin|epiderm|dermis|dermal|cutaneous|keratinocyt",
    "breast": r"breast|mammary",
    "colon": r"colon|colorectal|rectum|rectal|cecum|caecum|appendi|large bowel",
    "small_bowel": r"small intestin|small bowel|duoden|jejun|ileum|ileal",
    "stomach": r"stomach|gastric",
    "esophagus": r"esophag|oesophag",
    "pancreas": r"pancrea|islets? of langerhans",
    "thyroid": r"thyroid",
    "adrenal": r"adrenal",
    "prostate": r"prostat",
    "testis": r"testis|testes|testic|seminiferous",
    "ovary": r"ovary|ovari",
    "uterus": r"uter|endometri|myometri|cervix",
    "placenta": r"placent|chorionic",
    "bone": r"bone(?!\s+marrow)|osteo",
    "bone_marrow": r"marrow",
    "lymph_node": r"lymph nodes?",
    "spleen": r"spleen|splenic",
    "oral": r"oral cavity|tongue|salivary|gingiv",
    "bladder": r"(?<!gall )bladder|urothel",
    "gallbladder": r"gallbladder|gall bladder",
    "eye": r"\beyes?\b|retina|cornea|conjunctiv",
    "aorta_artery": r"aort|arter",
}

_compiled = {k: re.compile(r"\b(?:" + v + ")", re.I) for k, v in ORGAN_PATTERNS.items()}


def organs(text) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {name for name, rx in _compiled.items() if rx.search(text)}
