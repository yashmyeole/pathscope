"""Image identity. PathVQA (HF version) ships one row per question with the image
duplicated inline and no image id, so we derive stable ids from pixel content.

Two hashes, two jobs:
  * md5 of decoded RGB bytes  -> exact-duplicate detection and a stable `image_id`
  * pHash (perceptual)        -> near-duplicate detection (re-encoded / rescaled copies),
                                 used in Phase 1 leakage checks
"""

from __future__ import annotations

import hashlib

import imagehash
from PIL import Image


def pixel_md5(img: Image.Image) -> str:
    """MD5 over decoded RGB pixels — identical images give identical ids regardless of
    container/encoding. Mode + size are mixed in so a 1x1 black and 2x2 black differ."""
    rgb = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{rgb.size[0]}x{rgb.size[1]}".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def phash(img: Image.Image, hash_size: int = 16) -> str:
    """Perceptual hash (DCT-based). 16x16 -> 256-bit hex string.
    Hamming distance between two phashes <= ~10% of bits means 'probably the same figure'."""
    return str(imagehash.phash(img.convert("RGB"), hash_size=hash_size))


def phash_distance(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
