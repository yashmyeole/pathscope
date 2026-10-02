from PIL import Image

from pathscope.hashing import phash, phash_distance, pixel_md5


def _img(color, size=(64, 64), mode="RGB"):
    return Image.new(mode, size, color)


def test_md5_identical_images_same_id():
    assert pixel_md5(_img("red")) == pixel_md5(_img("red"))


def test_md5_differs_on_pixels_and_size():
    assert pixel_md5(_img("red")) != pixel_md5(_img("blue"))
    assert pixel_md5(_img("red")) != pixel_md5(_img("red", size=(32, 32)))


def test_md5_ignores_container_mode():
    # same pixels encoded as RGB vs RGBA should hash the same after RGB conversion
    assert pixel_md5(_img("red")) == pixel_md5(_img((255, 0, 0, 255), mode="RGBA"))


def test_phash_rescaled_copy_is_close():
    import numpy as np

    rng = np.random.default_rng(0)
    arr = (rng.random((128, 128, 3)) * 255).astype("uint8")
    a = Image.fromarray(arr)
    b = a.resize((96, 96))
    assert phash_distance(phash(a), phash(b)) <= 24  # out of 256 bits
    assert pixel_md5(a) != pixel_md5(b)  # but exact hash differs
