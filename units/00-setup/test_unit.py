"""Checks for unit 00 (setup): the warm-up exercise.

  notebook:  import test_unit as T;  T.check_shadow(shadow)
  pytest:    pytest units/00-setup   (tests the copies in src/xray3d/toy.py; skipped until moved)
"""
from __future__ import annotations

import numpy as np

from xray3d import toy
from xray3d.checks import call, close, run, skip_if_missing


def square_image(n: int = 10, lo: int = 3, hi: int = 7) -> np.ndarray:
    """An n x n image of zeros with a filled square (rows and columns lo..hi-1)."""
    img = np.zeros((n, n), dtype=np.uint8)
    img[lo:hi, lo:hi] = 1
    return img


def check_shadow(fn):
    img = square_image()  # rows 3..6 and columns 3..6 are filled: a 4 x 4 square

    def down_the_columns():
        s = np.asarray(call(fn, img, 0, 1.0))
        assert s.shape == (10,), f"one value per column: expected shape (10,), got {s.shape}"
        close(s, [0, 0, 0, 4, 4, 4, 4, 0, 0, 0], "axis 0, 1 mm pixels")

    def pixel_size():
        close(call(fn, img, 0, 0.5), [0, 0, 0, 2, 2, 2, 2, 0, 0, 0], "axis 0, 0.5 mm pixels")

    def along_rows():
        tall = np.zeros((10, 6), np.uint8)
        tall[1:9, 2] = 1  # a vertical line 8 pixels long in column 2
        close(call(fn, tall, 1, 2.0), [0, 2, 2, 2, 2, 2, 2, 2, 2, 0], "axis 1, 2 mm pixels: one value per row")

    run("shadow", down_the_columns, pixel_size, along_rows)


def check_same_shadow(fn):
    img = square_image()

    def is_a_different_binary_image():
        other = np.asarray(call(fn, img))
        assert other.shape == img.shape, f"same size please: {other.shape} vs {img.shape}"
        assert set(np.unique(other)) <= {0, 1}, "the image should only hold 0 and 1"
        assert not np.array_equal(other, img), "that's the same image; find a different shape"

    def casts_the_same_shadow():
        other = np.asarray(call(fn, img))
        assert np.array_equal(other.sum(axis=0), img.sum(axis=0)), (
            f"shadow down the columns differs:\n  yours    {other.sum(axis=0).tolist()}\n  original {img.sum(axis=0).tolist()}")

    run("same_shadow", is_a_different_binary_image, casts_the_same_shadow)


def test_shadow():
    check_shadow(skip_if_missing(toy, "shadow"))


def test_same_shadow():
    check_same_shadow(skip_if_missing(toy, "same_shadow"))
