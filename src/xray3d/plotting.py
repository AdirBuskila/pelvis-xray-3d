"""Small display helpers, so notebook cells stay about the idea and not about matplotlib.

Convention: images are shown with imshow(origin="upper"): row 0 at the top. If an image comes
out upside down, flip it in *your* code (that's usually an orientation question worth asking).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def show(img, title: str = "", cmap: str = "gray", vmin=None, vmax=None, aspect=None, ax=None, colorbar=False):
    """Show one 2D array. `aspect` = row spacing / column spacing (mm) for non-square pixels."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.asarray(img), cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    if own:
        plt.show()
    return ax


def show_row(images, titles=None, cmap: str = "gray", vmin=None, vmax=None, size: float = 4.5, aspects=None):
    """Show several 2D arrays side by side."""
    images = list(images)
    titles = titles or [""] * len(images)
    aspects = aspects or [None] * len(images)
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size))
    axes = np.atleast_1d(axes)
    for ax, img, t, a in zip(axes, images, titles, aspects):
        show(img, t, cmap=cmap, vmin=vmin, vmax=vmax, aspect=a, ax=ax)
    plt.tight_layout()
    plt.show()


def histogram(values, bins: int = 200, range=None, title: str = "", xlabel: str = "", log: bool = True, marks=None):
    """Histogram of a big array (e.g. every HU value in a CT). `marks` = {label: x} vertical lines."""
    v = np.asarray(values).ravel()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(v, bins=bins, range=range, color="#4a90c2", log=log)
    for label, x in (marks or {}).items():
        ax.axvline(x, color="#e0a040", lw=1)
        ax.text(x, ax.get_ylim()[1], " " + label, rotation=90, va="top", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (log)" if log else "count")
    plt.tight_layout()
    plt.show()


def scatter2d(points_list, labels=None, title: str = "", equal: bool = True):
    """Scatter several (N, 2) point sets, e.g. the same 3D points under two projections."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for i, pts in enumerate(points_list):
        pts = np.asarray(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=12, label=None if labels is None else labels[i])
    if labels:
        ax.legend()
    if equal:
        ax.set_aspect("equal")
    ax.invert_yaxis()  # image convention: v grows downwards
    ax.set_title(title)
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    plt.show()
