"""The pass/fail plumbing behind every unit's test_unit.py.

A check is a function that takes YOUR function, calls it on small cases whose answer is known,
and either prints a green line or raises an AssertionError that says what went wrong.
The same checks run two ways:
  * in the notebook:  T.check_hu_to_mu(hu_to_mu)          (right after you write it)
  * with pytest:      pytest units/01-data-and-drr         (after you move it into src/xray3d)
"""
from __future__ import annotations

import numpy as np


class TodoNotDone(Exception):
    """Raised when the function under test still raises NotImplementedError."""


def call(fn, *args, **kwargs):
    """Call the student's function; turn NotImplementedError into a clear message."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        raise TodoNotDone(f"{fn.__name__} is still a TODO: {e}") from None


def close(got, want, what: str, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{what}: shape is {got.shape}, expected {want.shape}"
    if not np.allclose(got, want, atol=atol, rtol=rtol):
        diff = np.abs(got - want).max()
        raise AssertionError(f"{what}: off by up to {diff:.4g}\n  got      {np.round(got, 5).tolist()}\n  expected {np.round(want, 5).tolist()}")


def run(name: str, *cases) -> None:
    """Run the cases (zero-arg callables); print a summary line when all pass."""
    for case in cases:
        case()
    msg = f"✔ {name}: all {len(cases)} checks passed"
    try:
        print(msg)
    except UnicodeEncodeError:  # a Windows console with a non-UTF-8 code page
        print("OK" + msg[1:])


def skip_if_missing(module, name: str):
    """For pytest: the function hasn't been moved into src/xray3d yet -> skip, don't fail."""
    import pytest

    fn = getattr(module, name, None)
    if fn is None:
        pytest.skip(f"{module.__name__}.{name} not moved into src/xray3d yet")
    return fn
