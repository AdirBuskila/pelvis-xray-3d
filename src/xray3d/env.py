"""Where are we running, and where is the data?

The same notebook runs in three places. The only things that differ are the folder the
repo lives in and the folder the data lives in; everything else asks this module.
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


def where() -> str:
    """Return "colab", "kaggle" or "local"."""
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def repo_root() -> Path:
    """The repo folder: walk up from this file until pyproject.toml shows up."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not find the repo root (no pyproject.toml above " + str(p) + ")")


def data_root() -> Path:
    """Folder that holds all datasets. Never inside git (see .gitignore).

    Override with the environment variable XRAY3D_DATA. Defaults:
      local   <repo>/data
      colab   /content/data         (wiped when the runtime ends)
      kaggle  /kaggle/working/data  (kept as notebook output up to ~20 GB)
    """
    if os.environ.get("XRAY3D_DATA"):
        return Path(os.environ["XRAY3D_DATA"])
    return {
        "colab": Path("/content/data"),
        "kaggle": Path("/kaggle/working/data"),
    }.get(where(), repo_root() / "data")


def processed_root() -> Path:
    """Folder for datasets WE build (unit 02's 64^3 cubes + DRR pairs). Also never in git.

    Override with XRAY3D_PROCESSED. Default: <data_root>/processed
    """
    if os.environ.get("XRAY3D_PROCESSED"):
        return Path(os.environ["XRAY3D_PROCESSED"])
    # Kaggle: the built dataset is attached as a Kaggle dataset named "pelvis-xray3d-v1" (read-only input)
    kaggle_input = Path("/kaggle/input/pelvis-xray3d-v1")
    if where() == "kaggle" and kaggle_input.exists():
        return kaggle_input
    return data_root() / "processed"


def results_root() -> Path:
    """Small result files that ARE committed (metric tables, run summaries): <repo>/results.

    Override with XRAY3D_RESULTS (used when testing the notebooks, so test runs never touch the repo).
    Nothing derived from DeepFluoro goes here except aggregate numbers.
    """
    if os.environ.get("XRAY3D_RESULTS"):
        return Path(os.environ["XRAY3D_RESULTS"])
    return repo_root() / "results"


def has_gpu() -> bool:
    """True if PyTorch is installed and sees a CUDA GPU (Colab/Kaggle with a GPU runtime)."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def summary() -> None:
    """Print a short report of the environment. Unit 00 runs this first."""
    print(f"running on : {where()}  ({platform.system()} {platform.release()})")
    print(f"python     : {sys.version.split()[0]}")
    for name in ["numpy", "scipy", "nibabel", "matplotlib", "torch", "diffdrr"]:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<11}: {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"{name:<11}: not installed")
    print(f"GPU        : {'yes' if has_gpu() else 'no'}")
    print(f"repo       : {repo_root()}")
    d = data_root()
    print(f"data root  : {d}  ({'exists' if d.exists() else 'missing'})")
