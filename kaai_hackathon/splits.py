"""Which simulations are public, and what their parameters are.

Each suite has 1000 latin-hypercube simulations, ``LH_0`` ... ``LH_999``. They are split
**by simulation id** into 900 public and 100 held-out. Splitting by id rather than by
anything derived from the catalogs is what keeps a simulation whole: every galaxy from a
given simulation lands on the same side of the split.

The six parameters of ``LH_n`` are row ``n`` of ``params_LH_<suite>.txt``, in the order
``(Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2)``. The mapping is positional, so never
zip a sorted list of directory names against a separately sorted label array.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SPLIT_SEED = 20260825
N_SIMS = 1000
N_PUBLIC = 900


def _suite_seed(suite: str, seed: int) -> int:
    """Per-suite seed, so the three suites do not hold out the same ids."""
    return int.from_bytes(hashlib.sha256(f"{seed}|{suite}".encode()).digest()[:4], "big")


def make_split(suite: str, n_sims: int = N_SIMS, n_public: int = N_PUBLIC,
               seed: int = SPLIT_SEED) -> dict[str, list[int]]:
    """``{"public": [...], "private_test": [...]}`` -- disjoint, sorted, covering 0..n_sims-1."""
    if not 0 < n_public < n_sims:
        raise ValueError(f"n_public must be in (0, {n_sims}), got {n_public}")
    shuffled = np.random.default_rng(_suite_seed(suite, seed)).permutation(n_sims)
    return {
        "public": sorted(int(i) for i in shuffled[:n_public]),
        "private_test": sorted(int(i) for i in shuffled[n_public:]),
    }


def local_split(suite: str, n_test: int = 100, n_sims: int = N_SIMS,
                n_public: int = N_PUBLIC, seed: int = SPLIT_SEED) -> dict[str, list[int]]:
    """Split the simulations **you actually have** into train and a held-out set.

    ``make_split`` says which simulations are public. The other 100 per suite are the
    organizers' test set and are not shipped, so anything that reads them works on one machine
    and dies with ``FileNotFoundError`` on everyone else's. Use this instead: it only ever
    returns ids from the public set.

    The public ids are **not** ``0..899``. They are a pinned random 900 out of 1000, so
    ``range(300)`` will hit a simulation you do not have roughly one time in ten.

        >>> ids = local_split("IllustrisTNG")
        >>> len(ids["train"]), len(ids["test"])
        (800, 100)

    Splitting off the tail rather than sampling keeps it reproducible and keeps whole
    simulations on one side, which is the part that matters -- galaxies from the same
    simulation are not independent samples.
    """
    public = make_split(suite, n_sims=n_sims, n_public=n_public, seed=seed)["public"]
    if not 0 < n_test < len(public):
        raise ValueError(f"n_test must be in (0, {len(public)}), got {n_test}")
    return {"train": public[:-n_test], "test": public[-n_test:]}


def example_sims(suite: str, count: int = 1, **kwargs) -> list[int]:
    """A few simulation ids that are certain to exist, for illustrations."""
    return local_split(suite, **kwargs)["train"][:count]


def load_labels(params_root: str | Path, suite: str) -> np.ndarray:
    """The ``(1000, 6)`` parameter table for a suite, row ``n`` being ``LH_n``."""
    path = Path(params_root) / f"params_LH_{suite}.txt"
    table = np.loadtxt(path)
    if table.ndim != 2 or table.shape[1] < 6:
        raise ValueError(f"expected a (N, >=6) parameter table, got {table.shape}: {path}")
    return np.asarray(table[:, :6], dtype=np.float64)
