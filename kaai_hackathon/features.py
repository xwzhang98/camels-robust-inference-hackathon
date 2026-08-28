"""A small, deliberately transparent summary of a catalog.

A catalog is a *set* of a few hundred to a few thousand galaxies, and its size changes from
simulation to simulation. A regressor needs a fixed-length vector. This module supplies the
simplest reasonable one: how many galaxies there are, how their masses are distributed, and
how clustered they are.

Every feature is invariant under the Tier S symmetries **by construction**:

* counts and mass histograms do not care about row order or about where the box origin is;
* neighbour counts use the periodic minimum-image convention, so translating or rotating
  the box by a right angle leaves them unchanged.

That is the point. Notebook 03 uses this to show what a symmetry-respecting representation
buys you compared to, say, sorting the rows and feeding them to an MLP.

This is a floor to beat, not a good model. It throws away every column except stellar mass
and position, and it never looks at a galaxy's individual properties.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from kaai_hackathon.catalog_io import Catalog

#: log10(M_star / 1e10 Msun/h) bin edges -- 23 bins spanning the resolved range.
MSTAR_BINS = np.linspace(-2.5, 2.0, 24)
#: Neighbour-counting radii in ckpc/h. 1000 ckpc/h = 1 cMpc/h.
PAIR_RADII = (500.0, 1000.0, 2500.0, 5000.0)
#: Default selection: M_star > 1.3e8 Msun/h, about 20 star particles at CAMELS resolution.
DEFAULT_MSTAR_MIN = 1.3e-2

FEATURE_NAMES: tuple[str, ...] = (
    ("log1p_n_selected", "log1p_n_subhalos", "log1p_n_groups",
     "mean_log_mstar", "std_log_mstar")
    + tuple(f"mstar_hist_{i:02d}" for i in range(len(MSTAR_BINS) - 1))
    + tuple(f"log1p_neighbours_r{int(r)}" for r in PAIR_RADII)
)
N_FEATURES = len(FEATURE_NAMES)


def mean_neighbour_counts(positions: np.ndarray, box_size: float,
                          radii=PAIR_RADII) -> np.ndarray:
    """Mean number of other galaxies within each radius, with periodic wrapping.

    ``cKDTree(..., boxsize=...)`` applies the minimum-image convention natively and is
    roughly two orders of magnitude faster than building the full N x N distance matrix,
    which matters once you are looping over thousands of catalogs.
    """
    n = len(positions)
    if n < 2:
        return np.zeros(len(radii), dtype=np.float64)
    wrapped = np.mod(np.asarray(positions, dtype=np.float64), box_size)
    tree = cKDTree(wrapped, boxsize=box_size)
    # count_neighbors counts ordered pairs and includes each point with itself.
    counts = np.asarray(tree.count_neighbors(tree, list(radii)), dtype=np.float64)
    return (counts - n) / n


def catalog_features(cat: Catalog, mstar_min: float = DEFAULT_MSTAR_MIN) -> np.ndarray:
    """Fixed-length summary of one catalog. ``mstar_min`` is in 1e10 Msun/h."""
    n_groups = cat.n_groups
    n_subhalos = cat.n_subhalos
    if n_subhalos and "SubhaloMassType" in cat.subhalo:
        mstar = np.asarray(cat.subhalo["SubhaloMassType"])[:, 4].astype(np.float64)
    else:
        mstar = np.zeros(0, dtype=np.float64)

    selected = mstar > mstar_min
    n_selected = int(selected.sum())
    log_mstar = np.log10(mstar[selected]) if n_selected else np.zeros(0)

    scalars = np.array([
        np.log1p(n_selected),
        np.log1p(n_subhalos),
        np.log1p(n_groups),
        float(log_mstar.mean()) if n_selected else 0.0,
        float(log_mstar.std()) if n_selected > 1 else 0.0,
    ])

    histogram = (np.histogram(log_mstar, bins=MSTAR_BINS)[0].astype(np.float64)
                 if n_selected else np.zeros(len(MSTAR_BINS) - 1))

    if n_selected > 1 and "SubhaloPos" in cat.subhalo:
        positions = np.asarray(cat.subhalo["SubhaloPos"])[selected]
        neighbours = mean_neighbour_counts(positions, cat.box_size)
    else:
        neighbours = np.zeros(len(PAIR_RADII))

    out = np.concatenate([scalars, np.log1p(histogram), np.log1p(neighbours)])
    if out.shape != (N_FEATURES,):
        raise AssertionError(f"expected {N_FEATURES} features, built {out.shape[0]}")
    return out
