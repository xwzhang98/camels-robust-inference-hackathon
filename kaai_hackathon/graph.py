"""Turning a catalog into a graph: periodic radius graph + de Santi's edge features.

Spec: de Santi et al. 2023 (arXiv:2302.14101), Eqs. 1 and 3-7, cross-checked against the
reference implementation it follows, PabloVD/CosmoGraphNet. Where the two disagree we
follow the **paper**:

  * **No self-loops.** The paper says so explicitly; the reference code adds them.
  * **Selection by stellar mass**, not by star-particle count.

One place we deliberately depart from *both*: see ``edge_features`` and ``CENTROID_MODES``.
The paper's alpha/beta features are measured from the catalog centroid, computed as a plain
mean of the wrapped coordinates -- and a plain mean is not periodic. Translate the box and
the reference point does not translate with it, so alpha and beta change by up to ~2 (they
are cosines). That breaks ``periodic_translation``, which is a Tier S condition. We default
to a circular centroid, which fixes it exactly; ``centroid="mean"`` reproduces the paper.

Everything here works in NORMALIZED coordinates -- positions in ``[0, 1)``, box side 1 --
so the paper's linking length of 1.25 cMpc/h becomes ``1.25 / 25 = 0.05``. Passing raw
ckpc/h coordinates is the easiest mistake to make here, so it raises instead of silently
building a graph with no edges.

The global feature is ``log10(N_g)``: the number of selected galaxies is a deliberate,
explicit input to the published baseline (paper footnote 3), not an oversight. It is worth
knowing, because it means the baseline can score on abundance alone.
"""
from __future__ import annotations

import numpy as np

from kaai_hackathon.catalog_io import Catalog

R_LINK_MPC_H = 1.25           #: de Santi's Optuna-found linking length, "around 1.25 h^-1 Mpc"
R_LINK = R_LINK_MPC_H / 25.0  #: = 0.05 in normalized box units
#: Selection used at evaluation time: M_star > 1.95e8 Msun/h, in units of 1e10 Msun/h.
#: This is de Santi's fixed test threshold -- the midpoint of the range they randomize over
#: during training (see :func:`sample_mstar_threshold`).
MSTAR_TEST_THRESHOLD = 1.95e-2
#: Bottom of the randomized training range: the cut is ``MSTAR_TRAIN_BASE * U(1, 2)``.
MSTAR_TRAIN_BASE = 1.3e-2
DEFAULT_MSTAR_MIN = MSTAR_TEST_THRESHOLD
#: Reference point for the alpha/beta edge features. ``"circular"`` is periodic and is the
#: default; ``"mean"`` is what de Santi et al. and CosmoGraphNet do.
CENTROID_MODES = ("circular", "mean")
#: How velocity enters the node feature. ``"vz"`` is the published choice and is not
#: rotation invariant; ``"speed"`` is. See :func:`node_features`.
VELOCITY_MODES = ("vz", "speed", "none")


def catalog_centroid(pos: np.ndarray, mode: str = "circular") -> np.ndarray:
    """A reference point for the direction features, as a ``(3,)`` array in ``[0, 1)``.

    ``"mean"`` is the arithmetic mean of the wrapped coordinates -- the reference
    implementation's choice, and **not** a periodic quantity: shift every galaxy by the same
    vector and this moves somewhere else entirely.

    ``"circular"`` treats each axis as an angle and takes ``atan2(<sin>, <cos>)``, the
    standard circular mean. Shift the box and it shifts with it, exactly, so the direction
    features built on it are translation-invariant. It is equivariant under the cubic
    rotations too, so nothing is lost.

    Measured on 400 uniform points at ``r_link = 0.15``: under a periodic translation the
    edge features move by at most 2.0 with ``"mean"`` and 2e-6 with ``"circular"``.
    """
    p = np.mod(np.asarray(pos, dtype=np.float64), 1.0)
    if mode == "mean":
        return p.mean(axis=0)
    if mode == "circular":
        angle = 2.0 * np.pi * p
        return np.mod(np.arctan2(np.sin(angle).mean(axis=0),
                                 np.cos(angle).mean(axis=0)) / (2.0 * np.pi), 1.0)
    raise ValueError(f"unknown centroid mode {mode!r}; known: {CENTROID_MODES}")


def minimum_image(delta: np.ndarray) -> np.ndarray:
    """Periodic minimum-image displacement for coordinates normalized to ``[0, 1)``.

    ``d - round(d)`` maps each component into ``[-0.5, 0.5)``, the correct convention for
    a cubic torus of side 1. Works on any trailing-axis-3 array.
    """
    d = np.asarray(delta, dtype=np.float64)
    return (d - np.rint(d)).astype(np.float32)


def sample_mstar_threshold(rng: np.random.Generator) -> float:
    """Draw one stellar-mass cut, ``1.3e8 * U(1, 2)`` Msun/h, in units of 1e10 Msun/h.

    de Santi et al. redraw this **per catalog, per epoch**, and it is not cosmetic. The
    lowest galaxy mass a simulation can resolve is set by its particle mass, which in CAMELS
    is a direct function of Omega_m. A model trained at one fixed cut can read the parameter
    off where the mass function stops instead of off the physics. Randomizing the cut takes
    that shortcut away, and it is also what makes the model tolerate a catalog selected
    differently from the ones it trained on.
    """
    return float(MSTAR_TRAIN_BASE * rng.uniform(1.0, 2.0))


def node_features(vel: np.ndarray, mstar: np.ndarray | None = None,
                  velocity: str = "vz") -> np.ndarray:
    """What each galaxy carries into the network.

    de Santi Eq. 1 uses the line-of-sight velocity alone, ``sign(v_z) log10(1 + |v_z|)``;
    the "+M*" variant appends ``log10(1 + M_star)``.

    There is **no position** here, in any form. Positions enter through the graph -- which
    pairs are connected, and the distance and two angles on each edge -- never as a
    coordinate. That is what makes the model translation- and rotation-invariant... as far
    as the edges go.

    The node feature is where that breaks, and it is worth being explicit about it, because
    rotating the box is one of the conditions this event scores:

    ``velocity="vz"``
        The published choice. It is one projected component, which is the right stand-in for
        an observable line-of-sight velocity -- and it is emphatically **not** rotation
        invariant. Turn the box by a right angle that moves the z axis and every node's
        feature becomes a different component of the same velocity vector. The physics is
        isotropic so the *distribution* is unchanged, but a given catalog gets different
        numbers, and the model's answer moves with them.
    ``velocity="speed"``
        ``log10(1 + |v|)``. Rotation invariant by construction, at the cost of throwing away
        the line-of-sight structure that carries redshift-space information.
    ``velocity="none"``
        No velocity at all. Combine with ``mstar`` for a mass-only node.

    Units: ``vel`` in km/s, ``mstar`` in 1e10 Msun/h, matching the catalog's own columns.
    """
    v = np.asarray(vel, dtype=np.float64)
    if v.ndim != 2 or (v.shape[1] != 3 and v.size):
        raise ValueError(f"velocities must be (N, 3); got {v.shape}")
    columns = []
    if velocity == "vz":
        vz = v[:, 2] if v.size else np.zeros(0)
        columns.append(np.sign(vz) * np.log10(1.0 + np.abs(vz)))
    elif velocity == "speed":
        columns.append(np.log10(1.0 + np.linalg.norm(v, axis=-1)) if v.size
                       else np.zeros(0))
    elif velocity != "none":
        raise ValueError(f"unknown velocity mode {velocity!r}; known: {VELOCITY_MODES}")
    if mstar is not None:
        columns.append(np.log10(1.0 + np.asarray(mstar, dtype=np.float64)))
    if not columns:
        raise ValueError("node_features needs a velocity mode or a mass column")
    return np.stack(columns, axis=-1).astype(np.float32)


def _validate_positions(pos: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float32)
    if p.ndim != 2 or (p.shape[1] != 3 and p.size):
        raise ValueError(f"positions must be (N, 3); got {p.shape}")
    if p.size and (p.min() < -1e-4 or p.max() > 1.0 + 1e-4):
        raise ValueError(
            f"positions must be NORMALIZED to [0, 1) (box side 1), got range "
            f"[{p.min():.4g}, {p.max():.4g}] -- did you pass raw ckpc/h coordinates?"
        )
    return p


def periodic_radius_graph(pos: np.ndarray, r_link: float = R_LINK,
                          use_kdtree: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Directed edge index ``(src, dst)`` for every pair within ``r_link``, periodic.

    Both directions per pair, never a self-loop. ``use_kdtree`` follows the reference
    recipe (scipy ``KDTree`` with ``boxsize``, which handles periodicity natively); the
    O(N^2) fallback exists so the periodic logic can be checked against the definition,
    and the two are cross-checked by a test.
    """
    p = _validate_positions(pos)
    n = p.shape[0]
    empty = np.zeros(0, dtype=np.int64)
    if n < 2:
        return empty, empty
    if use_kdtree:
        from scipy.spatial import KDTree
        # boxsize slightly > 1 mirrors the reference: it guards points sitting exactly
        # on the boundary, which KDTree would otherwise reject.
        pairs = KDTree(np.mod(p, 1.0), leafsize=16, boxsize=1.0001).query_pairs(
            r=float(r_link), output_type="ndarray")
        if pairs.size == 0:
            return empty, empty
        src = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)
        return src, dst
    delta = minimum_image(p[:, None, :] - p[None, :, :])
    dist = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dist, np.inf)
    src, dst = np.nonzero(dist <= float(r_link))
    return src.astype(np.int64), dst.astype(np.int64)


def edge_features(pos: np.ndarray, src: np.ndarray, dst: np.ndarray,
                  r_link: float = R_LINK, centroid: str = "circular") -> np.ndarray:
    """de Santi Eqs. 3-7: ``e_ij = [ |d_ij| / r_link , alpha_ij , beta_ij ]``.

    ::

        d_ij     = r_i - r_j                        (minimum-image)
        delta_i  = r_i - c,   c = catalog centroid  (minimum-image)
        alpha_ij = unit(delta_i) . unit(delta_j)    cosine subtended at the centroid
        beta_ij  = unit(delta_i) . unit(d_ij)

    ``|d_ij|`` is invariant under every Tier S symmetry on its own. ``alpha`` and ``beta``
    are invariant under rotation and row permutation for any choice of ``c``, but under a
    periodic *translation* only if ``c`` translates with the box -- which is what
    ``centroid="circular"`` buys and ``centroid="mean"`` (the paper's choice) does not.
    See :func:`catalog_centroid`.
    """
    p = _validate_positions(pos)
    if len(src) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    d = minimum_image(p[src] - p[dst])
    dist = np.linalg.norm(d, axis=-1, keepdims=True)
    c = catalog_centroid(p, mode=centroid).reshape(1, 3)
    delta = minimum_image(p - c)
    unit_delta = delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-12)
    unit_d = d / np.maximum(dist, 1e-12)
    alpha = np.sum(unit_delta[src] * unit_delta[dst], axis=-1, keepdims=True)
    beta = np.sum(unit_delta[src] * unit_d, axis=-1, keepdims=True)
    return np.concatenate([dist / float(r_link), alpha, beta], axis=-1).astype(np.float32)


def catalog_to_graph(cat: Catalog, r_link: float = R_LINK,
                     mstar_min: float = DEFAULT_MSTAR_MIN,
                     centroid: str = "circular",
                     use_mstar: bool = False,
                     positions_only: bool = False,
                     velocity: str = "vz") -> dict:
    """One catalog -> ``{"x", "edge_index", "edge_attr", "u"}``, all numpy arrays.

    Selects galaxies above ``mstar_min`` (in 1e10 Msun/h), normalizes their positions to
    ``[0, 1)``, builds the periodic radius graph, and computes the edge features.

    The node tensor follows :func:`node_features`: line-of-sight velocity by default,
    ``+ log10(1 + M_star)`` with ``use_mstar=True``, and a zeros placeholder with
    ``positions_only=True`` (de Santi's positions-only variant, where every bit of
    information arrives through the edges and the galaxy count).

    The global feature is ``log10(N_galaxies)``. The count is an explicit input to the
    published model, not an accident -- worth remembering when reading the scores, because
    some of what the model achieves is available from abundance alone.
    """
    mstar = np.asarray(cat.subhalo["SubhaloMassType"])[:, 4].astype(np.float64)
    selected = mstar > mstar_min
    pos = np.asarray(cat.subhalo["SubhaloPos"])[selected].astype(np.float64) / cat.box_size
    pos = np.mod(pos, 1.0).astype(np.float32)

    if positions_only:
        x = np.zeros((len(pos), 1), dtype=np.float32)
    else:
        vel = np.asarray(cat.subhalo["SubhaloVel"])[selected]
        x = node_features(vel, mstar[selected] if use_mstar else None,
                          velocity=velocity)

    src, dst = periodic_radius_graph(pos, r_link=r_link)
    edges = edge_features(pos, src, dst, r_link=r_link, centroid=centroid)
    return {"x": x,
            "edge_index": np.stack([src, dst]).astype(np.int64),
            "edge_attr": edges,
            "u": np.array([np.log10(max(len(pos), 1))], dtype=np.float32)}
