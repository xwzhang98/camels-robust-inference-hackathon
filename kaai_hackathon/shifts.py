"""Distribution shifts applied to a catalog.

Tier S -- symmetries. A model that respects the physics should be *unaffected* by these,
so a drop in accuracy here points at the architecture, not at a robustness limit:

    periodic_translation   the box has no origin
    rotation_90            the box has no preferred orientation
    row_permutation        a catalog is a set; row order carries no information

Tier C -- corruptions. These really do destroy information, so accuracy is expected to
drop; what is measured is how gracefully:

    position_noise         positions are measured imperfectly
    velocity_noise         velocities are measured imperfectly
    mass_noise             the mass calibration is off

Every shift takes ``(catalog, rng)`` and returns a NEW catalog. The input is never
mutated, and every output still satisfies ``catalog_io.validate_linkage``.
"""
from __future__ import annotations

import itertools

import numpy as np

from kaai_hackathon.catalog_io import Catalog

# Positions live in the box and must be wrapped; velocities and spin must not.
POSITION_FIELDS = {
    "group": ("GroupPos", "GroupCM"),
    "subhalo": ("SubhaloPos", "SubhaloCM"),
}
# Every vector quantity in the file. All seven rotate with the SAME matrix -- rotating
# positions while leaving velocities alone would produce an incoherent catalog.
ROTATING_FIELDS = {
    "group": ("GroupPos", "GroupCM", "GroupVel"),
    "subhalo": ("SubhaloPos", "SubhaloCM", "SubhaloVel", "SubhaloSpin"),
}


def _wrap(x: np.ndarray, box: float) -> np.ndarray:
    """Fold into [0, box) in float32, so the stored values are in range as stored.

    Wrapping in float64 and then casting can round a value just under the box up to
    exactly the box; doing the modulo in the output dtype avoids that.
    """
    return np.mod(np.asarray(x, dtype=np.float32), np.float32(box))


def cubic_rotations() -> np.ndarray:
    """The 24 proper rotations of the cube, as signed permutation matrices.

    Of the 48 signed permutation matrices exactly 24 have det = +1. Reflections are
    excluded deliberately: ``SubhaloSpin`` is a pseudo-vector and would need an extra
    sign flip under them.

    Only the cubic group is used because an arbitrary 3D rotation does not preserve a
    periodic cube -- the rotated box stops tiling space.
    """
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3), dtype=np.float64)
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if round(float(np.linalg.det(m))) == 1:
                mats.append(m)
    return np.asarray(mats, dtype=np.float64)


def periodic_translation(cat: Catalog, rng: np.random.Generator) -> Catalog:
    """Shift every position by a random offset and wrap. Velocities are untouched."""
    out = cat.copy()
    offset = rng.uniform(0.0, cat.box_size, size=3)
    for kind, names in POSITION_FIELDS.items():
        store = out.group if kind == "group" else out.subhalo
        for name in names:
            if name in store:
                store[name] = _wrap(store[name].astype(np.float64) + offset, cat.box_size)
    return out


def rotation_90(cat: Catalog, rng: np.random.Generator) -> Catalog:
    """Rotate the whole box by one of the 24 proper cubic rotations, about its centre."""
    out = cat.copy()
    rotations = cubic_rotations()
    matrix = rotations[int(rng.integers(len(rotations)))]
    centre = cat.box_size / 2.0
    for kind, names in ROTATING_FIELDS.items():
        store = out.group if kind == "group" else out.subhalo
        positional = POSITION_FIELDS[kind]
        for name in names:
            if name not in store:
                continue
            values = store[name].astype(np.float64)
            if name in positional:
                store[name] = _wrap((values - centre) @ matrix.T + centre, cat.box_size)
            else:
                store[name] = ((values @ matrix.T)).astype(np.float32)
    return out


def row_permutation(cat: Catalog, rng: np.random.Generator) -> Catalog:
    """Shuffle subhalo rows within each group.

    Real Subfind files keep ``SubhaloGrNr`` sorted, and ``GroupFirstSub`` /
    ``SubhaloParent`` are defined against that ordering. Shuffling *within* each group
    destroys the row order a model might exploit while leaving a valid Subfind catalog.
    """
    out = cat.copy()
    n = cat.n_subhalos
    if n == 0:
        return out

    grnr = np.asarray(cat.subhalo["SubhaloGrNr"]).astype(np.int64)
    first = np.asarray(cat.group["GroupFirstSub"]).astype(np.int64)

    order = np.arange(n, dtype=np.int64)          # order[new_row] = old_row
    for g in np.unique(grnr):
        rows = np.flatnonzero(grnr == g)
        order[rows] = rng.permutation(rows)
    inverse = np.empty(n, dtype=np.int64)         # inverse[old_row] = new_row
    inverse[order] = np.arange(n, dtype=np.int64)

    for name, values in list(out.subhalo.items()):
        if name == "SubhaloGrNr":
            continue                              # invariant under a within-group shuffle
        out.subhalo[name] = values[order]

    if "SubhaloParent" in out.subhalo:
        parent_local = np.asarray(cat.subhalo["SubhaloParent"]).astype(np.int64)[order]
        base = first[grnr]                        # aligned to new rows: grnr is unchanged
        parent_new = np.where(
            parent_local >= 0,
            inverse[np.where(parent_local >= 0, base + parent_local, 0)] - base,
            -1,
        )
        out.subhalo["SubhaloParent"] = parent_new.astype(
            cat.subhalo["SubhaloParent"].dtype)
    return out


# --- Tier C: corruptions -------------------------------------------------------------
#
# Unlike Tier S these genuinely destroy information, so some loss of accuracy is expected.
# What is being measured is how gracefully a model degrades.


def _is_mass_field(name: str) -> bool:
    """True for the 19 mass-carrying fields of a Subfind catalog.

    ``"Mass" in name`` is case-sensitive on purpose: it catches SubhaloMass*,
    GroupMass*, *BHMass and *WindMass while correctly skipping SubhaloHalfmassRad
    (lower-case ``m``), which is a radius. The spherical-overdensity masses
    Group_M_Crit200 / Crit500 / Mean200 / TopHat200 do not contain "Mass" at all and
    need the second clause. ``*BHMdot`` is an accretion rate and is left alone.
    """
    return "Mass" in name or name.startswith("Group_M_")


def position_noise(cat: Catalog, rng: np.random.Generator, sigma_ckpch: float) -> Catalog:
    """Isotropic Gaussian displacement of every position, wrapped back into the box."""
    if sigma_ckpch < 0:
        raise ValueError("sigma_ckpch must be >= 0")
    out = cat.copy()
    for kind, names in POSITION_FIELDS.items():
        store = out.group if kind == "group" else out.subhalo
        for name in names:
            if name not in store:
                continue
            values = store[name].astype(np.float64)
            store[name] = _wrap(values + rng.normal(0.0, sigma_ckpch, values.shape),
                                cat.box_size)
    return out


def velocity_noise(cat: Catalog, rng: np.random.Generator, sigma_kms: float) -> Catalog:
    """Isotropic Gaussian noise on GroupVel and SubhaloVel. Positions are untouched."""
    if sigma_kms < 0:
        raise ValueError("sigma_kms must be >= 0")
    out = cat.copy()
    for store, name in ((out.group, "GroupVel"), (out.subhalo, "SubhaloVel")):
        if name in store:
            values = store[name].astype(np.float64)
            store[name] = (values + rng.normal(0.0, sigma_kms, values.shape)).astype(np.float32)
    return out


def mass_noise(cat: Catalog, rng: np.random.Generator, sigma_dex: float) -> Catalog:
    """One log-normal factor per object, applied to all of that object's mass fields.

    Because every mass field of a given subhalo is scaled by the same factor, ratios such
    as M_star / M_total and the additivity M_total = sum(M_type) both survive. This probes
    sensitivity to overall mass calibration rather than to inconsistent per-component
    measurement error.

    Groups and subhalos are separate objects and get separate factors, so a subhalo and
    its host halo are not scaled together.
    """
    if sigma_dex < 0:
        raise ValueError("sigma_dex must be >= 0")
    out = cat.copy()
    for store, n_rows in ((out.subhalo, cat.n_subhalos), (out.group, cat.n_groups)):
        if n_rows == 0:
            continue
        factor = 10.0 ** rng.normal(0.0, sigma_dex, size=n_rows)
        for name, values in store.items():
            if not _is_mass_field(name) or not np.issubdtype(values.dtype, np.floating):
                continue
            shaped = factor.reshape((-1,) + (1,) * (values.ndim - 1))
            store[name] = (values.astype(np.float64) * shaped).astype(values.dtype)
    return out
