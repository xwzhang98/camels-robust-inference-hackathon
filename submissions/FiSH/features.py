"""Graph construction for the FiSH submission's node feature set ("all_together": line-of-
sight velocity plus 9 extra Subhalo/Group columns, including SubhaloGrNr).

Ported, not imported, from this project's own
``experiments/astrid_illustristng_vs_simba/step2_train_gnn_featuresweep.py`` (the script
that produced every all_together checkpoint this project trained) -- that script isn't part
of the ``kaai_hackathon`` package the organizers ship, so it has to travel with the
submission. The feature RECIPE (which columns, log1p/signed-log1p transforms, GroupPos
mapped through SubhaloGrNr) is copied verbatim; nothing about it has changed for this
submission.

Positions are NEVER a raw per-node feature -- they only decide the graph edges (relative
distance + circular-centroid angles, via ``kaai_hackathon.graph``, unmodified), which is
what makes the model translation/rotation invariant by construction (see this project's own
robustness measurements: translate/permute are exact; rotate90 costs a little, because the
line-of-sight velocity feature is not rotation-invariant -- a known, documented tradeoff,
not a bug).
"""
from __future__ import annotations

import numpy as np

from kaai_hackathon import BOX_SIZE
from kaai_hackathon.graph import R_LINK, edge_features, periodic_radius_graph

#: Fixed for this submission -- the exact 10 extra columns "all_together" trains on,
#: added on top of the always-present line-of-sight velocity. Order matters: it must match
#: training exactly, since it sets which column of `x` each weight learned to read.
EXTRA_FIELDS = ["SubhaloStellarMass", "SubhaloVelDisp", "SubhaloSpin", "SubhaloGrNr",
                "SubhaloMassType0", "GroupPos", "GroupMassType1", "SubhaloMassType1",
                "GroupNsubs", "SubhaloHalfmassRadType1"]


def _log1p_nonneg(v: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(v, 0.0, None))[:, None]


def _signed_log1p(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * np.log1p(np.abs(v))


_SUBHALO_COLS = {
    "SubhaloStellarMass": ["SubhaloMassType"],
    "SubhaloVelDisp": ["SubhaloVelDisp"],
    "SubhaloSpin": ["SubhaloSpin"],
    "SubhaloGrNr": ["SubhaloGrNr"],
    "SubhaloMassType0": ["SubhaloMassType"],
    "SubhaloMassType1": ["SubhaloMassType"],
    "SubhaloHalfmassRadType1": ["SubhaloHalfmassRadType"],
    "GroupPos": ["SubhaloGrNr"],
    "GroupMassType1": ["SubhaloGrNr"],
    "GroupNsubs": ["SubhaloGrNr"],
}
_GROUP_COLS = {
    "GroupPos": ["GroupPos"],
    "GroupMassType1": ["GroupMassType"],
    "GroupNsubs": ["GroupNsubs"],
}


def catalog_fields() -> tuple[list[str], list[str]]:
    """-> (subhalo_fields, group_fields) to pass to ``read_catalog`` -- everything
    EXTRA_FIELDS (plus the base position/velocity/stellar-mass columns) needs."""
    subhalo = {"SubhaloPos", "SubhaloVel", "SubhaloMassType"}
    group: set[str] = set()
    for name in EXTRA_FIELDS:
        subhalo.update(_SUBHALO_COLS[name])
        group.update(_GROUP_COLS.get(name, []))
    return sorted(subhalo), sorted(group)


def _extra_feature_columns(subhalo: dict, group: dict, selected: np.ndarray) -> list[np.ndarray]:
    cols = []
    for name in EXTRA_FIELDS:
        if name == "SubhaloStellarMass":
            v = np.asarray(subhalo["SubhaloMassType"])[selected, 4].astype(np.float64)
            cols.append(_log1p_nonneg(v))
        elif name == "SubhaloVelDisp":
            v = np.asarray(subhalo["SubhaloVelDisp"])[selected].astype(np.float64)
            cols.append(_log1p_nonneg(v))
        elif name == "SubhaloSpin":
            v = np.asarray(subhalo["SubhaloSpin"])[selected].astype(np.float64)
            cols.append(_signed_log1p(v))
        elif name == "SubhaloGrNr":
            v = np.asarray(subhalo["SubhaloGrNr"])[selected].astype(np.float64)
            cols.append(v[:, None])
        elif name == "SubhaloMassType0":
            v = np.asarray(subhalo["SubhaloMassType"])[selected, 0].astype(np.float64)
            cols.append(_log1p_nonneg(v))
        elif name == "SubhaloMassType1":
            v = np.asarray(subhalo["SubhaloMassType"])[selected, 1].astype(np.float64)
            cols.append(_log1p_nonneg(v))
        elif name == "SubhaloHalfmassRadType1":
            v = np.asarray(subhalo["SubhaloHalfmassRadType"])[selected, 1].astype(np.float64)
            cols.append(_log1p_nonneg(v))
        elif name == "GroupPos":
            grnr = np.asarray(subhalo["SubhaloGrNr"]).astype(np.int64)
            gpos = np.asarray(group["GroupPos"])[grnr][selected].astype(np.float64) / BOX_SIZE
            cols.append(np.mod(gpos, 1.0))
        elif name == "GroupMassType1":
            grnr = np.asarray(subhalo["SubhaloGrNr"]).astype(np.int64)
            v = np.asarray(group["GroupMassType"])[grnr][selected, 1].astype(np.float64)
            cols.append(_log1p_nonneg(v))
        elif name == "GroupNsubs":
            grnr = np.asarray(subhalo["SubhaloGrNr"]).astype(np.int64)
            v = np.asarray(group["GroupNsubs"])[grnr][selected].astype(np.float64)
            cols.append(_log1p_nonneg(v))
        else:
            raise ValueError(f"unknown EXTRA_FIELDS entry {name!r}")
    return cols


def to_graph(cat, mstar_min: float) -> dict:
    """One catalog -> one graph, exactly as trained (fixed evaluation mass cut, no
    augmentation -- augmentation is a training-time-only device)."""
    mstar = np.asarray(cat.subhalo["SubhaloMassType"])[:, 4].astype(np.float64)
    selected = mstar > mstar_min
    pos = np.asarray(cat.subhalo["SubhaloPos"])[selected].astype(np.float64) / cat.box_size
    pos = np.mod(pos, 1.0).astype(np.float32)

    vz = np.asarray(cat.subhalo["SubhaloVel"])[selected, 2].astype(np.float64)
    base = [_signed_log1p(vz)[:, None]]
    x = np.concatenate(base + _extra_feature_columns(cat.subhalo, cat.group, selected),
                       axis=1).astype(np.float32)

    src, dst = periodic_radius_graph(pos, r_link=R_LINK)
    edges = edge_features(pos, src, dst, r_link=R_LINK, centroid="circular")
    return {"x": x, "edge_index": np.stack([src, dst]).astype(np.int64), "edge_attr": edges,
            "u": np.array([np.log10(max(len(pos), 1))], dtype=np.float32)}
