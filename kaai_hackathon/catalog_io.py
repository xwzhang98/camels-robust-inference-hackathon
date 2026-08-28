"""Minimal reader/writer for CAMELS Subfind group catalogs.

A catalog file has two tables:

    Group/     one row per FoF halo        (the "host")
    Subhalo/   one row per Subfind subhalo (a "galaxy" if it has stellar mass)

They are linked by ``Subhalo/SubhaloGrNr`` -> the row index in ``Group/``.
The notebooks walk through what those rows mean; this module only moves data.

Reading is selective on purpose: an Astrid catalog is ~141 MB across 50 subhalo
fields, so pass ``subhalo_fields=[...]`` and load only the columns you need.

    >>> read_header(path)
    {'box_size': 25000.0, 'redshift': 0.0, 'n_groups': 18180, 'n_subhalos': 15712}
    >>> list_fields(path)["Subhalo"][:3]
    ['SubhaloBHMass', 'SubhaloBHMdot', 'SubhaloBfldDisk']
    >>> cat = read_catalog(path, subhalo_fields=["SubhaloPos", "SubhaloMassType"])
    >>> cat.subhalo["SubhaloPos"].shape
    (15712, 3)

`write_catalog` emits only Group, Subhalo and a minimal Header. It has no code path
that writes `Parameters` or `Config`, which is what keeps the simulation parameters
out of any catalog handed to a model:

    Header/Omega0, Parameters/Omega0              -> Omega_m, exactly
    Header/OmegaLambda                            -> Omega_m, as 1 - Omega_m
    Parameters/RadioFeedbackFactor                -> A_AGN1  (x1)
    Parameters/RadioFeedbackReiorientationFactor  -> A_AGN2  (x20)
    Parameters/WindEnergyIn1e51erg                -> A_SN1   (x3.6)
    Parameters/VariableWindVelFactor              -> A_SN2   (x7.4)

`Header/Git_commit` is dropped too: it differs between simulation codes, so it names
the suite.

`IDs` is dropped as well. It is the list of member particle IDs, which is useless
without the particle snapshots -- and those are not part of this dataset. It is not
free to carry: it is empty in IllustrisTNG but 68% of a SIMBA file, 88% of an Astrid
file and 75% of a Swift-EAGLE one, so dropping it takes the 2700 public catalogs from
204 GB to 44 GB. Its length is also sum(GroupLen), one more way to count particles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from kaai_hackathon import BOX_SIZE, HUBBLE_PARAM

DROPPED_TOP_LEVEL_GROUPS = ("Config", "Parameters", "IDs")


@dataclass
class Catalog:
    """Group and Subhalo tables held as ``{field_name: ndarray}``."""

    group: dict[str, np.ndarray] = field(default_factory=dict)
    subhalo: dict[str, np.ndarray] = field(default_factory=dict)
    box_size: float = BOX_SIZE
    redshift: float = 0.0
    n_groups: int = 0
    n_subhalos: int = 0

    def copy(self) -> "Catalog":
        return Catalog(
            group={k: v.copy() for k, v in self.group.items()},
            subhalo={k: v.copy() for k, v in self.subhalo.items()},
            box_size=self.box_size,
            redshift=self.redshift,
            n_groups=self.n_groups,
            n_subhalos=self.n_subhalos,
        )


def read_header(path: str | Path) -> dict:
    """Box size, redshift and row counts, without loading any table."""
    with h5py.File(str(path), "r") as f:
        attrs = f["Header"].attrs
        return {
            "box_size": float(np.asarray(attrs["BoxSize"]).ravel()[0]),
            "redshift": float(np.asarray(attrs["Redshift"]).ravel()[0]),
            "n_groups": int(np.asarray(attrs["Ngroups_Total"]).ravel()[0]),
            "n_subhalos": int(np.asarray(attrs["Nsubgroups_Total"]).ravel()[0]),
        }


def list_fields(path: str | Path) -> dict[str, list[str]]:
    """``{"Group": [...], "Subhalo": [...]}`` -- what this file actually contains.

    Worth calling: Swift-EAGLE has 47 subhalo fields where the other suites have 50.
    """
    with h5py.File(str(path), "r") as f:
        return {name: sorted(f[name].keys()) for name in ("Group", "Subhalo") if name in f}


def read_catalog(path: str | Path, group_fields=None, subhalo_fields=None) -> Catalog:
    """Load a catalog. ``None`` means every field; a list loads only those columns."""
    path = Path(path)
    with h5py.File(str(path), "r") as f:
        head = f["Header"].attrs

        def _load(table: str, wanted) -> dict[str, np.ndarray]:
            if table not in f:
                return {}
            names = sorted(f[table].keys()) if wanted is None else list(wanted)
            out = {}
            for name in names:
                if name not in f[table]:
                    raise KeyError(f"{path.name} has no {table}/{name}")
                out[name] = f[table][name][...]
            return out

        group = _load("Group", group_fields)
        subhalo = _load("Subhalo", subhalo_fields)
        return Catalog(
            group=group,
            subhalo=subhalo,
            box_size=float(np.asarray(head["BoxSize"]).ravel()[0]),
            redshift=float(np.asarray(head["Redshift"]).ravel()[0]),
            n_groups=int(np.asarray(head["Ngroups_Total"]).ravel()[0]),
            n_subhalos=int(np.asarray(head["Nsubgroups_Total"]).ravel()[0]),
        )


def write_catalog(catalog: Catalog, path: str | Path) -> None:
    """Write Group, Subhalo and a minimal Header. Nothing else is ever written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        for table, data in (("Group", catalog.group), ("Subhalo", catalog.subhalo)):
            handle = f.create_group(table)
            for name, values in data.items():
                handle.create_dataset(name, data=np.asarray(values))
        head = f.create_group("Header")
        head.attrs["BoxSize"] = float(catalog.box_size)
        head.attrs["Redshift"] = float(catalog.redshift)
        head.attrs["HubbleParam"] = float(HUBBLE_PARAM)
        head.attrs["NumFiles"] = np.int32(1)
        head.attrs["Ngroups_Total"] = np.int32(catalog.n_groups)
        head.attrs["Nsubgroups_Total"] = np.int32(catalog.n_subhalos)


def validate_linkage(catalog: Catalog) -> None:
    """Raise ``ValueError`` unless Group <-> Subhalo bookkeeping is self-consistent.

    The invariants, all confirmed against real CAMELS files:

      * ``SubhaloGrNr`` is sorted non-decreasing
      * ``bincount(SubhaloGrNr) == GroupNsubs``
      * ``GroupFirstSub`` is the global row of a group's first subhalo, or -1 if empty
      * ``SubhaloParent`` is GROUP-LOCAL: ``SubhaloParent[i] < GroupNsubs[GrNr[i]]``

    Dropping subhalo rows invalidates all four, which is why the dropout shift has to
    rebuild them.
    """
    nsubs = np.asarray(catalog.group["GroupNsubs"]).astype(np.int64)
    first = np.asarray(catalog.group["GroupFirstSub"]).astype(np.int64)
    grnr = np.asarray(catalog.subhalo["SubhaloGrNr"]).astype(np.int64)
    n_groups = len(nsubs)

    if grnr.size:
        if not np.all(np.diff(grnr) >= 0):
            raise ValueError("SubhaloGrNr must be sorted non-decreasing")
        if grnr.min() < 0 or grnr.max() >= n_groups:
            raise ValueError("SubhaloGrNr out of range")

    counts = np.bincount(grnr, minlength=n_groups) if grnr.size else np.zeros(n_groups, np.int64)
    if not np.array_equal(counts, nsubs):
        raise ValueError("GroupNsubs does not match bincount(SubhaloGrNr)")

    expected_first = np.full(n_groups, -1, dtype=np.int64)
    if grnr.size:
        starts = np.zeros(n_groups, dtype=np.int64)
        starts[1:] = np.cumsum(counts)[:-1]
        expected_first[counts > 0] = starts[counts > 0]
    if not np.array_equal(first, expected_first):
        raise ValueError("GroupFirstSub inconsistent with GroupNsubs / SubhaloGrNr")

    if "SubhaloParent" in catalog.subhalo and grnr.size:
        parent = np.asarray(catalog.subhalo["SubhaloParent"]).astype(np.int64)
        seen = parent >= 0
        if np.any(parent[seen] >= nsubs[grnr][seen]):
            raise ValueError("SubhaloParent must be group-local and < GroupNsubs of its group")
