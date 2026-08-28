"""The named test conditions, and how a catalog is put through one.

A condition is a shift plus its settings. Seeds come from
``(base_seed, condition, suite, sim_id)`` through a stable hash, which has two
consequences worth knowing:

  * every submission is scored on byte-identical inputs, and
  * shifted catalogs can be regenerated on demand instead of being stored.

    >>> spec = condition("mass_noise")
    >>> shifted = apply_condition(cat, spec, seed=condition_seed(7, spec.name, "SIMBA", 42))

The tiers say what a result means. Tier S is a symmetry of the physics, so a model that
loses accuracy there has an architectural problem rather than a robustness problem. Tier C
really does destroy information, so some loss is expected and the question is how much.
Tier O is a different simulation code, which is the whole point of the exercise.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from kaai_hackathon import shifts
from kaai_hackathon.catalog_io import Catalog

_OPS = {
    "identity": lambda cat, rng: cat.copy(),
    "periodic_translation": shifts.periodic_translation,
    "rotation_90": shifts.rotation_90,
    "row_permutation": shifts.row_permutation,
    "position_noise": shifts.position_noise,
    "velocity_noise": shifts.velocity_noise,
    "mass_noise": shifts.mass_noise,
}


@dataclass(frozen=True)
class ShiftSpec:
    name: str
    op: str
    params: dict = field(default_factory=dict)
    tier: str = "C"


# Noise levels are provisional: they need calibrating against the baseline so that each
# condition costs a measurable but not catastrophic amount of accuracy. For scale, the box
# is 25000 ckpc/h and typical halo separations are of order 1000 ckpc/h.
PUBLISHED_CONDITIONS: tuple[ShiftSpec, ...] = (
    ShiftSpec("clean", "identity", {}, "clean"),
    # Tier S -- a model that respects the physics should lose nothing here.
    ShiftSpec("translate", "periodic_translation", {}, "S"),
    ShiftSpec("rotate90", "rotation_90", {}, "S"),
    ShiftSpec("permute", "row_permutation", {}, "S"),
    # Tier C -- recipes are published; the seeds are not.
    ShiftSpec("pos_noise_lo", "position_noise", {"sigma_ckpch": 100.0}, "C"),
    ShiftSpec("pos_noise_hi", "position_noise", {"sigma_ckpch": 500.0}, "C"),
    ShiftSpec("vel_noise", "velocity_noise", {"sigma_kms": 50.0}, "C"),
    ShiftSpec("mass_noise", "mass_noise", {"sigma_dex": 0.1}, "C"),
    # Tier O is not listed here: it is the same conditions run on a held-out suite.
)


def condition(name: str) -> ShiftSpec:
    """Look up a published condition by name."""
    for spec in PUBLISHED_CONDITIONS:
        if spec.name == name:
            return spec
    known = ", ".join(spec.name for spec in PUBLISHED_CONDITIONS)
    raise KeyError(f"unknown condition {name!r}; known: {known}")


def condition_seed(base_seed: int, condition_name: str, suite: str, sim_id: int) -> int:
    """A reproducible per-catalog seed. Stable across processes, unlike ``hash()``."""
    key = f"{base_seed}|{condition_name}|{suite}|{sim_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def apply_condition(cat: Catalog, spec: ShiftSpec, seed: int) -> Catalog:
    """Run one condition over a catalog and return the shifted copy."""
    if spec.op not in _OPS:
        known = ", ".join(sorted(_OPS))
        raise KeyError(f"unknown shift op {spec.op!r}; known: {known}")
    return _OPS[spec.op](cat, np.random.default_rng(int(seed)), **spec.params)
