#!/usr/bin/env python
"""Run one submission over every condition and write per-condition, per-target R^2.

Organizer-side. Shifted catalogs are materialized here with pinned seeds, scored, then
deleted -- which is why they are never stored, and why every team is scored on
byte-identical inputs.

The submission sees only a path to a catalog file. Those files come out of
``write_catalog``, which emits ``Group``, ``Subhalo`` and six neutral ``Header``
attributes and has no code path that writes ``Parameters`` or ``Config``. The ``clean``
condition goes through the same writer as an identity shift, so every test input is
scrubbed by construction and there is no separate sanitization step to forget.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from kaai_hackathon.catalog_io import read_catalog, write_catalog
from kaai_hackathon.conditions import (
    PUBLISHED_CONDITIONS, apply_condition, condition_seed,
)
from kaai_hackathon.scoring import macro_average, score_condition
from kaai_hackathon.splits import load_labels
from kaai_hackathon.submission import load_submission, validate_prediction

TARGETS = ("Omega_m", "sigma_8")

GROUP_FIELDS = ["GroupPos", "GroupCM", "GroupVel", "GroupMass",
                "GroupNsubs", "GroupFirstSub"]
SUBHALO_FIELDS = ["SubhaloPos", "SubhaloCM", "SubhaloVel", "SubhaloSpin",
                  "SubhaloMass", "SubhaloMassType", "SubhaloGrNr", "SubhaloParent"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="directory holding predict.py")
    parser.add_argument("--team", default=None, help="name for the report; default = dir name")
    parser.add_argument("--test-root", required=True, help="PRIVATE clean test catalogs")
    parser.add_argument("--params-root", required=True)
    parser.add_argument("--private-split", required=True,
                        help="JSON with {'private_test': {suite: [sim_id, ...]}}")
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--time-limit-s", type=float, default=120.0,
                        help="wall-clock cap per catalog")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.private_split).read_text())["private_test"]
    specs = [s for s in PUBLISHED_CONDITIONS
             if args.conditions is None or s.name in args.conditions]
    team = args.team or Path(args.submission).resolve().name
    model, predict = load_submission(args.submission)
    started = time.time()

    results: dict = {}
    slowest = 0.0
    for spec in specs:
        per_suite: dict[str, dict[str, float]] = {}
        for suite in args.suites:
            labels = load_labels(args.params_root, suite)
            preds = {t: [] for t in TARGETS}
            truths = {t: [] for t in TARGETS}
            for sim_id in split.get(suite, []):
                source = Path(args.test_root) / suite / f"LH_{sim_id}" / "groups_090.hdf5"
                if not source.is_file():
                    continue
                cat = read_catalog(source, group_fields=GROUP_FIELDS,
                                   subhalo_fields=SUBHALO_FIELDS)
                seed = condition_seed(args.base_seed, spec.name, suite, sim_id)
                shifted = apply_condition(cat, spec, seed)
                with tempfile.TemporaryDirectory() as scratch:
                    path = Path(scratch) / "groups_090.hdf5"
                    write_catalog(shifted, path)
                    call_started = time.time()
                    out = validate_prediction(predict(model, str(path)))
                    elapsed = time.time() - call_started
                slowest = max(slowest, elapsed)
                if elapsed > args.time_limit_s:
                    raise TimeoutError(
                        f"{team} took {elapsed:.1f}s on {suite}/LH_{sim_id} under "
                        f"{spec.name} (limit {args.time_limit_s:.0f}s)")
                for j, target in enumerate(TARGETS):
                    preds[target].append(out[target])
                    truths[target].append(labels[sim_id, j])
            per_suite[suite] = score_condition(
                {t: np.asarray(v) for t, v in preds.items()},
                {t: np.asarray(v) for t, v in truths.items()}, TARGETS)
        results[spec.name] = {"per_suite": per_suite,
                              "macro": macro_average(per_suite, TARGETS),
                              "tier": spec.tier}
        macro = results[spec.name]["macro"]
        print(f"{spec.name:14s} {spec.tier:5s} " +
              "  ".join(f"{t}={macro[t]:+.3f}" for t in TARGETS) +
              f"   [{time.time()-started:6.1f}s]", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"team": team, "submission": str(Path(args.submission).resolve()),
         "base_seed": args.base_seed, "suites": args.suites,
         "slowest_catalog_s": slowest, "scores": results}, indent=2))
    print(f"\nwrote {out}   slowest catalog {slowest:.2f}s   "
          f"total {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
