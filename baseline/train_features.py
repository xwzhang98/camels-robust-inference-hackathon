#!/usr/bin/env python
"""Fit the Ridge summary-feature reference submission. Runs under Slurm, not on a login node.

Writes ``baseline/features/model.pkl``, which is what ``baseline/features/predict.py``
picks up. With no model file present that submission returns the prior mean and scores
R^2 = 0, so the harness still runs before this has ever been executed.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from kaai_hackathon import PUBLIC_SUITES
from kaai_hackathon.catalog_io import read_catalog
from kaai_hackathon.features import catalog_features
from kaai_hackathon.splits import load_labels, make_split

ALPHAS = np.logspace(-3, 4, 30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--params-root", required=True)
    parser.add_argument("--out", default="baseline/features/model.pkl")
    parser.add_argument("--max-sims", type=int, default=None,
                        help="limit per suite, for a fast smoke run")
    args = parser.parse_args()

    started = time.time()
    rows, targets = [], []
    for suite in PUBLIC_SUITES:
        labels = load_labels(args.params_root, suite)
        ids = make_split(suite)["public"]
        if args.max_sims:
            ids = ids[: args.max_sims]
        for sim_id in ids:
            path = Path(args.data_root) / suite / f"LH_{sim_id}" / "groups_090.hdf5"
            if not path.is_file():
                continue
            cat = read_catalog(path, group_fields=[],
                               subhalo_fields=["SubhaloPos", "SubhaloMassType"])
            rows.append(catalog_features(cat))
            targets.append(labels[sim_id, :2])          # Omega_m, sigma_8
        print(f"{suite:14s} {len(rows)} catalogs so far  [{time.time()-started:6.1f}s]",
              flush=True)

    x, y = np.asarray(rows), np.asarray(targets)
    print(f"fitting on {x.shape[0]} catalogs, {x.shape[1]} features")
    scaler = StandardScaler().fit(x)
    regressor = RidgeCV(alphas=ALPHAS).fit(scaler.transform(x), y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump({"scaler": scaler, "regressor": regressor}, handle)
    print(f"wrote {out}  [{time.time()-started:.1f}s]")


if __name__ == "__main__":
    main()
