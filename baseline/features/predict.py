"""Reference submission A: symmetry-invariant summary features + Ridge regression.

Deliberately simple, and the whole point is that it is beatable. It exists to (a) prove the
submission interface works end to end, (b) give every team a floor, and (c) act as the
harness smoke test.

It throws away every column except stellar mass and position. What it keeps -- counts, a
mass function, periodic neighbour counts -- is invariant under all three Tier S symmetries
by construction, so it scores identically before and after them. That is not a trick; it is
what "build the symmetry into the representation" buys you.

Interface (identical for every submission):

    load_model(model_dir) -> object          called once
    predict(model, catalog_path) -> dict     called once per test catalog
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from kaai_hackathon.catalog_io import read_catalog
from kaai_hackathon.features import catalog_features

#: Centre of the CAMELS latin hypercube, used only when no trained model is present.
#: A submission that returns this scores R^2 = 0 by construction: the mean predictor.
PRIOR = {"Omega_m": 0.3, "sigma_8": 0.8}


def load_model(model_dir: str):
    path = Path(model_dir) / "model.pkl"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict(model, catalog_path: str) -> dict:
    if model is None:
        return dict(PRIOR)
    cat = read_catalog(catalog_path, group_fields=[],
                       subhalo_fields=["SubhaloPos", "SubhaloMassType"])
    x = model["scaler"].transform(catalog_features(cat).reshape(1, -1))
    y = np.asarray(model["regressor"].predict(x)).ravel()
    return {"Omega_m": float(y[0]), "sigma_8": float(y[1])}
