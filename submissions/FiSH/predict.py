"""Team FiSH's submission.

Model: the reference de Santi et al. GNN (``gnn.py`` -- unmodified, re-exported from
``kaai_hackathon.gnn``), node features = "all_together" (line-of-sight velocity + 9 extra
Subhalo/Group columns including SubhaloGrNr; see ``features.py``).

Training summary (full details, every intermediate result, and every design decision are
in the source project's PROJECT_LOG.md):
  1. Trained from scratch on Astrid + IllustrisTNG (SIMBA held out), 800 sims/suite.
  2. Continued (not retrained from scratch) on ALL THREE simulation codes jointly
     (Astrid, IllustrisTNG, SIMBA), 80/20 train/val split per suit, WITH augmentation --
     per real training catalog: 1 clean copy + 3 randomly-rotated copies (``rotate90``) +
     1 randomly-corrupted copy (one of the 4 published Tier-C noise conditions) + 1
     randomly-translated copy (``translate``) -- built from
     ``kaai_hackathon.conditions.PUBLISHED_CONDITIONS`` / ``apply_condition``, unmodified,
     i.e. literally the same operations the event itself scores robustness with, used here
     to train it in instead of only measuring its absence.
  This checkpoint is the result of step 2 -- see this submission's README.md for the
  final numbers.

Interface (identical for every submission):

    load_model(model_dir) -> object          called once
    predict(model, catalog_path) -> dict     called once per test catalog
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from features import EXTRA_FIELDS, catalog_fields, to_graph
from gnn import DeSantiGNN, collate

from kaai_hackathon.catalog_io import read_catalog
from kaai_hackathon.graph import MSTAR_TEST_THRESHOLD

#: Must match the training script. Targets are trained on [0, 1] and mapped back here.
PRIOR_LO = np.array([0.1, 0.6])
PRIOR_HI = np.array([0.5, 1.0])

#: Fallback prior-centre prediction, used only if the checkpoint is missing.
PRIOR = {"Omega_m": 0.3, "sigma_8": 0.8}

_SUBHALO_FIELDS, _GROUP_FIELDS = catalog_fields()


def load_model(model_dir: str):
    path = Path(model_dir) / "model.pt"
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in checkpoint["stats"].items()}
    node_features = int(stats["x_mean"].shape[0])
    model = DeSantiGNN(node_features=node_features, edge_features=3, n_global=1,
                       hidden=64, n_layers=3, n_params=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "stats": stats}


@torch.no_grad()
def predict(model, catalog_path: str) -> dict:
    if model is None:
        return dict(PRIOR)
    cat = read_catalog(catalog_path, group_fields=_GROUP_FIELDS, subhalo_fields=_SUBHALO_FIELDS)
    graph = to_graph(cat, MSTAR_TEST_THRESHOLD)
    stats = model["stats"]
    graph = {**graph,
             "x": ((graph["x"] - stats["x_mean"]) / stats["x_std"]).astype(np.float32),
             "u": ((graph["u"] - stats["u_mean"]) / stats["u_std"]).astype(np.float32)}
    batch = collate([graph])
    mu, _ = DeSantiGNN.split_output(
        model["model"](batch["x"], batch["edge_index"], batch["edge_attr"],
                       batch["u"], batch["batch"]))
    y = mu.numpy().ravel() * (PRIOR_HI - PRIOR_LO) + PRIOR_LO
    return {"Omega_m": float(y[0]), "sigma_8": float(y[1])}
