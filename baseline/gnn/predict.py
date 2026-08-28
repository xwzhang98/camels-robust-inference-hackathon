"""Reference submission B: the de Santi et al. 2023 graph neural network.

Each catalog becomes a graph: galaxies above a stellar-mass cut are the nodes, nearby pairs
are the edges (periodic), and each edge carries the three features from the paper -- a scaled
distance and two angles.

Positions are never fed to the model as coordinates. They decide which pairs are connected
and they set those three edge features, and that is where the translation and rotation
invariance comes from. What a node carries is the galaxy itself.

**The specifics come from the checkpoint, not from this file.** The linking length, the
stellar-mass cut, whether the node carries a stellar mass alongside its velocity, and which
velocity -- all of it is read back out of `model.pt`, so this file never has to be kept in
sync with the training script and a checkpoint from a different configuration rebuilds the
right model. Print `checkpoint["args"]` if you want to know what a particular one is; the
shipped weights also carry a `provenance` block with their scores.

See notebook 04 for the walkthrough and notebook 05 for running this as a submission.

Interface (identical for every submission):

    load_model(model_dir) -> object          called once
    predict(model, catalog_path) -> dict     called once per test catalog
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kaai_hackathon.catalog_io import read_catalog
from kaai_hackathon.gnn import DeSantiGNN, collate
from kaai_hackathon.graph import (
    MSTAR_TEST_THRESHOLD, R_LINK, catalog_to_graph,
)

#: Centre of the CAMELS latin hypercube, used only when no checkpoint is present.
PRIOR = {"Omega_m": 0.3, "sigma_8": 0.8}
#: Must match the training script. Targets are trained on [0, 1] and mapped back here.
PRIOR_LO = np.array([0.1, 0.6])
PRIOR_HI = np.array([0.5, 1.0])


def load_model(model_dir: str):
    path = Path(model_dir) / "model.pt"
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    settings = checkpoint.get("args", {})
    use_mstar = bool(settings.get("use_mstar", False))
    positions_only = bool(settings.get("positions_only", False))
    velocity = str(settings.get("node_velocity", "vz"))
    node_dim = 1
    if not positions_only:
        node_dim = (velocity != "none") + use_mstar
    model = DeSantiGNN(node_features=node_dim, edge_features=3, n_global=1,
                       hidden=int(settings.get("hidden", 64)),
                       n_layers=int(settings.get("layers", 3)), n_params=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    stats = {k: np.asarray(v, dtype=np.float32)
             for k, v in checkpoint["stats"].items()}
    return {"model": model, "stats": stats,
            "r_link": float(settings.get("r_link", R_LINK)),
            "mstar_min": float(settings.get("mstar_min", MSTAR_TEST_THRESHOLD)),
            "centroid": str(settings.get("centroid", "circular")),
            "use_mstar": use_mstar, "positions_only": positions_only,
            "velocity": velocity}


@torch.no_grad()
def predict(model, catalog_path: str) -> dict:
    if model is None:
        return dict(PRIOR)
    cat = read_catalog(catalog_path, group_fields=[],
                       subhalo_fields=["SubhaloPos", "SubhaloVel", "SubhaloMassType"])
    graph = catalog_to_graph(cat, r_link=model["r_link"],
                             mstar_min=model["mstar_min"],
                             centroid=model["centroid"],
                             use_mstar=model["use_mstar"],
                             positions_only=model["positions_only"],
                             velocity=model["velocity"])
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
