# KAAI 2026 — robust inference from galaxy catalogs

Predict the cosmological parameters of a universe from a catalog of its galaxies — and keep
predicting them correctly when the catalog is changed.

You are given galaxy catalogs from cosmological hydrodynamic simulations, labelled with the
parameters that produced them. You train a regressor. At test time your model is run on
held-out catalogs that have been put through a ladder of changes, and you are scored under
each change separately.

**Targets.** `Omega_m` and `sigma_8`, scored separately.

## The conditions you are scored under

| kind | conditions | what a drop means |
|---|---|---|
| **symmetries** | shift the box origin, turn it by a right angle, reorder the rows | none of these change the physics. |
| **corruptions** | noise on positions, on velocities, on masses | these really do destroy information. Some loss is expected; how much is the result |
| **a different simulation code** | a fourth hydrodynamics code you never see | the open research question this event is built around |

`sigma_8` under an unseen simulation code is expected to be hard.

## Start here

Click the Colab badge opens the notebook, and its first cell installs the toolkit, mounts the
data and sets everything up. 

| | notebook | |
|---|---|---|
| **00** | What is in a catalog | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xwzhang98/kaai-robust-inference-hackathon-2026/blob/main/notebooks/00_explore_the_catalog.ipynb) |
| **01** | The labels, and what the task is | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xwzhang98/kaai-robust-inference-hackathon-2026/blob/main/notebooks/01_labels_and_the_task.ipynb) |
| **02** | A set of galaxies to a first model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xwzhang98/kaai-robust-inference-hackathon-2026/blob/main/notebooks/02_set_to_vector_first_model.ipynb) |
| **03** | Symmetries and shifts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xwzhang98/kaai-robust-inference-hackathon-2026/blob/main/notebooks/03_symmetries_and_shifts.ipynb) |
| **04** | The graph network baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xwzhang98/kaai-robust-inference-hackathon-2026/blob/main/notebooks/04_gnn_baseline.ipynb) |
| **05** | Running a baseline, and submitting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xwzhang98/kaai-robust-inference-hackathon-2026/blob/main/notebooks/05_submission.ipynb) |

Work through them in order.

The same notebooks run unchanged on a cluster where the data is already on disk. The setup
cell notices it is not on Colab and does nothing, so there is one set of notebooks rather
than two that drift apart. Set `CAMELS_HACKATHON_DATA` to the directory holding
`<suite>/LH_<n>/groups_090.hdf5` and `CAMELS_HACKATHON_PARAMS` to the one holding
`params_LH_<suite>.txt`.

A GPU node is needed to run notebook 04. On Colab: **Runtime →
Change runtime type → Some GPU**.

## What is in here

```
kaai_hackathon/       the package you import
  catalog_io.py         minimal reader/writer for a Subfind catalog
  splits.py             the public/held-out split and the parameter tables
  shifts.py             the shift operations
  conditions.py         the named conditions, with pinned per-catalog seeds
  features.py           a symmetry-invariant summary feature vector
  graph.py, gnn.py      a periodic radius graph and the de Santi et al. 2023 GNN
  submission.py         the submission contract
  scoring.py            R^2, per condition and per target
baseline/                 three working submissions -- see baseline/README.md
  gnn/                    the main baseline: a trained graph network, weights included
  gnn_pos_vz/             the same network with the paper's fiducial node feature
  features/               32 summary features + Ridge. No GPU, no torch
notebooks/                00 the catalogs, 01 the labels and the task,
                          02 a first model, 03 symmetries and shifts,
                          04 the GNN baseline, 05 running one and submitting
tools/                    the organizers' own scripts -- scoring, and training
                          both baselines. See tools/README.md
```

## Submitting

A submission is a directory containing `predict.py`, model.pt and gnn.py:

```python
def load_model(model_dir: str) -> object: ...
def predict(model, catalog_path: str) -> dict:
    """-> {"Omega_m": float, "sigma_8": float}, optionally the four feedback parameters."""
```

## Data

CAMELS Subfind halo/subhalo catalogs at z = 0, `L25n256` latin-hypercube set, from the
IllustrisTNG, SIMBA and Astrid suites. 900 simulations per suite with full labels.

Positions are in ckpc/h, masses in 1e10 Msun/h, `HubbleParam = 0.6711`, and the 25000 ckpc/h
box is periodic. Notebook 00 covers this properly.

## Credits

The graph baseline follows de Santi et al. 2023 ([arXiv:2302.14101](https://arxiv.org/abs/2302.14101)),
ported here without `torch_geometric`.
