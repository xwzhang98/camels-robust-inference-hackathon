# KAAI 2026 — robust inference from galaxy catalogs

Predict the cosmological parameters of a universe from a catalog of its galaxies — and keep
predicting them correctly when the catalog is changed.

You are given galaxy catalogs from cosmological hydrodynamic simulations, labelled with the
parameters that produced them. You train a regressor. At test time your model is run on
held-out catalogs that have been put through a ladder of changes, and you are scored under
each change separately.

**Targets.** `Omega_m` and `sigma_8`, scored separately and never averaged together. The four
astrophysical feedback parameters are an optional bonus track.

## The conditions you are scored under

| kind | conditions | what a drop means |
|---|---|---|
| **symmetries** | shift the box origin, turn it by a right angle, reorder the rows | none of these change the physics. If your score moves, that is your architecture, not the data |
| **corruptions** | noise on positions, on velocities, on masses | these really do destroy information. Some loss is expected; how much is the result |
| **a different simulation code** | a fourth hydrodynamics code you never see | the open research question this event is built around |

`sigma_8` under an unseen simulation code is expected to be hard — near zero or negative for
most methods that have been tried, in the published literature and in our own runs. That is
disclosed up front rather than hidden, and a team that gets a positive number there has done
something genuinely notable.

## Start here

Click a badge. Colab opens the notebook, and its first cell installs the toolkit, mounts the
data and sets everything up — you do not need an account beyond your Google login, and you do
not need to download anything.

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

A GPU makes notebook 04 pleasant and is not needed anywhere else. On Colab: **Runtime →
Change runtime type → T4 GPU**.

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

All three are valid submissions, load through the same function the scorer uses, and appear
on the leaderboard as floors. Clean scores, $R^2$ macro-averaged over the three suites:

| | `Omega_m` | `sigma_8` |
|---|---|---|
| `gnn` | **0.903** | 0.403 |
| `gnn_pos_vz` | 0.890 | 0.242 |
| `features` | 0.676 | **0.463** |

No single one wins every column, and that is the shape of the whole leaderboard. The two graph
networks differ by one node feature, and comparing them across simulation codes is the most
useful thing in this repository — `baseline/README.md` has that table.

## Submitting

A submission is a directory containing `predict.py`:

```python
def load_model(model_dir: str) -> object: ...
def predict(model, catalog_path: str) -> dict:
    """-> {"Omega_m": float, "sigma_8": float}, optionally the four feedback parameters."""
```

You submit code, not predictions, which is what keeps the test catalogs — and the unseen
simulation code — genuinely unseen. Copy `baseline/features/` and edit it; it already works.

## Data

CAMELS Subfind halo/subhalo catalogs at z = 0, `L25n256` latin-hypercube set, from the
IllustrisTNG, SIMBA and Astrid suites. 900 simulations per suite with full labels.

Positions are in ckpc/h, masses in 1e10 Msun/h, `HubbleParam = 0.6711`, and the 25000 ckpc/h
box is periodic. Notebook 00 covers this properly; units and the minimum-image convention are
the two most common sources of silent errors in this dataset.

## Credits

The graph baseline follows de Santi et al. 2023 ([arXiv:2302.14101](https://arxiv.org/abs/2302.14101)),
ported here without `torch_geometric`. Two departures from the published recipe are documented
in `kaai_hackathon/graph.py` and measured in notebook 04.
