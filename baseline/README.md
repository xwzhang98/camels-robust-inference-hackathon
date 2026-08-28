# Reference submissions

Three working submissions. Each is a directory with a `predict.py` exposing `load_model` and
`predict`, loaded through exactly the same function the organizers' evaluation runner uses.
Copy one, edit it, and you have your own.

| directory | what it is | clean `Omega_m` | clean `sigma_8` |
|---|---|---|---|
| `gnn/` | **the main baseline.** de Santi graph network, node = $(v_z, \log_{10}(1+M_\star))$ | **0.903** | **0.403** |
| `gnn_pos_vz/` | the same network with the paper's fiducial node feature, $v_z$ alone | 0.890 | 0.242 |
| `features/` | 32 summary features + Ridge. No GPU, no torch, fits in a minute | 0.676 | 0.463 |

Scores are $R^2$ on the held-out 100 simulations of each public suite, macro-averaged over
suites, trained on all three. `features/` ships without weights on purpose — refit it with
`baseline/train_features.py`, because a pickle downloaded from the internet is a
code-execution hazard and you should not be in the habit of loading one.

## Why two graph networks

Because the difference between them is the most useful thing in this repository.

| | in-distribution | trained without TNG | trained without SIMBA | trained without Astrid |
|---|---|---|---|---|
| `gnn` (with $M_\star$) | 0.903 / 0.403 | 0.863 / 0.237 | 0.909 / 0.149 | **0.599 / −0.358** |
| `gnn_pos_vz` (without) | 0.890 / 0.242 | 0.875 / 0.236 | 0.897 / 0.205 | **0.773 / −0.101** |

Adding stellar mass to the node buys **+0.16 on `sigma_8`** in distribution and costs
**−0.17 on `Omega_m` and −0.26 on `sigma_8`** when the simulation code changes to Astrid.

That is not an artifact of our runs. The independent reproduction this port is derived from
sees the same thing in the other direction: its `pos_vz_mstar` variant scores 0.534 on
IllustrisTNG where `pos_vz` scores 0.829.

The reason is physical. Stellar mass is where the sub-grid physics of different simulation
codes disagrees most, so a model that learns to lean on the stellar mass distribution learns
something partly specific to the codes it was trained on. Velocities and positions are set by
gravity, which every code implements the same way.

**The best in-distribution model is not the best out-of-distribution model.** You will meet
this again. Your own validation split is in distribution; the condition that decides this
event is not.
