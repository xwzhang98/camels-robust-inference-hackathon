# Tools

The code the organizers use, in the repository so you can read it, change it and run it. If
you want to know exactly how something is scored, this is where to look rather than guess.

Everything here takes `--data-root` and `--params-root`, so nothing assumes a cluster.

```
--data-root     the directory holding <suite>/LH_<n>/groups_090.hdf5
--params-root   the directory holding params_LH_<suite>.txt
```

On Colab, after the setup cell has run, those are `$CAMELS_HACKATHON_DATA` and
`$CAMELS_HACKATHON_PARAMS`.

## `score_submission.py`

Runs one submission over every condition and writes per-condition, per-target R². This is the
harness. Your submission is scored by this exact file, so if it runs here it runs there.

```bash
python tools/score_submission.py \
    --submission baseline/gnn --team my_team \
    --test-root  "$CAMELS_HACKATHON_DATA" \
    --params-root "$CAMELS_HACKATHON_PARAMS" \
    --private-split my_split.json \
    --suites IllustrisTNG SIMBA Astrid \
    --base-seed 12345 \
    --out results/my_team.json
```

`--private-split` is a JSON file of the form
`{"private_test": {"IllustrisTNG": [1, 2, 3], ...}}`. Ours holds simulations you do not have;
make your own from simulations you held out of training. Use your own `--base-seed` too —
the recipes are published, the organizers' seed is not.

## `train_gnn_baseline.py`

Trains the graph network and scores it under every condition, in four arms: one trained on all
three simulation codes, and three trained on two codes and tested on the third. That last shape
is the one that matters — it is the same kind of gap as the unseen simulation code you are
actually scored on.

```bash
python tools/train_gnn_baseline.py \
    --data-root "$CAMELS_HACKATHON_DATA" --params-root "$CAMELS_HACKATHON_PARAMS" \
    --arms all_three --n-train 300 --n-aug 3 --epochs 150 \
    --ckpt-dir ckpts --out my_gnn.json
```

Useful flags, and the reasons they exist:

| flag | what it changes |
|---|---|
| `--test-split` | `public_tail` holds out the last simulations of the public set. `auto` uses the organizers' held-out set if those files are present and falls back to `public_tail` otherwise, printing which it chose |
| `--node-velocity` | `vz` is the published feature and is **not** rotation invariant; `speed` is |
| `--use-mstar` | adds `log10(1 + M_star)` to the node. Better in distribution, worse across simulation codes — see `baseline/README.md` |
| `--centroid` | `mean` reproduces the published recipe, which is not invariant under a periodic translation; `circular` fixes it |
| `--r-link`, `--layers`, `--hidden` | the tuned values are the defaults. They came from a search over `Omega_m` alone; the same search against `sigma_8` picks a smaller radius and a bigger network |
| `--targets` | `Omega_m` alone is the published task; both is what this event scores |
| `--arms insuite_<suite>` | trains on one suite and tests on all three, which is the protocol de Santi et al. use |

It writes its report after every arm, not at the end, because a long run that is killed at the
end should not lose everything.

## `train_feature_baseline.py`

The same, for the 32 summary features and Ridge. No GPU, no torch, a few minutes.

## `compare_reports.py`

Renders any number of those JSON reports into one markdown table.

```bash
python tools/compare_reports.py \
    features=my_features.json gnn=my_gnn.json --out comparison.md
```
