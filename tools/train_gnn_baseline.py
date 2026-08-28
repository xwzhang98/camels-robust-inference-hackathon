#!/usr/bin/env python
"""Train the de Santi GNN baseline and score it under every shift condition.

Organizer-side twin of ``evaluate_feature_baseline.py``: same arms, same JSON shape, same
printed table, so the two baselines are directly comparable row for row. The GNN is the
stronger floor and the content of notebook 04; the summary-feature model is the floor that
needs no GPU.

Two things this script is really measuring, beyond "how good is the GNN":

  1. **Does Tier S cost it anything?** It should not. The edge features are invariant by
     construction, so any degradation is a bug in ours or theirs -- see the ``--centroid``
     flag, which reproduces the published behaviour and does lose accuracy on
     ``translate``.
  2. **How big is the cross-suite gap for a real model?** The leave-one-suite-out arms are
     the shape of the held-out OOD condition.

Not for the login node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from kaai_hackathon import PUBLIC_SUITES
from kaai_hackathon.catalog_io import read_catalog
from kaai_hackathon.conditions import (
    PUBLISHED_CONDITIONS, apply_condition, condition_seed,
)
from kaai_hackathon.gnn import DeSantiGNN, collate, moment_loss
from kaai_hackathon.graph import (
    MSTAR_TEST_THRESHOLD, R_LINK, catalog_to_graph, sample_mstar_threshold,
)
from kaai_hackathon.splits import load_labels, make_split

ALL_TARGETS = ("Omega_m", "sigma_8")
#: The latin-hypercube prior, used to map targets onto [0, 1]. Training on raw values makes
#: the loss lopsided: sigma_8 lives on [0.6, 1.0] and Omega_m on [0.1, 0.5].
ALL_PRIOR_LO = np.array([0.1, 0.6])
ALL_PRIOR_HI = np.array([0.5, 1.0])
# Set from --targets in main(). Kept module-level because the helpers below are also used
# by the reproduction check, which trains on Omega_m alone the way the paper does.
TARGETS = ALL_TARGETS
PRIOR_LO, PRIOR_HI = ALL_PRIOR_LO, ALL_PRIOR_HI

GROUP_FIELDS = ["GroupPos", "GroupCM", "GroupVel", "GroupMass",
                "GroupNsubs", "GroupFirstSub"]
SUBHALO_FIELDS = ["SubhaloPos", "SubhaloCM", "SubhaloVel", "SubhaloSpin",
                  "SubhaloMass", "SubhaloMassType", "SubhaloGrNr", "SubhaloParent"]
#: Columns needed to build a graph from an unshifted catalog.
GRAPH_FIELDS = ["SubhaloPos", "SubhaloVel", "SubhaloMassType"]


def resolve_split(data_root: Path, suite: str, mode: str, n_train: int, n_test: int):
    """-> (train_ids, test_ids, chosen_mode).

    The organizers hold out 100 simulations per suite that participants never receive, so a
    script that assumes those files exist works for us and crashes for everyone else. This
    picks whichever split the data on disk can actually support and says which it picked --
    a fallback you cannot see is worse than a crash.

      private_test  the organizers' held-out 100. Trains on all 900 public simulations.
      public_tail   the last `n_test` public simulations, held out of training. What a
                    participant gets, and the right thing to develop against.
      auto          private_test if those catalogs are present, otherwise public_tail.
    """
    split = make_split(suite)
    if mode == "auto":
        first = split["private_test"][0]
        mode = ("private_test"
                if catalog_file(data_root, suite, first).is_file() else "public_tail")
    if mode == "private_test":
        return split["public"][:n_train], split["private_test"][:n_test], mode
    if mode == "public_tail":
        public = split["public"]
        if n_test >= len(public):
            raise SystemExit(f"--n-test {n_test} leaves no training simulations")
        return public[:-n_test][:n_train], public[-n_test:], mode
    raise SystemExit(f"unknown --test-split {mode!r}")


def stable_seed(seed: int, suite: str) -> int:
    """A per-suite seed that is the same in every process.

    ``hash()`` on a string is salted per interpreter run, so using it here would silently
    give a different training set on every rerun.
    """
    digest = hashlib.sha256(f"{seed}|{suite}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def epsilon_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean relative error in percent -- the figure de Santi et al. report.

    R^2 depends on the spread of the test set, so it is not comparable across papers that
    used different splits. This is, which is why the reproduction check below is judged on it.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12)) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - residual / total


def catalog_file(data_root: Path, suite: str, sim_id: int) -> Path:
    return data_root / suite / f"LH_{sim_id}" / "groups_090.hdf5"


def normalize_targets(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype=np.float64) - PRIOR_LO) / (PRIOR_HI - PRIOR_LO)


def denormalize_targets(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) * (PRIOR_HI - PRIOR_LO) + PRIOR_LO


class GraphStats:
    """Feature standardization, fitted on the training graphs only.

    Node and global features are log-masses and a log-count; leaving them unstandardized
    trains fine but slowly, and the scale differs between suites.
    """

    def __init__(self, graphs: list[dict]):
        x = np.concatenate([g["x"] for g in graphs]) if graphs else np.zeros((1, 1))
        u = np.stack([g["u"] for g in graphs]) if graphs else np.zeros((1, 1))
        self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-8
        self.u_mean, self.u_std = u.mean(0), u.std(0) + 1e-8

    def apply(self, g: dict) -> dict:
        return {**g,
                "x": ((g["x"] - self.x_mean) / self.x_std).astype(np.float32),
                "u": ((g["u"] - self.u_mean) / self.u_std).astype(np.float32)}

    def as_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in
                (("x_mean", self.x_mean), ("x_std", self.x_std),
                 ("u_mean", self.u_mean), ("u_std", self.u_std))}


def batches(n: int, size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, size):
        yield order[start:start + size]


def run_model(model, graphs: list[dict], index, device) -> torch.Tensor:
    batch = collate([graphs[i] for i in index])
    return model(batch["x"].to(device), batch["edge_index"].to(device),
                 batch["edge_attr"].to(device), batch["u"].to(device),
                 batch["batch"].to(device))


def train_one(train_graphs: list[dict], train_targets: np.ndarray,
              val_graphs: list[dict], val_targets: np.ndarray,
              args, device, log_prefix: str):
    """Fit on the given graphs; return the best-validation model and its loss.

    The train and validation sets are built by the caller and are disjoint **by simulation**,
    which matters here in a way it does not for an ordinary model: every simulation appears
    ``n_aug`` times in training with a different random stellar-mass cut, so splitting the
    graph list at random would put copies of the same simulation on both sides and the
    validation loss would be measuring memorization.

    Validation uses the fixed evaluation cut and no augmentation, so the number it reports is
    comparable to the test scores. Selection is on **validation** loss; selecting on training
    loss is the classic way to ship a model that has memorized its training suites, which is
    exactly the failure the held-out-suite arms are meant to expose.
    """
    rng = np.random.default_rng(args.seed)
    y_train = torch.as_tensor(normalize_targets(train_targets),
                              dtype=torch.float32, device=device)
    y_val = torch.as_tensor(normalize_targets(val_targets),
                            dtype=torch.float32, device=device)
    val_idx = np.arange(len(val_graphs))

    torch.manual_seed(args.seed)
    node_dim = int(train_graphs[0]["x"].shape[1])
    model = DeSantiGNN(node_features=node_dim, edge_features=3, n_global=1,
                       hidden=args.hidden, n_layers=args.layers, n_params=len(TARGETS)).to(device)
    # de Santi's recipe: Adam plus a triangular cyclic learning rate from 1e-6 to 1e-3.
    # CyclicLR counts step_size_up in OPTIMIZER STEPS, so pinning it in steps makes the
    # schedule silently depend on the batch size and on n_aug. Pin the half-cycle in EPOCHS
    # instead, so the LR curve keeps its shape whatever the loader length.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    steps_per_epoch = max(1, int(np.ceil(len(train_graphs) / args.batch_size)))
    schedule = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=args.lr, max_lr=args.max_lr,
        step_size_up=max(1, int(round(args.lr_cycle_epochs * steps_per_epoch))),
        cycle_momentum=False)
    print(f"{log_prefix} {len(train_graphs)} train / {len(val_graphs)} val graphs, "
          f"{steps_per_epoch} steps/epoch, node_dim={node_dim}", flush=True)

    best = (float("inf"), {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()})
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for index in batches(len(train_graphs), args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = moment_loss(run_model(model, train_graphs, index, device),
                                  y_train[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += float(loss.item()) * len(index)
        model.eval()
        with torch.no_grad():
            val_loss = float(moment_loss(
                run_model(model, val_graphs, val_idx, device), y_val)[0].item())
        if val_loss < best[0]:
            best = (val_loss, {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()})
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"{log_prefix} epoch {epoch:4d}  train {total/len(train_graphs):+8.4f}  "
                  f"val {val_loss:+8.4f}  best {best[0]:+8.4f}  "
                  f"lr {schedule.get_last_lr()[0]:.2e}", flush=True)
    model.load_state_dict(best[1])
    return model, best[0]


@torch.no_grad()
def predict(model, graphs: list[dict], device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for index in batches(len(graphs), batch_size, None):
        mu, _ = DeSantiGNN.split_output(run_model(model, graphs, index, device))
        out.append(mu.cpu().numpy())
    return denormalize_targets(np.concatenate(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--params-root", required=True)
    parser.add_argument("--n-train", type=int, default=900)
    parser.add_argument("--test-split", default="auto",
                        choices=("auto", "private_test", "public_tail"),
                        help="which simulations to score on; 'auto' detects what is on disk")
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r-link", type=float, default=R_LINK)
    parser.add_argument("--mstar-min", type=float, default=MSTAR_TEST_THRESHOLD,
                        help="fixed cut used for every evaluated catalog")
    parser.add_argument("--n-aug", type=int, default=10,
                        help="training copies per simulation, each with its own randomly "
                             "drawn stellar-mass cut (de Santi's augmentation); 1 disables it")
    parser.add_argument("--use-mstar", action="store_true",
                        help="append log10(1 + M_star) to the node feature")
    parser.add_argument("--node-velocity", default="vz", choices=("vz", "speed", "none"),
                        help="'vz' is de Santi's line-of-sight feature and is NOT invariant "
                             "under the cubic rotations; 'speed' is")
    parser.add_argument("--positions-only", action="store_true",
                        help="zero node features; everything enters via edges and the count")
    parser.add_argument("--centroid", default="circular", choices=("circular", "mean"),
                        help="'mean' reproduces the published de Santi behaviour, which is "
                             "not invariant under periodic translation")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-6, help="CyclicLR base_lr")
    parser.add_argument("--max-lr", type=float, default=1e-3, help="CyclicLR max_lr")
    parser.add_argument("--lr-cycle-epochs", type=float, default=62.5,
                        help="half-cycle length in epochs; 62.5 over 300 epochs reproduces "
                             "the published LR-versus-progress curve")
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--targets", nargs="+", default=list(ALL_TARGETS),
                        choices=list(ALL_TARGETS),
                        help="which parameters to predict. The published de Santi headline "
                             "task is Omega_m alone")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="subset of arm names. Beyond the four defaults, 'insuite_<suite>' "
                             "trains on that suite alone and tests on all three -- the "
                             "protocol the paper uses")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-dir", default=None,
                        help="if given, write <arm>.pt for each fitted arm")
    parser.add_argument("--out", default="reports/gnn_baseline.json")
    args = parser.parse_args()

    global TARGETS, PRIOR_LO, PRIOR_HI
    TARGETS = tuple(args.targets)
    keep = [ALL_TARGETS.index(t) for t in TARGETS]
    PRIOR_LO, PRIOR_HI = ALL_PRIOR_LO[keep], ALL_PRIOR_HI[keep]

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    started = time.time()

    def to_graph(cat, mstar_min=None):
        return catalog_to_graph(cat, r_link=args.r_link,
                                mstar_min=args.mstar_min if mstar_min is None else mstar_min,
                                centroid=args.centroid, use_mstar=args.use_mstar,
                                positions_only=args.positions_only,
                                velocity=args.node_velocity)

    # ---- training graphs (clean only) -----------------------------------------------
    chosen: dict[str, tuple] = {}
    train_g: dict[str, list] = {}
    train_y: dict[str, np.ndarray] = {}
    val_g: dict[str, list] = {}
    val_y: dict[str, np.ndarray] = {}
    for suite in PUBLIC_SUITES:
        labels = load_labels(args.params_root, suite)
        train_ids_all, test_ids, split_mode = resolve_split(
            data_root, suite, args.test_split, args.n_train, args.n_test)
        chosen[suite] = (test_ids, split_mode)
        ids = np.asarray(train_ids_all)
        # Split by SIMULATION before any augmentation, so no simulation can appear on both
        # sides of the split under a different stellar-mass cut.
        rng = np.random.default_rng(stable_seed(args.seed, suite))
        shuffled = rng.permutation(len(ids))
        n_val = max(1, int(round(args.val_fraction * len(ids))))
        val_ids, train_ids = ids[shuffled[:n_val]], ids[shuffled[n_val:]]

        graphs, targets = [], []
        for sim_id in train_ids:
            cat = read_catalog(catalog_file(data_root, suite, int(sim_id)),
                               group_fields=[], subhalo_fields=GRAPH_FIELDS)
            for _ in range(max(1, args.n_aug)):
                cut = args.mstar_min if args.n_aug <= 1 else sample_mstar_threshold(rng)
                graphs.append(to_graph(cat, mstar_min=cut))
                targets.append(labels[int(sim_id), keep])
        train_g[suite], train_y[suite] = graphs, np.asarray(targets)

        graphs, targets = [], []
        for sim_id in val_ids:          # fixed cut, no augmentation -- like the test set
            cat = read_catalog(catalog_file(data_root, suite, int(sim_id)),
                               group_fields=[], subhalo_fields=GRAPH_FIELDS)
            graphs.append(to_graph(cat))
            targets.append(labels[int(sim_id), keep])
        val_g[suite], val_y[suite] = graphs, np.asarray(targets)

        sizes = np.array([len(g["x"]) for g in train_g[suite]])
        edges = np.array([g["edge_index"].shape[1] for g in train_g[suite]])
        print(f"train {suite:14s} {len(train_ids)} sims x {max(1, args.n_aug)} aug = "
              f"{len(train_g[suite])} graphs (+{len(val_ids)} val sims)  "
              f"nodes {sizes.mean():6.0f}+-{sizes.std():.0f}  edges {edges.mean():7.0f}  "
              f"under 20 galaxies: {int((sizes < 20).sum())}  "
              f"[{time.time()-started:6.1f}s]", flush=True)

    # ---- test graphs, once per condition ---------------------------------------------
    test_g: dict[tuple[str, str], list] = {}
    test_y: dict[str, np.ndarray] = {}
    for suite in PUBLIC_SUITES:
        labels = load_labels(args.params_root, suite)
        ids, split_mode = chosen[suite]
        print(f"      test split for {suite}: {split_mode} ({len(ids)} simulations)",
              flush=True)
        test_y[suite] = np.asarray([labels[i, keep] for i in ids])
        per_condition: dict[str, list] = {s.name: [] for s in PUBLISHED_CONDITIONS}
        for sim_id in ids:
            cat = read_catalog(catalog_file(data_root, suite, sim_id),
                               group_fields=GROUP_FIELDS, subhalo_fields=SUBHALO_FIELDS)
            for spec in PUBLISHED_CONDITIONS:  # evaluation always uses the fixed cut
                seed = condition_seed(args.base_seed, spec.name, suite, sim_id)
                per_condition[spec.name].append(to_graph(apply_condition(cat, spec, seed)))
        for name, graphs in per_condition.items():
            test_g[(suite, name)] = graphs
        print(f"test  {suite:14s} {len(ids)} sims x {len(PUBLISHED_CONDITIONS)} conditions"
              f"  [{time.time()-started:6.1f}s]", flush=True)

    # ---- arms -------------------------------------------------------------------------
    all_arms = {"all_three": {"train": list(PUBLIC_SUITES)}}
    for suite in PUBLIC_SUITES:
        all_arms[f"holdout_{suite}"] = {"train": [s for s in PUBLIC_SUITES if s != suite],
                                        "test_only": suite}
        # The paper's own protocol: one training suite, evaluated in-suite and cross-suite.
        all_arms[f"insuite_{suite}"] = {"train": [suite]}
    arms = {k: v for k, v in all_arms.items() if args.arms is None or k in args.arms}
    if args.arms:
        missing = [a for a in args.arms if a not in all_arms]
        if missing:
            raise SystemExit(f"unknown arm(s): {missing}; known: {sorted(all_arms)}")

    report: dict = {"model": "de Santi MetaLayer GNN",
                    "n_train_per_suite": args.n_train, "n_test_per_suite": args.n_test,
                    "base_seed": args.base_seed, "seed": args.seed,
                    "r_link": args.r_link, "mstar_min": args.mstar_min,
                    "centroid": args.centroid, "hidden": args.hidden,
                    "layers": args.layers, "epochs": args.epochs,
                    "batch_size": args.batch_size, "lr": args.lr, "max_lr": args.max_lr,
                    "targets": list(TARGETS),
                    "n_aug": args.n_aug, "use_mstar": args.use_mstar,
                    "node_velocity": args.node_velocity,
                    "positions_only": args.positions_only, "arms": {}}

    for arm_name, arm in arms.items():
        graphs = [g for s in arm["train"] for g in train_g[s]]
        targets = np.concatenate([train_y[s] for s in arm["train"]])
        stats = GraphStats(graphs)                      # fitted on training graphs only
        val_graphs = [stats.apply(g) for s in arm["train"] for g in val_g[s]]
        val_targets = np.concatenate([val_y[s] for s in arm["train"]])
        model, val_loss = train_one([stats.apply(g) for g in graphs], targets,
                                    val_graphs, val_targets, args, device,
                                    f"[{arm_name}]")
        if args.ckpt_dir:
            ckpt_dir = Path(args.ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "stats": stats.as_dict(),
                        "args": vars(args), "val_loss": val_loss},
                       ckpt_dir / f"{arm_name}.pt")

        test_suites = [arm["test_only"]] if "test_only" in arm else list(PUBLIC_SUITES)
        scores: dict = {}
        for spec in PUBLISHED_CONDITIONS:
            per_suite = {}
            for suite in test_suites:
                prediction = predict(model,
                                     [stats.apply(g) for g in test_g[(suite, spec.name)]],
                                     device, args.batch_size)
                per_suite[suite] = {t: r2(test_y[suite][:, j], prediction[:, j])
                                    for j, t in enumerate(TARGETS)}
                per_suite[suite].update(
                    {f"{t}_epsilon_pct": epsilon_pct(test_y[suite][:, j], prediction[:, j])
                     for j, t in enumerate(TARGETS)})
            scores[spec.name] = {
                "per_suite": per_suite,
                "macro": {t: float(np.mean([per_suite[s][t] for s in test_suites]))
                          for t in TARGETS},
                "tier": spec.tier,
            }
        report["arms"][arm_name] = {"train_suites": arm["train"],
                                    "test_suites": test_suites,
                                    "val_loss": val_loss, "scores": scores}
        # Written after every arm, not at the end: these runs are long enough to meet the
        # queue's wall-clock limit, and a report that only exists at the end is a report you
        # lose all of when the last arm is the one that gets killed.
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[{arm_name}] done, report updated  [{time.time()-started:6.1f}s]", flush=True)

    out = Path(args.out)

    for arm_name, arm in report["arms"].items():
        print(f"\n### {arm_name}   train={'+'.join(arm['train_suites'])}   "
              f"test={'+'.join(arm['test_suites'])}   val_loss={arm['val_loss']:.4f}")
        print(f"| {'condition':<14} | tier | " +
              " | ".join(f"{t:>8}" for t in TARGETS) + " |")
        print(f"|{'-'*16}|------|" + "|".join(["-"*10] * len(TARGETS)) + "|")
        clean = arm["scores"]["clean"]["per_suite"]
        for suite, sc in clean.items():
            print(f"  clean {suite:14s} " + "  ".join(
                f"{t} R2 {sc[t]:+.3f} eps {sc[f'{t}_epsilon_pct']:.2f}%" for t in TARGETS))
        for name, entry in arm["scores"].items():
            print(f"| {name:<14} | {entry['tier']:<4} | " +
                  " | ".join(f"{entry['macro'][t]:8.3f}" for t in TARGETS) + " |")

    print(f"\nwrote {out}   total {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
