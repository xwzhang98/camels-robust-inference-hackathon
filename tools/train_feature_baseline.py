#!/usr/bin/env python
"""Fit the summary-feature baseline and score it under every shift condition.

Organizer-side. Answers two questions at once:

  1. How much accuracy does each condition actually cost? Levels that leave the baseline
     unscathed are not testing anything; levels that take it to R^2 < 0 teach nothing.
  2. How large is the cross-suite gap? A leave-one-suite-out arm trains on two simulation
     codes and tests on the third, which is the shape of the held-out OOD condition.

Writes a JSON report and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from kaai_hackathon import PUBLIC_SUITES
from kaai_hackathon.catalog_io import read_catalog
from kaai_hackathon.conditions import (
    PUBLISHED_CONDITIONS, apply_condition, condition_seed,
)
from kaai_hackathon.features import catalog_features
from kaai_hackathon.splits import load_labels, make_split

TARGETS = ("Omega_m", "sigma_8")
ALPHAS = np.logspace(-3, 4, 30)

# Enough columns for every published condition and for the features.
GROUP_FIELDS = ["GroupPos", "GroupCM", "GroupVel", "GroupMass",
                "GroupNsubs", "GroupFirstSub"]
SUBHALO_FIELDS = ["SubhaloPos", "SubhaloCM", "SubhaloVel", "SubhaloSpin",
                  "SubhaloMass", "SubhaloMassType", "SubhaloGrNr", "SubhaloParent"]


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


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - residual / total


def catalog_file(data_root: Path, suite: str, sim_id: int) -> Path:
    return data_root / suite / f"LH_{sim_id}" / "groups_090.hdf5"


def fit(features: np.ndarray, labels: np.ndarray):
    scaler = StandardScaler().fit(features)
    model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(features), labels)
    return scaler, model


def predict(fitted, features: np.ndarray) -> np.ndarray:
    scaler, model = fitted
    return model.predict(scaler.transform(features))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--params-root", required=True)
    parser.add_argument("--n-train", type=int, default=900, help="public sims per suite")
    parser.add_argument("--test-split", default="auto",
                        choices=("auto", "private_test", "public_tail"),
                        help="which simulations to score on; 'auto' detects what is on disk")
    parser.add_argument("--n-test", type=int, default=100, help="held-out sims per suite")
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--out", default="reports/feature_baseline.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    started = time.time()

    # ---- training features (clean only) --------------------------------------------
    train_x: dict[str, list] = {}
    train_y: dict[str, list] = {}
    chosen = {}
    for suite in PUBLIC_SUITES:
        labels = load_labels(args.params_root, suite)
        ids, test_ids, mode = resolve_split(data_root, suite, args.test_split,
                                            args.n_train, args.n_test)
        chosen[suite] = (test_ids, mode)
        rows, targets = [], []
        for sim_id in ids:
            cat = read_catalog(catalog_file(data_root, suite, sim_id),
                               group_fields=[], subhalo_fields=["SubhaloPos",
                                                                "SubhaloMassType"])
            rows.append(catalog_features(cat))
            targets.append(labels[sim_id, :2])
        train_x[suite] = np.asarray(rows)
        train_y[suite] = np.asarray(targets)
        print(f"train {suite:14s} {len(rows)} sims  [{time.time()-started:6.1f}s]",
              flush=True)

    # ---- test features, once per condition ------------------------------------------
    test_x: dict[tuple[str, str], np.ndarray] = {}
    test_y: dict[str, np.ndarray] = {}
    for suite in PUBLIC_SUITES:
        labels = load_labels(args.params_root, suite)
        ids, mode = chosen[suite]
        print(f"      test split for {suite}: {mode} ({len(ids)} simulations)", flush=True)
        test_y[suite] = np.asarray([labels[sim_id, :2] for sim_id in ids])
        per_condition = {spec.name: [] for spec in PUBLISHED_CONDITIONS}
        for sim_id in ids:
            cat = read_catalog(catalog_file(data_root, suite, sim_id),
                               group_fields=GROUP_FIELDS, subhalo_fields=SUBHALO_FIELDS)
            for spec in PUBLISHED_CONDITIONS:
                seed = condition_seed(args.base_seed, spec.name, suite, sim_id)
                per_condition[spec.name].append(
                    catalog_features(apply_condition(cat, spec, seed)))
        for name, rows in per_condition.items():
            test_x[(suite, name)] = np.asarray(rows)
        print(f"test  {suite:14s} {len(ids)} sims x {len(PUBLISHED_CONDITIONS)} conditions"
              f"  [{time.time()-started:6.1f}s]", flush=True)

    # ---- arms -----------------------------------------------------------------------
    arms = {"all_three": {"train": list(PUBLIC_SUITES)}}
    for suite in PUBLIC_SUITES:
        arms[f"holdout_{suite}"] = {"train": [s for s in PUBLIC_SUITES if s != suite],
                                    "test_only": suite}

    report: dict = {"n_train_per_suite": args.n_train, "n_test_per_suite": args.n_test,
                    "base_seed": args.base_seed, "arms": {}}

    for arm_name, arm in arms.items():
        fitted = fit(np.concatenate([train_x[s] for s in arm["train"]]),
                     np.concatenate([train_y[s] for s in arm["train"]]))
        test_suites = ([arm["test_only"]] if "test_only" in arm else list(PUBLIC_SUITES))
        scores: dict = {}
        for spec in PUBLISHED_CONDITIONS:
            per_suite = {}
            for suite in test_suites:
                prediction = predict(fitted, test_x[(suite, spec.name)])
                per_suite[suite] = {t: r2(test_y[suite][:, j], prediction[:, j])
                                    for j, t in enumerate(TARGETS)}
            scores[spec.name] = {
                "per_suite": per_suite,
                "macro": {t: float(np.mean([per_suite[s][t] for s in test_suites]))
                          for t in TARGETS},
                "tier": spec.tier,
            }
        report["arms"][arm_name] = {"train_suites": arm["train"],
                                    "test_suites": test_suites, "scores": scores}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    # ---- printed summary ------------------------------------------------------------
    for arm_name, arm in report["arms"].items():
        print(f"\n### {arm_name}   train={'+'.join(arm['train_suites'])}   "
              f"test={'+'.join(arm['test_suites'])}")
        print(f"| {'condition':<14} | tier | {'Omega_m':>8} | {'sigma_8':>8} |")
        print(f"|{'-'*16}|------|{'-'*10}|{'-'*10}|")
        for name, entry in arm["scores"].items():
            print(f"| {name:<14} | {entry['tier']:<4} | "
                  f"{entry['macro']['Omega_m']:8.3f} | {entry['macro']['sigma_8']:8.3f} |")

    print(f"\nwrote {out}   total {time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
