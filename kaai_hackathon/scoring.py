"""R^2 scoring, reported per condition and per target.

Two rules are deliberate and are enforced here rather than left to whoever writes the
report:

* **Targets are never averaged together.** ``sigma_8`` is the hard one and is expected to
  fail for most methods; folding it into a single score with ``Omega_m`` would hide that.
* **Suites are macro-averaged.** A suite is one simulation code. Weighting by catalog count
  would let the largest test set decide the ranking.

The leaderboard is therefore a matrix of (condition x target), not a scalar. A team can win
on robustness without winning on clean accuracy, which is the point of the event.
"""
from __future__ import annotations

import numpy as np


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. 1 is perfect, 0 is the mean predictor, below 0 is worse.

    A constant target raises rather than returning NaN: it means the test set was built
    wrong, and silently propagating NaN makes that much harder to find.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot score an empty set")
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total == 0.0:
        raise ValueError("target has zero variance; R^2 is undefined")
    return 1.0 - residual / total


def score_condition(preds: dict, truths: dict, targets: tuple[str, ...]) -> dict:
    """``-> {target: R^2}`` for one condition on one suite."""
    return {t: r2(truths[t], preds[t]) for t in targets if t in truths and t in preds}


def macro_average(per_suite: dict[str, dict[str, float]],
                  targets: tuple[str, ...]) -> dict[str, float]:
    """Average each target over suites, weighting every suite equally.

    Raises if a suite is missing a target, rather than averaging over whatever happens to
    be present -- a silently dropped suite is a silently different metric.
    """
    if not per_suite:
        raise ValueError("nothing to average")
    out = {}
    for target in targets:
        values = []
        for suite, scores in per_suite.items():
            if target not in scores:
                raise ValueError(f"suite {suite!r} has no score for {target}")
            values.append(float(scores[target]))
        out[target] = float(np.mean(values))
    return out


def leaderboard_table(results: dict) -> str:
    """``{team: {condition: {target: r2}}}`` -> a markdown table, one row per team.

    Conditions keep the order they were first seen, so passing
    ``PUBLISHED_CONDITIONS`` order in gives tier order out. A cell with no score prints as
    an em dash instead of dropping the team.
    """
    conditions: list[str] = []
    targets: list[str] = []
    for per_condition in results.values():
        for condition, per_target in per_condition.items():
            if condition not in conditions:
                conditions.append(condition)
            for target in per_target:
                if target not in targets:
                    targets.append(target)

    header = ["team"] + [f"{c} / {t}" for c in conditions for t in targets]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for team in sorted(results):
        row = [team]
        for condition in conditions:
            for target in targets:
                value = results[team].get(condition, {}).get(target)
                row.append("—" if value is None else f"{value:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
