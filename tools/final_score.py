#!/usr/bin/env python
"""Turn the two scored reports into the 100-point leaderboard score.

    score(t) = 20 * <R2>_ID  +  15 * <R2>_OOD  +  15 * <r>_OOD
    total    = score(Omega_m) + score(sigma_8)

`<.>` averages over the four symmetry conditions -- clean, translate, rotate90, permute --
after clipping each value to `max(0, .)`. `<R2>_ID` is additionally macro-averaged over the
three public suites. In-distribution is worth 40 of the 100 points and the unseen
simulation code 60, which is the weighting saying out loud what the event is about.

Negatives clip to zero so the parts add up and nobody lands below zero because one cell
came out at -2. That also means a model predicting a constant scores 0 rather than being
punished twice.

`r` carries its own 15 points on the unseen code because it asks a different question than
R2 does. R2 asks whether the predictions are right; r asks whether they are right in order.
A model that has learned the trend on a code it never saw but is systematically offset can
have a negative R2 and a high r, and that is a real result worth points.

Usage:

    python tools/final_score.py --id results/team_id.json --ood results/team_ood.json
    python tools/final_score.py --results-dir results/round1 --out leaderboard.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TARGETS = ("Omega_m", "sigma_8")
#: The four conditions that count. None of the corruptions are here: the score is about
#: symmetries a correct model is unaffected by, and about transferring to an unseen code.
CONDITIONS = ("clean", "translate", "rotate90", "permute")
WEIGHTS = {"id_r2": 20.0, "ood_r2": 15.0, "ood_r": 15.0}


def _clip(x: float) -> float:
    return max(0.0, float(x)) if np.isfinite(x) else 0.0


def _mean_over_conditions(report: dict, key: str, target: str,
                          per_suite: bool) -> tuple[float, list]:
    """Average one metric over the four conditions, clipping each value to >= 0.

    `per_suite=True` clips every (condition, suite) cell before averaging, which is what
    "clip each value first" says literally. `per_suite=False` clips the suite-macro
    instead. They differ only when a suite is negative while the macro is positive.
    """
    values, detail = [], []
    for condition in CONDITIONS:
        entry = report["scores"].get(condition)
        if entry is None:
            raise SystemExit(f"{report.get('team')}: no result for condition {condition!r}")
        if per_suite:
            cells = [_clip(s[target]) for s in entry[key.replace("macro", "per_suite")].values()]
            value = float(np.mean(cells)) if cells else 0.0
        else:
            value = _clip(entry[key][target])
        values.append(value)
        detail.append((condition, value))
    return float(np.mean(values)), detail


def score_team(id_report: dict, ood_report: dict, per_suite_clip: bool = True) -> dict:
    out = {"targets": {}, "total": 0.0}
    for target in TARGETS:
        id_r2, id_detail = _mean_over_conditions(id_report, "macro", target, per_suite_clip)
        ood_r2, _ = _mean_over_conditions(ood_report, "macro", target, per_suite_clip)
        ood_r, _ = _mean_over_conditions(ood_report, "macro_r", target, per_suite_clip)
        points = (WEIGHTS["id_r2"] * id_r2 + WEIGHTS["ood_r2"] * ood_r2
                  + WEIGHTS["ood_r"] * ood_r)
        out["targets"][target] = {
            "id_r2": id_r2, "ood_r2": ood_r2, "ood_r": ood_r,
            "points": points, "per_condition_id_r2": dict(id_detail)}
        out["total"] += points
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", dest="id_path", help="report scored on the three public suites")
    ap.add_argument("--ood", dest="ood_path", help="report scored on the held-out code")
    ap.add_argument("--results-dir", help="directory of <team>_id.json / <team>_ood.json")
    ap.add_argument("--out", help="write the markdown table here as well")
    args = ap.parse_args()

    pairs: dict[str, tuple[dict, dict]] = {}
    if args.results_dir:
        d = Path(args.results_dir)
        for p in sorted(d.glob("*_id.json")):
            team = p.name[: -len("_id.json")]
            ood = d / f"{team}_ood.json"
            if not ood.is_file():
                print(f"skipping {team}: no {ood.name}")
                continue
            pairs[team] = (json.loads(p.read_text()), json.loads(ood.read_text()))
    else:
        if not (args.id_path and args.ood_path):
            raise SystemExit("give --results-dir, or both --id and --ood")
        a = json.loads(Path(args.id_path).read_text())
        b = json.loads(Path(args.ood_path).read_text())
        pairs[a.get("team", "submission")] = (a, b)

    rows = []
    for team, (id_report, ood_report) in pairs.items():
        s = score_team(id_report, ood_report)
        alt = score_team(id_report, ood_report, per_suite_clip=False)
        rows.append((team, s, alt, id_report, ood_report))
    rows.sort(key=lambda r: -r[1]["total"])

    lines = ["| team | total | Om pts | Om ID R2 | Om OOD R2 | Om OOD r "
             "| s8 pts | s8 ID R2 | s8 OOD R2 | s8 OOD r |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for team, s, _alt, _i, _o in rows:
        c = [f"**{s['total']:.1f}**"]
        for t in TARGETS:
            d = s["targets"][t]
            c += [f"{d['points']:.1f}", f"{d['id_r2']:.3f}",
                  f"{d['ood_r2']:.3f}", f"{d['ood_r']:.3f}"]
        lines.append("| " + " | ".join([team] + c) + " |")
    table = "\n".join(lines)
    print(table)

    notes = []
    for team, s, alt, id_report, ood_report in rows:
        if abs(s["total"] - alt["total"]) > 0.05:
            notes.append(f"{team}: {s['total']:.1f} clipping each (condition, suite) cell, "
                         f"{alt['total']:.1f} clipping the suite macro instead")
        for label, r in (("ID", id_report), ("OOD", ood_report)):
            if r.get("catalogs_failed"):
                notes.append(f"{team} {label}: failed {r['catalogs_failed']} of "
                             f"{r['catalogs_attempted']} catalogs")
    if notes:
        print("\n" + "\n".join(f"- {n}" for n in notes))
    if args.out:
        Path(args.out).write_text(table + "\n" +
                                  "\n".join(f"- {n}" for n in notes) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
