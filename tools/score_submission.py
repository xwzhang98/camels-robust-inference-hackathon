#!/usr/bin/env python
"""Run one submission over every condition and write per-condition, per-target R^2.

Organizer-side. Shifted catalogs are materialized here with pinned seeds, scored, then
deleted -- which is why they are never stored, and why every team is scored on
byte-identical inputs.

The submission sees only a path to a catalog file. Those files come out of
``write_catalog``, which emits ``Group``, ``Subhalo`` and six neutral ``Header``
attributes and has no code path that writes ``Parameters`` or ``Config``. The ``clean``
condition goes through the same writer as an identity shift, so every test input is
scrubbed by construction and there is no separate sanitization step to forget.

**It expects the submission to misbehave**, because at a hackathon some of them will. A
call that hangs is interrupted, a call that raises is recorded and skipped, and either way
the run continues to the end and reports how many catalogs the submission actually
answered. A team whose model dies on eleven catalogs out of 2400 should be scored on the
other 2389, not lose everything -- and the organizer should not discover any of this by
watching a job sit at 0% for an hour.
"""
from __future__ import annotations

import argparse
import json
import signal
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from kaai_hackathon.catalog_io import read_catalog, write_catalog
from kaai_hackathon.conditions import (
    PUBLISHED_CONDITIONS, apply_condition, condition_seed,
)
from kaai_hackathon.scoring import (
    correlation_condition, macro_average, score_condition,
)
from kaai_hackathon.splits import load_labels
from kaai_hackathon.submission import load_submission, validate_prediction

TARGETS = ("Omega_m", "sigma_8")

GROUP_FIELDS = ["GroupPos", "GroupCM", "GroupVel", "GroupMass",
                "GroupNsubs", "GroupFirstSub"]
SUBHALO_FIELDS = ["SubhaloPos", "SubhaloCM", "SubhaloVel", "SubhaloSpin",
                  "SubhaloMass", "SubhaloMassType", "SubhaloGrNr", "SubhaloParent"]


class CatalogTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Interrupt the enclosed block if it runs longer than `seconds`.

    Checking the clock after a call returns cannot catch a call that never returns, and a
    submission that hangs is a normal hackathon outcome -- an infinite loop, a wait on
    something that is not there. SIGALRM interrupts anything executing Python bytecode,
    which is every hang we can realistically expect. A C extension that blocks signals for
    the whole duration would still escape; nothing short of a subprocess catches that, and
    a subprocess per catalog would mean reloading the model 2400 times.
    """
    if seconds <= 0:
        yield
        return

    def _fire(signum, frame):
        raise CatalogTimeout(f"exceeded {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="directory holding predict.py")
    parser.add_argument("--team", default=None, help="name for the report; default = dir name")
    parser.add_argument("--test-root", required=True, help="PRIVATE clean test catalogs")
    parser.add_argument("--params-root", required=True)
    parser.add_argument("--private-split", required=True,
                        help="JSON with {'private_test': {suite: [sim_id, ...]}}")
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--time-limit-s", type=float, default=120.0,
                        help="wall-clock cap per catalog; the call is interrupted, not "
                             "merely measured. 0 disables it")
    parser.add_argument("--max-failures", type=int, default=200,
                        help="give up on a submission after this many failed catalogs")
    parser.add_argument("--report-failures", type=int, default=10,
                        help="print at most this many failures; all are written to the JSON")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.private_split).read_text())["private_test"]
    specs = [s for s in PUBLISHED_CONDITIONS
             if args.conditions is None or s.name in args.conditions]
    team = args.team or Path(args.submission).resolve().name
    model, predict = load_submission(args.submission)
    started = time.time()

    # Read each source catalog ONCE and put it through every condition, rather than
    # re-reading it per condition. The read dominates -- an Astrid catalog is 141 MB, a
    # SIMBA one 60 MB -- so the obvious loop order costs eight times what it needs to:
    # 37 minutes per submission instead of 5.
    slowest = 0.0
    failures: list[dict] = []
    attempted = 0
    preds: dict = {(spec.name, suite): {t: [] for t in TARGETS}
                   for spec in specs for suite in args.suites}
    truths: dict = {(spec.name, suite): {t: [] for t in TARGETS}
                    for spec in specs for suite in args.suites}

    for suite in args.suites:
        labels = load_labels(args.params_root, suite)
        for sim_id in split.get(suite, []):
            source = Path(args.test_root) / suite / f"LH_{sim_id}" / "groups_090.hdf5"
            if not source.is_file():
                continue
            cat = read_catalog(source, group_fields=GROUP_FIELDS,
                               subhalo_fields=SUBHALO_FIELDS)
            for spec in specs:
                seed = condition_seed(args.base_seed, spec.name, suite, sim_id)
                shifted = apply_condition(cat, spec, seed)
                attempted += 1
                with tempfile.TemporaryDirectory() as scratch:
                    path = Path(scratch) / "groups_090.hdf5"
                    write_catalog(shifted, path)
                    call_started = time.time()
                    try:
                        with time_limit(args.time_limit_s):
                            out = validate_prediction(predict(model, str(path)))
                    except BaseException as exc:      # noqa: BLE001 -- record, do not die
                        failures.append({
                            "condition": spec.name, "suite": suite, "sim_id": int(sim_id),
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                            "traceback": traceback.format_exc()[-1500:],
                        })
                        if len(failures) <= args.report_failures:
                            print(f"  ! {spec.name}/{suite}/LH_{sim_id}: "
                                  f"{type(exc).__name__}: {exc}"[:160], flush=True)
                        if len(failures) > args.max_failures:
                            raise SystemExit(
                                f"{team}: {len(failures)} failed catalogs exceeds "
                                f"--max-failures {args.max_failures}; stopping.")
                        continue
                    elapsed = time.time() - call_started
                slowest = max(slowest, elapsed)
                for j, target in enumerate(TARGETS):
                    preds[(spec.name, suite)][target].append(out[target])
                    truths[(spec.name, suite)][target].append(labels[sim_id, j])
        print(f"{suite:14s} done  {attempted} catalogs so far, {len(failures)} failed"
              f"   [{time.time()-started:6.1f}s]", flush=True)

    results: dict = {}
    for spec in specs:
        per_suite: dict[str, dict[str, float]] = {}
        per_suite_r: dict[str, dict[str, float]] = {}
        for suite in args.suites:
            got, want = preds[(spec.name, suite)], truths[(spec.name, suite)]
            # An R^2 over fewer than two answers is not a score and must not look like one.
            if len(got[TARGETS[0]]) < 2:
                per_suite[suite] = {t: float("nan") for t in TARGETS}
                per_suite_r[suite] = {t: float("nan") for t in TARGETS}
                continue
            as_array = ({t: np.asarray(v) for t, v in got.items()},
                        {t: np.asarray(v) for t, v in want.items()})
            per_suite[suite] = score_condition(*as_array, TARGETS)
            per_suite_r[suite] = correlation_condition(*as_array, TARGETS)
        scored = {s: v for s, v in per_suite.items()
                  if all(np.isfinite(list(v.values())))}
        scored_r = {s_: v for s_, v in per_suite_r.items()
                    if all(np.isfinite(list(v.values())))}
        results[spec.name] = {
            "per_suite": per_suite,
            "per_suite_r": per_suite_r,
            "macro": (macro_average(scored, TARGETS) if scored
                      else {t: float("nan") for t in TARGETS}),
            "macro_r": (macro_average(scored_r, TARGETS) if scored_r
                        else {t: float("nan") for t in TARGETS}),
            "tier": spec.tier}
        macro = results[spec.name]["macro"]
        macro_r = results[spec.name]["macro_r"]
        print(f"{spec.name:14s} {spec.tier:5s} " +
              "  ".join(f"{t} R2={macro[t]:+.3f} r={macro_r[t]:+.3f}" for t in TARGETS),
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"team": team, "submission": str(Path(args.submission).resolve()),
         "base_seed": args.base_seed, "suites": args.suites,
         "slowest_catalog_s": slowest,
         "catalogs_attempted": attempted, "catalogs_failed": len(failures),
         "failures": failures, "scores": results}, indent=2))
    print(f"\nwrote {out}")
    print(f"slowest catalog {slowest:.2f}s   total {time.time()-started:.1f}s")
    if failures:
        kinds: dict[str, int] = {}
        for f in failures:
            kinds[f["error"].split(":")[0]] = kinds.get(f["error"].split(":")[0], 0) + 1
        print(f"FAILED on {len(failures)} of {attempted} catalogs: " +
              ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())))
    else:
        print(f"answered all {attempted} catalogs")


if __name__ == "__main__":
    main()
