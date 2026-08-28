#!/usr/bin/env python
"""Render the baseline comparison report from the JSON files the evaluators write.

Organizer-side. Takes any number of ``label=path`` pairs and produces one markdown
document: a table per arm, one row per condition, one column pair per baseline. Written as
a script rather than by hand because the noise levels are still being calibrated and this
gets regenerated every time they move.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGETS = ("Omega_m", "sigma_8")
ARM_TITLES = {
    "all_three": "Trained on all three simulation codes",
    "holdout_IllustrisTNG": "Trained on SIMBA + Astrid, tested on IllustrisTNG",
    "holdout_SIMBA": "Trained on IllustrisTNG + Astrid, tested on SIMBA",
    "holdout_Astrid": "Trained on IllustrisTNG + SIMBA, tested on Astrid",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", metavar="LABEL=PATH")
    parser.add_argument("--title", default="Baselines under every shift condition")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    loaded = {}
    for item in args.reports:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        loaded[label] = json.loads(Path(path).read_text())

    labels = list(loaded)
    lines = [f"# {args.title}", ""]
    for label, report in loaded.items():
        bits = [f"`{label}`"]
        if "model" in report:
            bits.append(report["model"])
        bits.append(f"{report['n_train_per_suite']} train / "
                    f"{report['n_test_per_suite']} test per suite")
        for key in ("epochs", "hidden", "layers", "r_link", "centroid"):
            if key in report:
                bits.append(f"{key}={report[key]}")
        lines.append("- " + ", ".join(bits))
    lines.append("")

    arms = []
    for report in loaded.values():
        for arm in report["arms"]:
            if arm not in arms:
                arms.append(arm)

    for arm in arms:
        lines += [f"## {ARM_TITLES.get(arm, arm)}", ""]
        header = ["condition", "tier"]
        for label in labels:
            header += [f"{label} Om", f"{label} s8"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        conditions: list[str] = []
        for report in loaded.values():
            for name in report["arms"].get(arm, {}).get("scores", {}):
                if name not in conditions:
                    conditions.append(name)

        for name in conditions:
            tier = ""
            row = []
            for label in labels:
                entry = loaded[label]["arms"].get(arm, {}).get("scores", {}).get(name)
                if entry is None:
                    row += ["—", "—"]
                    continue
                tier = entry.get("tier", tier)
                row += [f"{entry['macro'][t]:+.3f}" for t in TARGETS]
            lines.append("| " + " | ".join([f"`{name}`", tier, *row]) + " |")
        lines.append("")

        for label in labels:
            entry = loaded[label]["arms"].get(arm)
            if entry and "val_loss" in entry:
                lines.append(f"- `{label}` validation loss: {entry['val_loss']:.4f}")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
