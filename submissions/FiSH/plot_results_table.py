"""Renders the final-model val results as a table image (results_table.png) -- the same
numbers as README.md's markdown table, for a submission-ready visual. The macro-average row
is labeled "test" here (this run has no held-out suite, so the 20% val split is the
closest thing to a held-out evaluation set)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = [
    ("Astrid",       "+0.993", "+0.997", "+0.883", "+0.940"),
    ("IllustrisTNG", "+0.992", "+0.996", "+0.857", "+0.927"),
    ("SIMBA",        "+0.992", "+0.997", "+0.905", "+0.952"),
    ("test",         "+0.992", "+0.997", "+0.882", "+0.940"),   # was "macro"
]
COLS = ["suite", "Omega_m  R2", "Omega_m  Pearson r", "sigma_8  R2", "sigma_8  Pearson r"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    col_widths = [0.14, 0.16, 0.2, 0.16, 0.2]
    table = ax.table(cellText=ROWS, colLabels=COLS, loc="center", cellLoc="center",
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    n_cols = len(COLS)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", weight="bold")
        elif row == len(ROWS):                      # the "test" row
            cell.set_facecolor("#ffe9b3")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")

    ax.set_title("Final model (all_together + GrNr, 3-suite, augmented) -- val results",
                 fontsize=12, pad=14)
    fig.tight_layout()
    fig.savefig("results_table.png", dpi=200, bbox_inches="tight")
    print("wrote results_table.png")


if __name__ == "__main__":
    main()
