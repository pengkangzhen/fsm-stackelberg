import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".agents/skills/figure-plotter/scripts"))
from figstyle import load_style, save_fig
load_style()

# Fig. 6 — Commitment-order effect vs. fault visibility (plant x order matrix).
# Annotated heatmap of FIRST-PROBE hit rates for three probe orders
# (causal / random / reverse) across five experimental settings.  Story: when
# the fault layer is visible to the ranking (ME, PD plants), order effects are
# extreme (causal ~1.0); for the DE plant (rank-invisible semantic fault) only
# random probing gets lucky hits.  All values are read from
# results/p4_grid/summary.json; nothing is hardcoded except the row mapping
# and the contract self-check.

import json

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results" / "p4_grid" / "summary.json"

# (row label, json group, json key) -- rows: ME block first, then PD, then DE.
ROWS = [
    ("ME plant (smoke)", "plants", "meforcezero"),
    ("ME instance H4·Ω10", "instances", "famH4O10_mefz"),
    ("ME instance H6·Ω5", "instances", "famH6O5_mefz"),
    ("PD plant", "plants", "pdbal"),
    ("DE plant", "plants", "deswap"),
]
ORDERS = ["causal", "random", "reverse"]

# Figure-contract self-check: (group, key) -> order -> (first_probe, n).
EXPECTED = {
    ("plants", "meforcezero"): {"causal": (30, 30), "random": (15, 30), "reverse": (0, 30)},
    ("instances", "famH4O10_mefz"): {"causal": (10, 10), "random": (4, 10), "reverse": (0, 10)},
    ("instances", "famH6O5_mefz"): {"causal": (10, 10), "random": (5, 10), "reverse": (0, 10)},
    ("plants", "pdbal"): {"causal": (9, 10), "random": (2, 10), "reverse": (10, 10)},
    ("plants", "deswap"): {"causal": (0, 10), "random": (4, 10), "reverse": (0, 10)},
}


def load_matrix():
    """Read first-probe hits and n from summary.json; verify against contract."""
    with open(DATA) as f:
        data = json.load(f)
    rates = np.zeros((len(ROWS), len(ORDERS)))
    counts = []
    for i, (_, group, key) in enumerate(ROWS):
        agg = data[group][key]["agg"]
        row_counts = []
        for j, order in enumerate(ORDERS):
            hits, n = int(agg[order]["first_probe"]), int(agg[order]["n"])
            exp = EXPECTED[(group, key)][order]
            if (hits, n) != exp:
                raise ValueError(
                    f"data drift: {key}/{order} = {hits}/{n}, "
                    f"expected {exp[0]}/{exp[1]}"
                )
            rates[i, j] = hits / n
            row_counts.append((hits, n))
        counts.append(row_counts)
    return rates, counts


def main():
    rates, counts = load_matrix()

    fig, ax = plt.subplots(figsize=(5.1, 2.7), constrained_layout=True)
    im = ax.imshow(rates, cmap="cividis", vmin=0, vmax=1, aspect="auto")

    # Thin white cell edges (standard minor-tick grid recipe over imshow).
    ax.set_xticks(np.arange(-0.5, len(ORDERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    # Slightly stronger white separators between setting groups (ME | PD | DE).
    for y in (2.5, 3.5):
        ax.axhline(y, color="white", linewidth=2.0)

    # Bold centered k/N annotation per cell; text color flips on luminance.
    for i in range(len(ROWS)):
        for j in range(len(ORDERS)):
            hits, n = counts[i][j]
            r, g, b, _ = im.cmap(im.norm(rates[i, j]))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j, i, f"{hits}/{n}", ha="center", va="center",
                fontsize=7.5, fontweight="bold",
                color="black" if lum > 0.55 else "white",
            )

    ax.set_xticks(range(len(ORDERS)))
    ax.set_xticklabels([o.capitalize() for o in ORDERS])
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([label for label, _, _ in ROWS])
    ax.set_xlabel("Probe order")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("First-probe hit rate")
    cbar.outline.set_visible(False)
    cbar.ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cbar.ax.tick_params(labelsize=7, length=2, width=0.6)

    save_fig(fig, "fig6_visibility_matrix", outdir="figures")

    print("Plotted first-probe hit matrix (hits/n, rate):")
    for (label, _, _), row in zip(ROWS, counts):
        cells = "  ".join(
            f"{o}={h}/{n} ({h / n:.2f})" for o, (h, n) in zip(ORDERS, row)
        )
        print(f"  {label:20s} {cells}")


if __name__ == "__main__":
    main()
