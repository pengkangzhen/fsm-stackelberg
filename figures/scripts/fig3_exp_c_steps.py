"""Fig. 3 (Exp-C): verified attribution rate vs. probe budget K (step curves).

Each curve is a step function (steps-post): its jump position equals the rank
of the true root-cause layer in the committed probe order omega. Sequential
stays at 0; sb_reverse jumps only at K>=2; K=2->3 gains are zero for every arm.
Direct method labels are placed at line ends in a right-hand gutter (staggered
to avoid collisions), each annotated with its final verified count k/N in
small gray text.

Data source: results/exp_c/summary.json (single source of truth).
Run: cd <repo root> && uv run --no-sync python figures/scripts/fig3_exp_c_steps.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".agents/skills/figure-plotter/scripts"))
from figstyle import load_style, save_fig

import matplotlib.pyplot as plt

load_style()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "exp_c" / "summary.json"

K = (1, 2, 3)

# Cross-figure family standard: method -> (color, marker, linestyle, lw, ms).
ARMS = {
    "sb_causal":   ("#0072B2", "o", "-",  1.8, 4.6),
    "sb_random":   ("#E69F00", "s", "--", 1.3, 3.4),
    "sb_reverse":  ("#CC79A7", "^", ":",  1.3, 3.6),
    "adversarial": ("#D55E00", "D", "-.", 1.3, 3.2),
    "sequential":  ("#000000", "v", "-",  1.0, 3.4),
    "debate":      ("#009E73", "P", "--", 1.3, 3.4),
    "reflexion":   ("#56B4E9", "X", ":",  1.3, 3.8),
}
ZORDER = {  # sb_causal (proposed) always on top of coincident points
    "sb_causal": 6.0, "reflexion": 5.0, "adversarial": 4.6, "debate": 4.5,
    "sb_random": 4.4, "sb_reverse": 4.3, "sequential": 4.2,
}

# Layout (89 mm single column; right portion of the axes is the label gutter).
FIG_W, FIG_H = 3.5, 2.6
X_LO, X_HI = 0.72, 3.28          # curve region (grid + bottom spine end here)
X_END = 4.60                     # axes right edge in data units
LBL_X = 3.36                     # left edge of direct labels
LEADER_X0, LEADER_X1 = 3.05, 3.33
GRID_FRAC = (X_HI - X_LO) / (X_END - X_LO)
Y_LO, Y_HI = -0.02, 1.03
LABEL_MIN_Y = 0.02               # keep bottom label off the axis line
NAME_FS, COUNT_FS = 6.5, 6.0     # >= 6 pt at print size
GAP = 8.5 / ((0.965 - 0.16) * FIG_H * 72.0)  # label row pitch in y-data units


def stagger_labels(desired, gap, sweeps=200):
    """Separation-preserving label positions closest to the true line ends.

    Input/output are ordered top-to-bottom (descending final rate). Feasible
    greedy init, then coordinate descent pulls each row toward its desired
    value within its neighbors' [y+gap, y-gap] window.
    """
    n = len(desired)
    y = [0.0] * n
    y[0] = min(desired[0], 1.0)
    for i in range(1, n):
        y[i] = min(desired[i], y[i - 1] - gap, 1.0)
    y[n - 1] = max(y[n - 1], LABEL_MIN_Y)
    for _ in range(sweeps):
        stable = True
        for i in range(n):
            lo = y[i + 1] + gap if i < n - 1 else LABEL_MIN_Y
            hi = y[i - 1] - gap if i > 0 else 1.0
            yi = min(max(desired[i], lo), hi)
            if abs(yi - y[i]) > 1e-9:
                y[i] = yi
                stable = False
        if stable:
            break
    return y


def main():
    data = json.loads(DATA.read_text())["methods"]

    print(f"{'method':<12} {'n':>3} {'v@1':>4} {'v@2':>4} {'v@3':>4}   "
          "rates @K=1/2/3")
    for m, d in data.items():
        n = d["n"]
        v = (d["verified_at_1"], d["verified_at_2"], d["verified_at_3"])
        if not (v[0] <= v[1] <= v[2]):
            print(f"  [warn] {m}: non-monotone counts {v}")
        if v[1] != v[2]:
            print(f"  [warn] {m}: K=2->3 gain nonzero ({v[1]} -> {v[2]})")
        r = [vi / n for vi in v]
        print(f"{m:<12} {n:>3} {v[0]:>4} {v[1]:>4} {v[2]:>4}   "
              f"{r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.135, right=0.99, top=0.965, bottom=0.16)

    # Light y-grid confined to the curve region (not under the labels).
    for yv in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.axhline(yv, xmin=0.0, xmax=GRID_FRAC, color="0.88", lw=0.5,
                   zorder=0.5)

    for m, (color, marker, ls, lw, ms) in ARMS.items():
        d = data[m]
        n = d["n"]
        rate = [d[f"verified_at_{k}"] / n for k in K]
        ax.plot(K, rate, drawstyle="steps-post", color=color, ls=ls, lw=lw,
                marker=marker, ms=ms, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=ZORDER[m], clip_on=False)

    # Direct labels: order top-to-bottom by final rate; ties keep sb_causal on top.
    order = sorted(data, key=lambda m: (-data[m]["verified_at_3"] / data[m]["n"],
                                        m != "sb_causal", m))
    finals = [data[m]["verified_at_3"] / data[m]["n"] for m in order]
    label_y = stagger_labels(finals, GAP)

    name_texts = {}
    for m, yl in zip(order, label_y):
        color = ARMS[m][0]
        v3, n = data[m]["verified_at_3"], data[m]["n"]
        yf = data[m]["verified_at_3"] / n
        ax.plot([LEADER_X0, LEADER_X1], [yf, yl], color=color, lw=0.55,
                alpha=0.85, zorder=4.0)
        name_texts[m] = ax.text(LBL_X, yl, m, color=color, fontsize=NAME_FS,
                                fontweight="bold", ha="left", va="center",
                                zorder=6.0)

    # Gray "k/N" suffixes, placed after each measured name width.
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        renderer = None
    inv = ax.transData.inverted()
    for m, yl in zip(order, label_y):
        v3, n = data[m]["verified_at_3"], data[m]["n"]
        bbox = name_texts[m].get_window_extent(renderer=renderer)
        x_after = inv.transform((bbox.x1, bbox.y0))[0] + 0.03
        ax.text(x_after, yl, f"{v3}/{n}", color="0.42", fontsize=COUNT_FS,
                ha="left", va="center", zorder=6.0)

    ax.set_xlim(X_LO, X_END)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks(K)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f"{v:g}" for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)])
    ax.spines["bottom"].set_bounds(X_LO, X_HI)
    ax.set_xlabel("Probe budget $K$")
    ax.set_ylabel("Verified attribution rate")
    ax.tick_params(axis="both", which="both", top=False, right=False)

    save_fig(fig, "fig3_exp_c_steps", outdir=str(ROOT / "figures"))


if __name__ == "__main__":
    main()
