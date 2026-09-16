import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / ".agents/skills/figure-plotter/scripts"))
from figstyle import load_style, save_fig
load_style()

# Fig. 4 — Exp-B Pareto view: verified attribution rate vs. mean loop-token
# cost.  Story: Stackelberg (sb_causal) sits on the efficient frontier at the
# LOWEST token cost; reflexion's higher point estimate overlaps SB's Wilson
# 95% CI (non-significant difference), and every non-SB baseline pays 0.9-3.7x
# the token budget.  All values are read from results/exp_b_external/
# summary.json (key "agg"); nothing is hardcoded except the display mapping
# and the contract self-check.  Wilson intervals use statistics.NormalDist
# (scipy is intentionally not a dependency).

import json
import math
from statistics import NormalDist

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results" / "exp_b_external" / "summary.json"

Z = NormalDist().inv_cdf(0.975)  # 1.9599639..., Wilson 95%

# key -> (display name, color, marker, markersize, is_hero)
# Cross-figure family standard: Okabe-Ito colors + fixed marker glyphs.
STYLE = {
    "sb_causal":   ("Stackelberg (ours)", "#0072B2", "o", 8.5, True),
    "sequential":  ("Sequential",         "#000000", "v", 6.0, False),
    "adversarial": ("Adversarial",        "#D55E00", "D", 6.0, False),
    "reflexion":   ("Reflexion",          "#56B4E9", "X", 6.5, False),
    "debate":      ("Debate",             "#009E73", "P", 6.5, False),
}

# Figure-contract self-check: key -> (verified, n, mean_loop_tokens).
EXPECTED = {
    "sb_causal":   (14, 30, 63768),
    "adversarial": (10, 30, 136010),
    "sequential":  (0, 30, 51301),
    "debate":      (7, 30, 257924),
    "reflexion":   (15, 30, 184469),
}


def wilson_ci(k, n):
    """Wilson score interval for k successes out of n (no continuity corr.)."""
    p = k / n
    z2 = Z * Z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (Z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return center - half, center + half


def load_aggregates():
    """Read per-method aggregates; verify against the figure contract."""
    with open(DATA) as f:
        agg = json.load(f)["agg"]
    rows = {}
    for key, (verified, n, tokens) in EXPECTED.items():
        a = agg[key]
        got = (int(a["verified"]), int(a["n"]), int(a["mean_loop_tokens"]))
        if got != (verified, n, tokens):
            raise ValueError(f"data drift: {key} = {got}, expected {(verified, n, tokens)}")
        rows[key] = {
            "k": verified,
            "n": n,
            "rate": verified / n,
            "ci_lo": None,
            "ci_hi": None,
            "x": tokens / 1e5,
            "tokens": tokens,
        }
        rows[key]["ci_lo"], rows[key]["ci_hi"] = wilson_ci(verified, n)
    return rows


def main():
    rows = load_aggregates()
    sb = rows["sb_causal"]

    fig, ax = plt.subplots(figsize=(3.5, 2.65), constrained_layout=True)

    for key, r in rows.items():
        name, color, marker, ms, hero = STYLE[key]
        x, y = r["x"], r["rate"]
        # Vertical Wilson 95% CI bar, then marker on top (hero marker last/above).
        ax.errorbar(
            x, y,
            yerr=[[y - r["ci_lo"]], [r["ci_hi"] - y]],
            fmt="none", ecolor=color, elinewidth=0.9,
            capsize=2.2, capthick=0.9, zorder=2,
        )
        ax.plot(
            x, y, linestyle="none", marker=marker,
            markersize=ms, markerfacecolor=color, markeredgecolor=color,
            markeredgewidth=0.8, zorder=5 if hero else 4,
        )

        # --- direct labels (spatially fixed categories, no legend) ---------
        if key == "sequential":
            # Name sits right of the bar at the axis (rate = 0); cost multiple
            # stacks above it at the CI cap to stay clear of SB's lower cap.
            ax.text(x + 0.055, 0.006, name, ha="left", va="bottom",
                    fontsize=7.5, color=color, zorder=6)
            ax.text(x + 0.055, r["ci_hi"] + 0.008,
                    f"{r['tokens'] / sb['tokens']:.1f}\u00d7",
                    ha="left", va="bottom", fontsize=6.5, color="0.35", zorder=6)
        else:
            ax.text(x, r["ci_hi"] + 0.042, name, ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    fontweight="bold" if hero else "normal", zorder=6)
            if not hero:
                # Cost multiple vs. SB, at the marker level right of the bar.
                ax.text(x + 0.055, y, f"{r['tokens'] / sb['tokens']:.1f}\u00d7",
                        ha="left", va="center", fontsize=6.5, color="0.35",
                        zorder=6)

    # Better-direction cue, bottom-left: unobtrusive diagonal arrow + phrase.
    ax.annotate(
        "better: lower cost / higher accuracy",
        xy=(0.30, 0.160), xytext=(0.42, -0.060),
        ha="left", va="center", fontsize=6.5, color="0.35",
        style="italic", zorder=3,
        arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.7,
                        mutation_scale=7),
    )

    ax.set_xlim(0.24, 2.82)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylim(-0.095, 1.03)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Mean loop tokens (10$^{5}$)")
    ax.set_ylabel("Verified attribution rate")

    save_fig(fig, "fig4_exp_b_pareto", outdir="figures")

    print("Plotted Exp-B Pareto (k/n, rate, Wilson 95% CI, tokens, cost multiple):")
    for key, r in rows.items():
        print(
            f"  {STYLE[key][0]:20s} {r['k']}/{r['n']}  rate={r['rate']:.2f}  "
            f"CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  "
            f"tokens={r['tokens']}  "
            f"mult={r['tokens'] / sb['tokens']:.2f}x"
        )
    print(
        f"  overlap check: reflexion CI lo {rows['reflexion']['ci_lo']:.4f} "
        f"< SB CI hi {sb['ci_hi']:.4f} "
        f"({rows['reflexion']['ci_lo'] < sb['ci_hi']})"
    )


if __name__ == "__main__":
    main()
