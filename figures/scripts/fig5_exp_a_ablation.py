"""Fig. 5 — Exp-A: commitment-order ablation (a) + alignment audit (b).

Panel (a): grouped bars over three metrics (first-probe hit, verified
attribution, SSR) for commitment orders causal / random / reverse, from
results/p4_grid/summary.json -> plants.meforcezero.agg.

Panel (b): seed-level alignment audit of the random arm: first-probe hit vs.
omega_1 aligned to the true root (model_expert), from the per-seed
run_manifest.json files. The two binary rows must coincide seed-for-seed.

Run:  uv run --no-sync python figures/scripts/fig5_exp_a_ablation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".agents/skills/figure-plotter/scripts"))
from figstyle import load_style, save_fig

import matplotlib.pyplot as plt

load_style()

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "p4_grid" / "summary.json"
MANIFEST_DIR = (
    ROOT / "results" / "mako" / "DeepSeek_deepseek-flash"
    / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5"
)
TRUE_ROOT = "model_expert"

# Okabe-Ito family colors for the three commitment orders
C_CAUSAL, C_RANDOM, C_REVERSE = "#0072B2", "#E69F00", "#CC79A7"
C_GRAY = "0.5"

ORDERS = ["causal", "random", "reverse"]
METRICS = [("first_probe", "First-probe"), ("verified", "Verified\nattribution"), ("ssr", "SSR")]


# ---------------------------------------------------------------- panel (a)
def load_agg():
    data = json.loads(SUMMARY.read_text())
    agg = data["plants"]["meforcezero"]["agg"]
    table = {}  # table[metric][order] = (k, n)
    for metric, _ in METRICS:
        table[metric] = {
            o: (agg[o][metric], agg[o]["n"]) for o in ORDERS
        }
    return table


def check_panel_a(table):
    expected = {
        "first_probe": {"causal": 20, "random": 10, "reverse": 0},
        "verified": {"causal": 9, "random": 9, "reverse": 5},
        "ssr": {"causal": 13, "random": 11, "reverse": 10},
    }
    ok = True
    for metric, by_order in expected.items():
        for o, k_exp in by_order.items():
            k, n = table[metric][o]
            mark = "OK " if k == k_exp else "MISMATCH"
            if k != k_exp:
                ok = False
            print(f"[panel a] {metric:>12s} {o:>7s}: {k}/{n}  (expected {k_exp}/{n}) [{mark}]")
    if not ok:
        print("WARNING [panel a]: DATA DISAGREES WITH EXPECTATION — plotted true values.")


def draw_panel_a(ax, table):
    xs = range(len(METRICS))
    offsets = {"causal": -0.26, "random": 0.0, "reverse": 0.26}
    colors = {"causal": C_CAUSAL, "random": C_RANDOM, "reverse": C_REVERSE}
    width = 0.24
    for o in ORDERS:
        heights, labels = [], []
        for metric, _ in METRICS:
            k, n = table[metric][o]
            heights.append(k / n)
            labels.append(f"{k}/{n}")
        xpos = [x + offsets[o] for x in xs]
        ax.bar(xpos, heights, width=width, color=colors[o],
               edgecolor="black", linewidth=0.4, label=o)
        for x, h, lab in zip(xpos, heights, labels):
            ax.text(x, h + 0.025, lab, ha="center", va="bottom", fontsize=6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for _, label in METRICS])
    ax.set_xlim(-0.55, len(METRICS) - 0.45)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Rate")
    ax.legend(
        ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0,
        handlelength=1.1, handletextpad=0.4, columnspacing=0.9,
    )
    ax.annotate("(a)", xy=(-0.20, 1.08), xycoords="axes fraction",
                fontsize=9, fontweight="bold", annotation_clip=False)


# ---------------------------------------------------------------- panel (b)
def load_audit():
    """Read random-arm manifests; return (records, skipped seeds).

    Scan seed ids 1..23: the n=20 campaign in summary.json uses every seed in
    1..23 that produced a run_manifest.json (seeds 7/12/14 crashed before the
    manifest was written and have no record).
    """
    records, skipped = [], []
    for s in range(1, 24):
        p = MANIFEST_DIR / f"ds41_ea_random_s{s}" / "run_manifest.json"
        if not p.exists():
            skipped.append(s)
            continue
        m = json.loads(p.read_text())
        ins = m["inspection"]
        omega = ins.get("committed_omega") or []
        records.append({
            "seed": s,
            "hit": bool(ins.get("first_probe_hit")),
            "aligned": bool(omega) and omega[0] == TRUE_ROOT,
        })
    return records, skipped


def check_panel_b(records):
    tt = sum(r["aligned"] and r["hit"] for r in records)
    tf = sum(r["aligned"] and not r["hit"] for r in records)
    ft = sum(not r["aligned"] and r["hit"] for r in records)
    ff = sum(not r["aligned"] and not r["hit"] for r in records)
    print(f"[panel b] crosstab (rows=omega_1 aligned, cols=first-probe hit):")
    print(f"          aligned  & hit = {tt}   aligned & miss = {tf}")
    print(f"          misalign & hit = {ft}   misalign & miss = {ff}")
    if (tt, tf, ft, ff) == (10, 0, 0, 10):
        print("[panel b] diagonal crosstab 10/0/0/10 as expected (rows coincide).")
    else:
        print("WARNING [panel b]: DATA DISAGREES WITH EXPECTED 10/0/0/10 — plotted true values.")
    return tt, tf, ft, ff


def draw_panel_b(ax, records):
    xs = list(range(1, len(records) + 1))
    seeds = [r["seed"] for r in records]
    hits = [r["hit"] for r in records]
    aligned = [r["aligned"] for r in records]

    # faint per-seed connectors + row guides to pair the two rows
    ax.vlines(xs, 0, 1, color="0.87", linewidth=0.5, zorder=1)
    for y in (0, 1):
        ax.axhline(y, color="0.90", linewidth=0.5, zorder=0.5)

    x_hit = [x for x, h in zip(xs, hits) if h]
    x_miss = [x for x, h in zip(xs, hits) if not h]
    ax.scatter(x_hit, [1] * len(x_hit), marker="o", s=18, c=C_CAUSAL, zorder=3)
    ax.scatter(x_miss, [1] * len(x_miss), marker="o", s=18, facecolors="none",
               edgecolors=C_GRAY, linewidths=0.7, zorder=3)

    x_al = [x for x, a in zip(xs, aligned) if a]
    x_mis = [x for x, a in zip(xs, aligned) if not a]
    ax.scatter(x_al, [0] * len(x_al), marker="s", s=16, c=C_RANDOM, zorder=3)
    ax.scatter(x_mis, [0] * len(x_mis), marker="s", s=16, facecolors="none",
               edgecolors=C_GRAY, linewidths=0.7, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(seeds)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlim(0.4, len(records) + 0.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["first-probe\nhit", r"$\omega_1 =$" + "\ntrue root"])
    ax.set_xlabel("Random-arm seed")
    ax.annotate("(b)", xy=(-0.26, 1.16), xycoords="axes fraction",
                fontsize=9, fontweight="bold", annotation_clip=False)


def main():
    table = load_agg()
    check_panel_a(table)

    records, skipped = load_audit()
    print(f"[panel b] {len(records)} manifests read; seeds without run_manifest.json "
          f"(skipped): {skipped}")
    check_panel_b(records)

    fig = plt.figure(figsize=(5.5, 2.3))
    gs = fig.add_gridspec(
        1, 2, left=0.100, right=0.985, bottom=0.185, top=0.845, wspace=0.31
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_a(ax_a, table)
    draw_panel_b(ax_b, records)

    save_fig(fig, "fig5_exp_a_ablation", outdir="figures")


if __name__ == "__main__":
    main()
