"""Recompute the Exp-B external/unified summary from run manifests.

The authoritative data source is the per-cell run manifests (snapshot-mode
cells over the shared me_force n=20 blackboards); this script re-derives
every aggregate (first-probe, verified attribution, SSR@K, mean loop
tokens, paired exact McNemar) under the current GAP_THRESHOLD and writes
results/exp_b_external/summary.json.

Strict success and verified attribution are re-derived from the measured
gap: stored manifest flags reflect the threshold live at run time, so a
tightened threshold reclassifies cells offline rather than silently
trusting stale flags. Use --gap-threshold to reproduce aggregates under a
historical threshold (e.g. 1e-2, the pre-tightening value).

Usage:
  uv run --no-sync python scripts/summarize_exp_b_external.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from fsm_stackelberg import constants

REPO = Path(__file__).resolve().parent.parent
RESULTS = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
           / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5")
SNAPSHOTS = REPO / "results" / "snapshots"
OUT = REPO / "results" / "exp_b_external"

# Method key -> run-directory prefix under RESULTS (paired per seed).
METHODS = {
    "sb_causal": "ds41_ea_causal",
    "adversarial": "ds41_eb_adversarial",
    "sequential": "ds41_eb_sequential",
    "debate": "ds41_ebext_debate",
    "reflexion": "ds41_ebext_reflexion",
}

PAIRS = [
    ("sb_causal", "adversarial"),
    ("sb_causal", "sequential"),
    ("sb_causal", "debate"),
    ("sb_causal", "reflexion"),
    ("adversarial", "debate"),
    ("adversarial", "reflexion"),
]
METRICS = ("first_probe", "verified")


def usable_seeds() -> list[int]:
    return sorted(
        int(p.name.rsplit("_s", 1)[1])
        for p in SNAPSHOTS.glob("me_force_zero_sea_s*")
        if (p / "snapshot_state.json").exists())


def load_cell(prefix: str, seed: int, gap_threshold: float) -> dict | None:
    mf = RESULTS / f"{prefix}_s{seed}" / "run_manifest.json"
    if not mf.exists():
        return None
    m = json.loads(mf.read_text())
    insp = m.get("inspection") or {}
    ep = insp.get("episode_payoff") or {}
    outcome = m.get("outcome") or {}
    cost = m.get("cost") or {}
    cfg = m.get("config") or {}
    gap = outcome.get("gap_percent")
    within = (gap <= gap_threshold * 100) if gap is not None else bool(
        outcome.get("practical_optimal"))
    practical = bool(outcome.get("solver_optimal")) and within
    forward = cfg.get("snapshot_forward_tokens") or 0
    total = cost.get("total_tokens") or 0
    return {
        "seed": seed,
        "first_probe": bool(insp.get("first_probe_hit")),
        "verified": bool(ep.get("verified_attribution_hit")) and practical,
        "ssr": practical,
        "loop_tokens": total - forward,
    }


def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict:
    """Two-sided exact McNemar: 2 * P(X <= min(b,c)), X~Bin(b+c, .5)."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * 0.5 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap-threshold", type=float, default=None,
                    help="fraction; default = package GAP_THRESHOLD "
                         f"({constants.GAP_THRESHOLD}); 1e-2 reproduces the "
                         "pre-tightening aggregates")
    args = ap.parse_args()
    thr = args.gap_threshold if args.gap_threshold is not None else constants.GAP_THRESHOLD

    seeds = usable_seeds()
    cells: dict[str, dict[int, dict]] = {}
    for method, prefix in METHODS.items():
        per_seed = {}
        for s in seeds:
            cell = load_cell(prefix, s, thr)
            if cell is not None:
                per_seed[s] = cell
        cells[method] = per_seed

    agg = {}
    for method, per_seed in cells.items():
        rows = list(per_seed.values())
        tokens = [r["loop_tokens"] for r in rows]
        agg[method] = {
            "n": len(rows),
            "first_probe": sum(r["first_probe"] for r in rows),
            "verified": sum(r["verified"] for r in rows),
            "ssr": sum(r["ssr"] for r in rows),
            "mean_loop_tokens": round(sum(tokens) / len(tokens)) if tokens else 0,
        }

    mcnemar = {}
    for a, b in PAIRS:
        shared = sorted(set(cells[a]) & set(cells[b]))
        for metric in METRICS:
            pairs = [(cells[a][s][metric], cells[b][s][metric]) for s in shared]
            a_key = {"sb_causal": "sb", "adversarial": "adv"}.get(a, a)
            mcnemar[f"{a_key}_vs_{b}:{metric}"] = mcnemar_exact(pairs)

    payload = {
        "seeds": seeds,
        "gap_threshold": thr,
        "agg": agg,
        "mcnemar": mcnemar,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"gap_threshold = {thr}")
    hdr = f"{'method':12} {'n':>2} {'fp':>3} {'ver':>3} {'ssr':>3} {'loop_tok':>9}"
    print(hdr)
    for m, a in agg.items():
        print(f"{m:12} {a['n']:>2} {a['first_probe']:>3} {a['verified']:>3} "
              f"{a['ssr']:>3} {a['mean_loop_tokens']:>9,}")
    for k, v in mcnemar.items():
        print(f"  {k:36} b={v['b']} c={v['c']} p={v['p']:.4f}")
    print("\nwrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
