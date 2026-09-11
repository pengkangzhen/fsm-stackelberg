"""Aggregate the Phase-4 multi-plant grid (snapshot mode) into summary tables.

Reads per-cell run manifests for the p4_<short>_ea_<order>_s<seed> arms of
each Phase-4 plant (usable seeds from results/p4_grid/<short>/state.json)
and, for the cross-plant block, the n=20 me_force_zero_sea campaign cells
(ds41_ea_*; usable seeds = seeds whose snapshot dir exists).

Per plant it emits the 3xN Exp-A table (first-probe hit, verified
attribution, SSR@K, mean loop tokens = total - snapshot_forward_tokens),
paired exact McNemar tests (two-sided binomial on discordant pairs, the
same procedure as the n=20 campaign summary), the random-alignment audit
(first-probe hit <=> shuffle placed a* first; verified hits reached from
un-aligned shuffles = multi-round search path), and a cross-plant
comparison (per-plant rates + discordant-pair-pooled McNemar, disclosed as
such).

Usage:
  uv run --no-sync python scripts/summarize_p4_grid.py   # writes
  results/p4_grid/summary.md + summary.json
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SMOKE = "smoke_H4_Omega5"
SNAPSHOTS = REPO / "results" / "snapshots"
P4_ROOT = REPO / "results" / "p4_grid"
FAMILY_ROOT = REPO / "results" / "instance_family"
ORDERS = ("causal", "random", "reverse")


def results_root(prob_name: str) -> Path:
    return (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
            / "prob_tslp_ecr_demand_k3" / prob_name)

PLANTS = {
    "deswap": {"plant": "de_swap_demand_supply_source",
               "astar": "data_engineer",
               "label": "de_swap_demand_supply_source (a*=data_engineer)",
               "prefix": "p4_deswap_ea"},
    "pdbal": {"plant": "pd_comment_out_balance",
              "astar": "python_developer",
              "label": "pd_comment_out_balance (a*=python_developer)",
              "prefix": "p4_pdbal_ea"},
    "meforcezero": {"plant": "me_force_zero_sea",
                    "astar": "model_expert",
                    "label": "me_force_zero_sea (a*=model_expert; n=20 "
                             "campaign, same engine/mode)",
                    "prefix": "ds41_ea",
                    "campaign": "exp_ab_ds41_snapshot"},
}


def collect_family() -> list[dict]:
    """Instance-family campaigns (p5): one block per state dir."""
    out = []
    for state_f in sorted(FAMILY_ROOT.glob("*/state.json")):
        st = json.loads(state_f.read_text())
        spec = {"plant": st["plant"], "astar": "model_expert",
                "label": f"{st['prob_name']} — {st['plant']} (a*=ME)",
                "prefix": f"p5_{st['short']}_ea",
                "prob_name": st["prob_name"]}
        seeds = sorted(int(s[1:]) for s in st["usable_seeds"])
        if not seeds:
            continue
        cells = {o: {} for o in ORDERS}
        for order in ORDERS:
            for s in seeds:
                cell = load_cell(spec["prob_name"],
                                 f"{spec['prefix']}_{order}_s{s}")
                if cell is not None:
                    cells[order][s] = cell
        out.append({"short": st["short"], "spec": spec, "seeds": seeds,
                    "failed_freezes": sorted(st["failed_freezes"],
                                             key=lambda x: int(x[1:])),
                    "cells": cells})
    return out


def load_cell(prob_name: str, suffix: str) -> dict | None:
    mf = results_root(prob_name) / suffix / "run_manifest.json"
    if not mf.exists():
        return None
    m = json.loads(mf.read_text())
    insp = m.get("inspection") or {}
    ep = insp.get("episode_payoff") or {}
    cost = m.get("cost") or {}
    cfg = m.get("config") or {}
    forward = cfg.get("snapshot_forward_tokens") or 0
    total = cost.get("total_tokens") or 0
    return {
        "suffix": suffix,
        "first_probe_hit": bool(insp.get("first_probe_hit")),
        "verified": bool(ep.get("verified_attribution_hit")),
        "ssr": bool((m.get("outcome") or {}).get("practical_optimal")),
        "omega": insp.get("committed_omega") or [],
        "first_probe_layer": insp.get("first_probe_layer"),
        "probe_seed": cfg.get("probe_seed"),
        "total_tokens": total,
        "loop_tokens": total - forward,
        "probe_rounds": cost.get("probe_rounds"),
    }


def usable_seeds(short: str) -> tuple[list[int], list[str]] | None:
    """(usable attempt seeds, failed freeze seeds) for a Phase-4 plant."""
    state_f = P4_ROOT / short / "state.json"
    if not state_f.exists():
        return None
    st = json.loads(state_f.read_text())
    return (sorted(int(s[1:]) for s in st["usable_seeds"]),
            sorted(st["failed_freezes"], key=lambda x: int(x[1:])))


def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict:
    """Two-sided exact McNemar: 2 * P(X <= min(b,c)), X~Bin(b+c, .5)."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * 0.5 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def fmt_p(p: float) -> str:
    return f"p = {p:.4f}" if p >= 0.0001 else f"p < 0.0001"


def collect(short: str) -> dict | None:
    """cells[order][seed] -> cell dict, per-plant meta."""
    spec = PLANTS[short]
    if "campaign" in spec:
        seeds = sorted(
            int(Path(p).name.rsplit("_s", 1)[1])
            for p in glob.glob(str(SNAPSHOTS / f"{spec['plant']}_s*"))
            if (Path(p) / "snapshot_state.json").exists())
        failed = []
    else:
        got = usable_seeds(short)
        if got is None:
            return None
        seeds, failed = got
    cells = {o: {} for o in ORDERS}
    prob = spec.get("prob_name", SMOKE)
    for order in ORDERS:
        for s in seeds:
            cell = load_cell(prob, f"{spec['prefix']}_{order}_s{s}")
            if cell is not None:
                cells[order][s] = cell
    return {"short": short, "spec": spec, "seeds": seeds,
            "failed_freezes": failed, "cells": cells}


def plant_table(r: dict) -> tuple[dict, list[str]]:
    """Per-order aggregates + per-seed pair lists for McNemar."""
    seeds = r["seeds"]
    agg, notes = {}, []
    for order in ORDERS:
        cs = [r["cells"][order][s] for s in seeds
              if s in r["cells"][order]]
        n = len(cs)
        if n == 0:
            continue
        agg[order] = {
            "n": n,
            "first_probe": sum(c["first_probe_hit"] for c in cs),
            "verified": sum(c["verified"] for c in cs),
            "ssr": sum(c["ssr"] for c in cs),
            "mean_loop_tokens": round(sum(c["loop_tokens"] for c in cs) / n),
        }
    # random-alignment audit
    rnd = r["cells"].get("random", {})
    if rnd:
        astar = r["spec"]["astar"]
        aligned = {s: (c["omega"][0] == astar if c["omega"] else None)
                   for s, c in rnd.items()}
        hits = {s: c["first_probe_hit"] for s, c in rnd.items()}
        consistent = all(bool(aligned[s]) == bool(hits[s]) for s in rnd)
        verified = {s: c["verified"] for s, c in rnd.items()}
        multi_round = [s for s in rnd if verified[s] and not aligned[s]]
        aligned_verified = [s for s in rnd if verified[s] and aligned[s]]
        notes.append({
            "random_alignment": {
                "consistent_with_shuffle": consistent,
                "n_aligned": sum(1 for v in aligned.values() if v),
                "verified_from_aligned": aligned_verified,
                "verified_from_unaligned_multiround": multi_round,
            }})
    pairs_fp = {f"{a}|{b}": mcnemar_exact([
        (r["cells"][a][s]["first_probe_hit"], r["cells"][b][s]["first_probe_hit"])
        for s in seeds if s in r["cells"][a] and s in r["cells"][b]])
        for a, b in (("causal", "reverse"), ("causal", "random"))}
    pairs_va = {f"{a}|{b}": mcnemar_exact([
        (r["cells"][a][s]["verified"], r["cells"][b][s]["verified"])
        for s in seeds if s in r["cells"][a] and s in r["cells"][b]])
        for a, b in (("causal", "reverse"), ("causal", "random"))}
    return {"agg": agg, "mcnemar_first_probe": pairs_fp,
            "mcnemar_verified": pairs_va, "notes": notes}


def main() -> None:
    out = {"plants": {}, "cross_plant": {}}
    for short in PLANTS:
        if short == "meforcezero":
            continue  # included below via cross-plant block
        r = collect(short)
        if r is None or not r["seeds"]:
            print(f"[p4sum] skip {short}: no state/seeds yet")
            continue
        out["plants"][short] = {**plant_table(r),
                                "seeds": r["seeds"],
                                "failed_freezes": r["failed_freezes"],
                                "label": r["spec"]["label"]}
    # n=20 campaign block for cross-plant comparison
    r20 = collect("meforcezero")
    out["plants"]["meforcezero"] = {**plant_table(r20),
                                    "seeds": r20["seeds"],
                                    "failed_freezes": r20["failed_freezes"],
                                    "label": r20["spec"]["label"]}

    # pooled discordant pairs across plants (disclosed stratified-sum)
    for metric in ("first_probe", "verified"):
        for a, b in (("causal", "reverse"), ("causal", "random")):
            b_sum = c_sum = 0
            for short, blk in out["plants"].items():
                mm = blk[f"mcnemar_{metric}"][f"{a}|{b}"]
                b_sum += mm["b"]
                c_sum += mm["c"]
            n = b_sum + c_sum
            if n:
                p = min(1.0, 2 * sum(math.comb(n, k)
                                    for k in range(0, min(b_sum, c_sum) + 1))
                        * 0.5 ** n)
            else:
                p = 1.0
            out["cross_plant"][f"{metric}:{a}_vs_{b}"] = {
                "b": b_sum, "c": c_sum, "p": round(p, 6)}

    out["instances"] = {}
    for r in collect_family():
        out["instances"][r["short"]] = {**plant_table(r),
                                        "seeds": r["seeds"],
                                        "failed_freezes": r["failed_freezes"],
                                        "label": r["spec"]["label"]}
        # pooled discordant pairs across instances
        for metric in ("first_probe", "verified"):
            for a, b in (("causal", "reverse"), ("causal", "random")):
                key = f"{metric}:{a}_vs_{b}"
                mm = out["instances"][r["short"]][f"mcnemar_{metric}"][f"{a}|{b}"]
                fam = out["instances"].setdefault("_pooled", {})
                slot = fam.setdefault(key, {"b": 0, "c": 0})
                slot["b"] += mm["b"]
                slot["c"] += mm["c"]

    if "_pooled" in out.get("instances", {}):
        for key, slot in out["instances"]["_pooled"].items():
            n = slot["b"] + slot["c"]
            slot["p"] = round(min(1.0, 2 * sum(math.comb(n, k)
                               for k in range(0, min(slot["b"], slot["c"]) + 1))
                            * 0.5 ** n), 6) if n else 1.0

    P4_ROOT.mkdir(parents=True, exist_ok=True)
    (P4_ROOT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out.get("cross_plant", {}), indent=1))
    print("wrote", P4_ROOT / "summary.json")


if __name__ == "__main__":
    main()
