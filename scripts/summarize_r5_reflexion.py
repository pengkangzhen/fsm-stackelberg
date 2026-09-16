"""Aggregate the Reflexion K=5 budget-tilted challenger arm.

Reads the ds41_r5_reflexion_s* cells from the K=5 results tree and the
sb_causal K=3 cells from the K=3 tree (same 30 shared blackboards), and
writes results/r5_reflexion/summary.json: verified-at-round-k curve
(within-episode cumulative, k = 1..5), SSR, mean loop tokens, and the
paired exact McNemar against Stackelberg's K=3 verified attribution.

Usage:
  uv run --no-sync python scripts/summarize_r5_reflexion.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from fsm_stackelberg.constants import within_threshold

REPO = Path(__file__).resolve().parent.parent
R5 = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
      / "prob_tslp_ecr_demand_k5" / "smoke_H4_Omega5")
K3 = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
      / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5")
SNAPSHOTS = REPO / "results" / "snapshots"
PLANT = "me_force_zero_sea"
OUT = REPO / "results" / "r5_reflexion"
K_MAX = 5


def usable_seeds() -> list[int]:
    return sorted(
        int(p.name.rsplit("_s", 1)[1])
        for p in SNAPSHOTS.glob(f"{PLANT}_s*")
        if (p / "snapshot_state.json").exists())


def load_cell(root: Path, suffix: str) -> dict | None:
    mf, ef = root / suffix / "run_manifest.json", root / suffix / "events.jsonl"
    if not mf.exists():
        return None
    m = json.loads(mf.read_text())
    insp = m.get("inspection") or {}
    ep = insp.get("episode_payoff") or {}
    outcome = m.get("outcome") or {}
    cost = m.get("cost") or {}
    cfg = m.get("config") or {}
    practical = bool(outcome.get("solver_optimal")) and bool(
        within_threshold(outcome.get("gap_percent"),
                         bool(outcome.get("practical_optimal"))))
    evs = ([json.loads(line) for line in ef.read_text().splitlines()
            if line.strip()] if ef.exists() else [])
    probes = []
    for idx, e in enumerate(evs):
        if e.get("step_type") != "backward":
            continue
        comply_resolved = (e.get("action") == "comply"
                           and bool(e.get("error_resolved")))
        overturned = comply_resolved and any(
            evs[j].get("step_type") == "diagnosis"
            for j in range(idx + 1, len(evs)))
        probes.append({"ordinal": len(probes) + 1,
                       "comply_resolved": comply_resolved,
                       "overturned": overturned})
    verified = bool(ep.get("verified_attribution_hit")) and practical
    v_round = None
    if verified:
        finals = [p for p in probes
                  if p["comply_resolved"] and not p["overturned"]]
        v_round = finals[-1]["ordinal"] if finals else None
    return {
        "verified": verified,
        "verified_round": v_round,
        "ssr": practical,
        "loop_tokens": (cost.get("total_tokens") or 0)
        - (cfg.get("snapshot_forward_tokens") or 0),
        "n_probes": len(probes),
    }


def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict:
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) * 0.5 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def main() -> None:
    seeds = usable_seeds()
    r5 = {s: load_cell(R5, f"ds41_r5_reflexion_s{s}") for s in seeds}
    sb = {s: load_cell(K3, f"ds41_ea_causal_s{s}") for s in seeds}
    r5 = {s: c for s, c in r5.items() if c is not None}
    sb = {s: c for s, c in sb.items() if c is not None}

    n = len(r5)
    curve = {f"verified_at_{k}":
             sum(c["verified"] and (c["verified_round"] or 99) <= k
                 for c in r5.values()) for k in range(1, K_MAX + 1)}
    shared = sorted(set(r5) & set(sb))
    out = {
        "n": n,
        "verified": sum(c["verified"] for c in r5.values()),
        "ssr": sum(c["ssr"] for c in r5.values()),
        "mean_loop_tokens": round(sum(c["loop_tokens"] for c in r5.values()) / n),
        "max_probes": max(c["n_probes"] for c in r5.values()),
        "verified_curve": curve,
        "verified_rounds": {str(s): c["verified_round"]
                            for s, c in sorted(r5.items())
                            if c["verified"]},
        "sb_k3": {"verified": sum(c["verified"] for c in sb.values()),
                  "mean_loop_tokens": round(
                      sum(c["loop_tokens"] for c in sb.values()) / len(sb))},
        "mcnemar_verified_vs_sb_k3": mcnemar_exact([
            (r5[s]["verified"], sb[s]["verified"]) for s in shared]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
