"""Exp-C offline re-analysis: verified-attr-vs-K and deflect/overturn rates.

Reconstructs per-probe dynamics from the events.jsonl of existing snapshot
cells (no new runs): probes are backward events in order; a comply with
error_resolved=True is *overturned* when another diagnosis event follows
(the re-solve refuted the repair); the verified round is the final
non-overturned comply in cells whose manifest records a verified
attribution hit. Verified@K is therefore the within-episode cumulative
rate of the completed K=3 episodes (monotone by construction; a true K>3
sweep would need new cells and is not claimed here).

Universe: the smoke-grid me_force campaigns, all sharing the same 20
frozen blackboards — stackelberg {causal, random, reverse}, adversarial,
sequential, debate, reflexion.

Usage:
  uv run --no-sync python scripts/summarize_exp_c.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
           / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5")
SNAPSHOTS = REPO / "results" / "snapshots"
OUT = REPO / "results" / "exp_c"

METHODS = {
    "sb_causal": "ds41_ea_causal",
    "sb_random": "ds41_ea_random",
    "sb_reverse": "ds41_ea_reverse",
    "adversarial": "ds41_eb_adversarial",
    "sequential": "ds41_eb_sequential",
    "debate": "ds41_ebext_debate",
    "reflexion": "ds41_ebext_reflexion",
}


def usable_seeds() -> list[int]:
    return sorted(
        int(p.name.rsplit("_s", 1)[1])
        for p in SNAPSHOTS.glob("me_force_zero_sea_s*")
        if (p / "snapshot_state.json").exists())


def probe_sequence(evs: list[dict]) -> list[dict]:
    """Ordered probes with overturn flags; events are already time-ordered."""
    probes = []
    for idx, e in enumerate(evs):
        if e.get("step_type") != "backward":
            continue
        comply_resolved = (e.get("action") == "comply"
                           and bool(e.get("error_resolved")))
        overturned = any(evs[j].get("step_type") == "diagnosis"
                         for j in range(idx + 1, len(evs)))
        probes.append({
            "ordinal": len(probes) + 1,
            "layer": e.get("node"),
            "action": e.get("action"),
            "resolved": bool(e.get("error_resolved")),
            "comply_resolved": comply_resolved,
            "overturned": comply_resolved and overturned,
        })
    return probes


def verified_round(manifest: dict, probes: list[dict]) -> int | None:
    if not manifest["inspection"]["episode_payoff"].get(
            "verified_attribution_hit"):
        return None
    finals = [p for p in probes if p["comply_resolved"]
              and not p["overturned"]]
    return finals[-1]["ordinal"] if finals else None


def main() -> None:
    seeds = usable_seeds()
    out = {"seeds": seeds, "methods": {}, "notes": {
        "reading": "verified@K = within-episode cumulative rate of the "
                   "completed K=3 episodes; K>3 not run",
        "universe": "smoke-grid me_force n=20 campaigns, shared blackboards",
    }}

    for method, prefix in METHODS.items():
        cells = []
        for s in seeds:
            cell_dir = RESULTS / f"{prefix}_s{s}"
            mf = cell_dir / "run_manifest.json"
            ef = cell_dir / "events.jsonl"
            if not (mf.exists() and ef.exists()):
                continue
            manifest = json.loads(mf.read_text())
            evs = [json.loads(line) for line in ef.read_text().splitlines()
                    if line.strip()]
            probes = probe_sequence(evs)
            vr = verified_round(manifest, probes)
            cells.append({
                "seed": s, "probes": probes, "verified_round": vr,
                "verified": vr is not None,
                "verified_layer": (manifest["inspection"]
                                   ["episode_payoff"].get(
                                       "verified_attributed_layer")),
            })

        n = len(cells)
        all_probes = [p for c in cells for p in c["probes"]]
        n_probe = len(all_probes)
        deflects = [p for p in all_probes if p["action"] == "deflect"]
        complies_resolved = [p for p in all_probes
                             if p["comply_resolved"]]
        overturned = [p for p in all_probes if p["overturned"]]
        false_deflects = [
            p for c in cells if c["verified"] for p in c["probes"]
            if p["action"] == "deflect" and p["layer"] == c["verified_layer"]]

        out["methods"][method] = {
            "n": n,
            "verified_at_1": sum(c["verified_round"] <= 1
                                 for c in cells if c["verified_round"]),
            "verified_at_2": sum(c["verified_round"] <= 2
                                 for c in cells if c["verified_round"]),
            "verified_at_3": sum(c["verified"] for c in cells),
            "mean_probes": round(n_probe / n, 2),
            "n_probes": n_probe,
            "n_deflect": len(deflects),
            "deflect_rate": round(len(deflects) / n_probe, 3) if n_probe else None,
            "n_comply_resolved": len(complies_resolved),
            "comply_overturn_rate": (round(len(overturned)
                                           / len(complies_resolved), 3)
                                     if complies_resolved else None),
            "false_deflect_rate": (round(len(false_deflects)
                                         / len(deflects), 3)
                                   if deflects else None),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    hdr = (f"{'method':12} {'n':>2} {'v@1':>4} {'v@2':>4} {'v@3':>4} "
           f"{'probes':>6} {'defl%':>6} {'ovt%':>6} {'fDefl%':>7}")
    print(hdr)
    for m, a in out["methods"].items():
        print(f"{m:12} {a['n']:>2} {a['verified_at_1']:>4} "
              f"{a['verified_at_2']:>4} {a['verified_at_3']:>4} "
              f"{a['mean_probes']:>6} "
              f"{100*(a['deflect_rate'] or 0):>5.0f}% "
              f"{100*(a['comply_overturn_rate'] or 0):>5.0f}% "
              f"{100*(a['false_deflect_rate'] or 0):>6.0f}%")
    print("\nwrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
