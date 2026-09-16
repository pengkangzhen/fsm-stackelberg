"""Exp-A/B narrative-funnel audit over the ds41 smoke-grid campaigns.

Derives the prose-level numbers reported in the manuscript's Exp-A/B/C
sections directly from run manifests and event logs (the authority):

  * causal confirmation funnel: aligned first probes -> confirmed at
    round one / overturned terminal failures (final gap range) /
    attributions stolen by downstream regeneration (practical success
    without verified attribution, credit settling on the code layer);
  * reverse-arm fake confirmations: unaligned first probe absorbs the
    upstream fault, re-solve passes, no verified attribution;
  * random-arm alignment audit: verified hits from aligned vs unaligned
    shuffles (multi-round executed-refutation path);
  * deflect census across the seven method arms: probes, deflects by
    layer, guilty-layer zero-deflection check;
  * gap census over all measured DeepSeek runs (bimodality paragraph in
    the provenance section).

Usage:
  uv run --no-sync python scripts/summarize_ds41_funnel.py \
      [--out results/ds41_extend/funnel_audit.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsm_stackelberg.constants import within_threshold

REPO = Path(__file__).resolve().parent.parent
RESULTS = (REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
           / "prob_tslp_ecr_demand_k3" / "smoke_H4_Omega5")
MAKO_ROOT = REPO / "results" / "mako" / "DeepSeek_deepseek-flash"
SNAPSHOTS = REPO / "results" / "snapshots"
PLANT = "me_force_zero_sea"
ASTAR = "model_expert"

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
        for p in SNAPSHOTS.glob(f"{PLANT}_s*")
        if (p / "snapshot_state.json").exists())


def practical_of(outcome: dict) -> bool:
    return bool(outcome.get("solver_optimal")) and bool(
        within_threshold(outcome.get("gap_percent"),
                         bool(outcome.get("practical_optimal"))))


def load_cell(prefix: str, seed: int) -> dict | None:
    d = RESULTS / f"{prefix}_s{seed}"
    mf, ef = d / "run_manifest.json", d / "events.jsonl"
    if not mf.exists():
        return None
    m = json.loads(mf.read_text())
    insp = m.get("inspection") or {}
    ep = insp.get("episode_payoff") or {}
    outcome = m.get("outcome") or {}
    cost = m.get("cost") or {}
    cfg = m.get("config") or {}
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
                       "layer": e.get("node"), "action": e.get("action"),
                       "comply_resolved": comply_resolved,
                       "overturned": overturned})
    practical = practical_of(outcome)
    return {
        "seed": seed,
        "omega": insp.get("committed_omega") or [],
        "first_probe_hit": bool(insp.get("first_probe_hit")),
        "first_probe_layer": insp.get("first_probe_layer"),
        "verified": bool(ep.get("verified_attribution_hit")) and practical,
        "verified_layer": ep.get("verified_attributed_layer"),
        "practical": practical,
        "gap": outcome.get("gap_percent"),
        "final_error_agent": outcome.get("error_agent"),
        "loop_tokens": (cost.get("total_tokens") or 0)
        - (cfg.get("snapshot_forward_tokens") or 0),
        "probes": probes,
        "n_probes": len(probes),
    }


def verified_round(cell: dict) -> int | None:
    if not cell["verified"]:
        return None
    finals = [p for p in cell["probes"]
              if p["comply_resolved"] and not p["overturned"]]
    return finals[-1]["ordinal"] if finals else None


def funnel_audit() -> dict:
    seeds = usable_seeds()
    cells = {m: {s: load_cell(p, s) for s in seeds}
             for m, p in METHODS.items()}
    cells = {m: {s: c for s, c in per.items() if c is not None}
             for m, per in cells.items()}

    causal = cells["sb_causal"]
    aligned = {s: c for s, c in causal.items()
               if c["omega"] and c["omega"][0] == ASTAR}
    confirmed = {s: c for s, c in aligned.items() if verified_round(c) == 1}
    stolen = {s: c for s, c in aligned.items()
              if c["practical"] and not c["verified"]}
    overturned_terminal = {
        s: c for s, c in aligned.items()
        if s not in confirmed and s not in stolen}
    other_verified = {s: c for s, c in causal.items()
                      if c["verified"] and s not in confirmed}

    reverse = cells["sb_reverse"]
    fake_conf = {s: c for s, c in reverse.items()
                 if not c["first_probe_hit"] and c["practical"]
                 and not c["verified"]}

    random = cells["sb_random"]
    rnd_aligned = {s: c for s, c in random.items()
                   if c["omega"] and c["omega"][0] == ASTAR}
    rnd_unaligned_ver = {s: c for s, c in random.items()
                         if c["verified"] and s not in rnd_aligned}
    consistent = all(bool(c["omega"] and c["omega"][0] == ASTAR)
                     == bool(c["first_probe_hit"])
                     for c in random.values())

    deflect_census = {}
    for m, per in cells.items():
        all_probes = [p for c in per.values() for p in c["probes"]]
        deflects = [p for p in all_probes if p["action"] == "deflect"]
        guilty_deflect = [p for p in deflects if p["layer"] == ASTAR]
        complies = [p for p in all_probes if p["comply_resolved"]]
        overturned = [p for p in all_probes if p["overturned"]]
        zero_probe = [s for s, c in per.items() if c["n_probes"] == 0]
        deflect_census[m] = {
            "n_cells": len(per), "n_probes": len(all_probes),
            "n_deflect": len(deflects),
            "deflect_rate": round(len(deflects) / len(all_probes), 3)
            if all_probes else None,
            "n_comply_resolved": len(complies),
            "comply_overturn_rate": round(len(overturned) / len(complies), 3)
            if complies else None,
            "guilty_deflects": len(guilty_deflect),
            "zero_probe_seeds": zero_probe,
        }
    total_probes = sum(v["n_probes"] for v in deflect_census.values())
    total_deflect = sum(v["n_deflect"] for v in deflect_census.values())
    total_guilty_deflect = sum(v["guilty_deflects"]
                               for v in deflect_census.values())

    return {
        "seeds": seeds,
        "n_boards": len(seeds),
        "causal_funnel": {
            "aligned": sorted(aligned),
            "confirmed_at_1": sorted(confirmed),
            "stolen_by_downstream_regen": sorted(stolen),
            "overturned_terminal": {
                s: c["gap"] for s, c in sorted(overturned_terminal.items())},
            "verified_after_round1": sorted(other_verified),
        },
        "reverse_fake_confirmations": sorted(fake_conf),
        "random_audit": {
            "aligned_with_astar_first": sorted(rnd_aligned),
            "alignment_consistent_with_hits": consistent,
            "verified_from_unaligned": sorted(rnd_unaligned_ver),
        },
        "deflect_census": {
            "per_method": deflect_census,
            "total_probes": total_probes,
            "total_deflects": total_deflect,
            "guilty_layer_deflects": total_guilty_deflect,
        },
    }


def gap_census() -> dict:
    """Bimodality census over every measured run with a computable gap."""
    rows = []
    for mf in MAKO_ROOT.glob("prob_tslp_ecr_demand_k3/*/*/run_manifest.json"):
        m = json.loads(mf.read_text())
        gap = (m.get("outcome") or {}).get("gap_percent")
        if gap is None:
            continue
        rows.append({"cell": f"{mf.parent.parent.name}/{mf.parent.name}",
                     "gap": gap})
    zeros = [r for r in rows if r["gap"] <= 1e-12]
    between = sorted((r for r in rows if 1e-12 < r["gap"] < 1.0),
                     key=lambda r: r["gap"])
    large = [r for r in rows if r["gap"] >= 1.0]
    return {
        "n_total": len(rows), "n_zero": len(zeros),
        "n_between": len(between),
        "between_cells": between,
        "n_large": len(large), "min_large": min(
            (r["gap"] for r in large), default=None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ds41_extend/funnel_audit.json")
    args = ap.parse_args()

    audit = funnel_audit()
    audit["gap_census"] = gap_census()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=1))

    f = audit["causal_funnel"]
    print(f"boards n={audit['n_boards']} seeds={audit['seeds']}")
    print(f"causal: aligned={len(f['aligned'])} confirmed@1="
          f"{len(f['confirmed_at_1'])} stolen={len(f['stolen_by_downstream_regen'])} "
          f"overturned_terminal={len(f['overturned_terminal'])} "
          f"verified_after_r1={len(f['verified_after_round1'])}")
    gaps = [g for g in f["overturned_terminal"].values() if g is not None]
    if gaps:
        print(f"  overturned final gaps: {min(gaps):.1f}--{max(gaps):.1f}%")
    print(f"reverse fake confirmations: {audit['reverse_fake_confirmations']}")
    ra = audit["random_audit"]
    print(f"random: aligned={len(ra['aligned_with_astar_first'])} "
          f"consistent={ra['alignment_consistent_with_hits']} "
          f"verified_from_unaligned={ra['verified_from_unaligned']}")
    dc = audit["deflect_census"]
    print(f"deflects: {dc['total_deflects']}/{dc['total_probes']} probes, "
          f"guilty-layer deflects={dc['guilty_layer_deflects']}")
    gc = audit["gap_census"]
    print(f"gap census: total={gc['n_total']} zero={gc['n_zero']} "
          f"between={gc['n_between']} large={gc['n_large']} "
          f"min_large={gc['min_large']}")
    for r in gc["between_cells"]:
        print(f"  between: {r['cell']} gap={r['gap']:.4g}%")
    print("wrote", out)


if __name__ == "__main__":
    main()
