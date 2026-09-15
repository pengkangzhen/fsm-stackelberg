#!/usr/bin/env python3
"""Phase 4 attribution-grid analyzer: runs -> layer confusion matrices.

Auto-discovers every run directory under a results root (any dir holding
``run_manifest.json`` — preferred — or legacy ``experiment_result.json``) and
emits:

1. ``grid_rows.csv``        — one normalized row per run
2. ``summary.json``         — matrices + per-method rates
3. printed tables:
   - Grid A: true root-cause layer x attributed layer (verified preferred)
   - Grid B: surface symptom category x true root-cause layer
     (evidence that the traceback locus is not the certified root)
   - Method table: first-probe / verified / attribution / SSR rates + tokens

The mako-era ``analyze_2d_attribution.py`` heuristically *guessed* dimension 2
from the surface (ERROR->PD, INFEASIBLE->ME). Here the true layer comes from
the injected plant (``--true_root_cause``), so the grid measures attribution
against ground truth instead of re-labeling symptoms.

Usage:
    python scripts/analyze_attribution_grid.py [--results-root results/mako]
        [--out-dir results/attribution_grid] [--prob smoke_H4_Omega5]
        [--provider DashScope]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fsm_stackelberg.constants import GAP_THRESHOLD, within_threshold
LAYERS = ["data_engineer", "model_expert", "python_developer"]

SURFACES = ["OPTIMAL_OK", "OPTIMAL_GAP", "INFEASIBLE", "UNBOUNDED", "ERROR", "NO_SOLVER"]

ROW_FIELDS = [
    "run_dir", "format", "provider", "model", "diagnosis_mode", "probe_order",
    "inject_id", "true_root_cause", "surface", "gurobi_status", "gap_pct",
    "practical_optimal", "status", "first_probe_layer", "first_probe_hit",
    "committed_omega", "attributed_layer", "attribution_hit",
    "verified_attributed_layer", "verified_attribution_hit", "rank_method",
    "probe_rounds", "tokens",
]


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def _gap_pct(obj: Any, expected: Any) -> Optional[float]:
    if obj is None or expected in (None, 0):
        return None
    return round(abs(obj - expected) / abs(expected) * 100, 4)


def _classify_surface(row: Dict[str, Any], error_msg: str) -> str:
    gs = (row.get("gurobi_status") or "").strip().upper()
    if not gs:
        return "ERROR" if (row.get("error_category") or "none") not in ("", "none") or error_msg else "NO_SOLVER"
    if "GUROBI_ERROR" in gs or gs == "ERROR":
        return "ERROR"
    if gs == "OPTIMAL":
        gap = row.get("gap_pct")
        if gap is not None:
            return "OPTIMAL_OK" if gap <= GAP_THRESHOLD * 100 else "OPTIMAL_GAP"
        # No measurable gap: fall back to the stored classification.
        if row.get("practical_optimal") or row.get("status"):
            return "OPTIMAL_OK"
        return "OPTIMAL_GAP"
    if gs in ("INFEASIBLE", "UNBOUNDED"):
        return gs
    return "OTHER"


def _row_from_manifest(m: dict, run_dir: Path) -> Dict[str, Any]:
    cfg = m.get("config", {})
    out = m.get("outcome", {})
    cost = m.get("cost", {})
    insp = m.get("inspection", {})
    payoff = insp.get("episode_payoff") or {}
    row = {
        "run_dir": str(run_dir),
        "format": "manifest",
        "provider": cfg.get("provider", ""),
        "model": cfg.get("model", ""),
        "diagnosis_mode": cfg.get("diagnosis_mode", ""),
        "probe_order": cfg.get("probe_order", ""),
        "inject_id": cfg.get("inject_id"),
        "true_root_cause": cfg.get("true_root_cause"),
        "gurobi_status": out.get("gurobi_status", ""),
        "practical_optimal": out.get("practical_optimal"),
        "solver_optimal": out.get("solver_optimal"),
        "status": out.get("status"),
        "error_category": out.get("error_category", "none"),
        "first_probe_layer": insp.get("first_probe_layer"),
        "first_probe_hit": insp.get("first_probe_hit"),
        "committed_omega": insp.get("committed_omega"),
        "attributed_layer": insp.get("attributed_layer"),
        "attribution_hit": insp.get("attribution_hit"),
        "verified_attributed_layer": payoff.get("verified_attributed_layer"),
        "verified_attribution_hit": payoff.get("verified_attribution_hit"),
        "rank_method": payoff.get("rank_method"),
        "probe_rounds": cost.get("probe_rounds"),
        "tokens": cost.get("total_tokens"),
        "obj_value": out.get("obj_value"),
        "expected_value": cfg.get("expected_value"),
    }
    row["gap_pct"] = out.get("gap_percent", _gap_pct(out.get("obj_value"), cfg.get("expected_value")))
    return row


def _row_from_legacy(d: dict, run_dir: Path) -> Dict[str, Any]:
    payoff = d.get("episode_payoff") or {}
    row = {
        "run_dir": str(run_dir),
        "format": "legacy",
        "provider": d.get("provider", ""),
        "model": d.get("model", ""),
        "diagnosis_mode": d.get("orchestrator_mode", ""),
        "probe_order": d.get("probe_order", "") or payoff.get("probe_order", ""),
        "inject_id": d.get("inject_id"),
        "true_root_cause": d.get("true_root_cause") or payoff.get("true_root_cause"),
        "gurobi_status": d.get("gurobi_status", ""),
        "practical_optimal": None,
        "status": d.get("status"),
        "error_category": d.get("error_category", "none"),
        "first_probe_layer": payoff.get("first_probe_layer"),
        "first_probe_hit": payoff.get("first_probe_hit"),
        "committed_omega": payoff.get("committed_omega"),
        "attributed_layer": d.get("attributed_layer") or payoff.get("attributed_layer"),
        "attribution_hit": payoff.get("attribution_hit"),
        "verified_attributed_layer": payoff.get("verified_attributed_layer"),
        "verified_attribution_hit": payoff.get("verified_attribution_hit"),
        "rank_method": payoff.get("rank_method"),
        "probe_rounds": payoff.get("probe_rounds"),
        "tokens": d.get("total_tokens") or payoff.get("tokens"),
        "obj_value": d.get("obj_value"),
        "expected_value": d.get("expected_value"),
    }
    row["gap_pct"] = _gap_pct(d.get("obj_value"), d.get("expected_value"))
    return row


def _reclassify_strict_success(row: Dict[str, Any]) -> None:
    """Re-derive strict-success flags from the measured gap.

    Stored flags (practical_optimal / status / verified hits) reflect the
    threshold live at run time; the measured gap is the authority, so a
    threshold change re-derives every verdict offline. Rows without a
    measurable gap keep their stored classification.
    """
    gap = row.get("gap_pct")
    if gap is None or within_threshold(gap) is None:
        return
    within = bool(within_threshold(gap))
    if row.get("format") == "manifest":
        solver_ok = row.get("solver_optimal")
        if solver_ok is None:
            solver_ok = (row.get("gurobi_status") or "").upper() == "OPTIMAL"
    else:
        solver_ok = (row.get("gurobi_status") or "").upper() == "OPTIMAL"
    practical = bool(solver_ok) and within
    row["practical_optimal"] = practical
    row["status"] = practical
    if row.get("verified_attribution_hit"):
        row["verified_attribution_hit"] = practical


def extract_row(run_dir: Path) -> Optional[Dict[str, Any]]:
    manifest = _load_json(run_dir / "run_manifest.json")
    if manifest is not None:
        row = _row_from_manifest(manifest, run_dir)
    else:
        legacy = _load_json(run_dir / "experiment_result.json")
        if legacy is None:
            return None
        row = _row_from_legacy(legacy, run_dir)
    _reclassify_strict_success(row)
    row["surface"] = _classify_surface(row, "")
    return row


def discover_runs(root: Path, prob_filter: str, provider_filter: str) -> List[Path]:
    runs = []
    for marker in ("run_manifest.json", "experiment_result.json"):
        for p in sorted(root.glob(f"**/{marker}")):
            d = p.parent
            if prob_filter and prob_filter not in str(d):
                continue
            if provider_filter and provider_filter not in str(d):
                continue
            if d not in runs:
                runs.append(d)
    return runs


def _fmt_layer(v: Optional[str]) -> str:
    return v if v else "(none)"


def attribution_matrix(rows: List[dict]) -> Dict[str, Dict[str, int]]:
    """Grid A: true layer x attributed layer (verified preferred, legacy fallback)."""
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not r.get("true_root_cause"):
            continue  # no-plant runs carry no ground truth for the grid
        attributed = (r.get("verified_attributed_layer")
                      if r.get("verified_attribution_hit") else None
                      ) or r.get("attributed_layer")
        matrix[_fmt_layer(r["true_root_cause"])][_fmt_layer(attributed)] += 1
    return {k: dict(v) for k, v in matrix.items()}


def surface_matrix(rows: List[dict]) -> Dict[str, Dict[str, int]]:
    """Grid B: surface symptom category x true root-cause layer."""
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if not r.get("true_root_cause"):
            continue
        matrix[r.get("surface", "OTHER")][_fmt_layer(r["true_root_cause"])] += 1
    return {k: dict(v) for k, v in matrix.items()}


def _rate(hits: List[Optional[bool]]) -> Optional[float]:
    known = [h for h in hits if h is not None]
    if not known:
        return None
    return round(sum(1 for h in known if h) / len(known), 4)


def method_table(rows: List[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("diagnosis_mode") or "(?)",
            r.get("probe_order") or "-",
            r.get("inject_id") or "(no-plant)",
        )
        groups[key].append(r)
    table = []
    for (mode, order, inject), rs in sorted(groups.items()):
        planted = [r for r in rs if r.get("true_root_cause")]
        toks = [r.get("tokens") or 0 for r in rs]
        table.append(
            {
                "diagnosis_mode": mode,
                "probe_order": order,
                "inject_id": inject,
                "n": len(rs),
                "ssr_rate": _rate([r.get("status") for r in rs]),
                "first_probe_rate": _rate([r.get("first_probe_hit") for r in planted]),
                "attribution_rate": _rate([r.get("attribution_hit") for r in planted]),
                "verified_rate": _rate([r.get("verified_attribution_hit") for r in planted]),
                "mean_tokens": round(sum(toks) / len(toks), 1) if toks else 0,
            }
        )
    return table


def _print_matrix(title: str, matrix: Dict[str, Dict[str, int]], col_order: List[str]) -> None:
    cols = [c for c in col_order if any(c in v for v in matrix.values())]
    extra = sorted({c for v in matrix.values() for c in v} - set(cols))
    cols += extra
    print(f"\n== {title} ==")
    header = f"{'':<18s}" + "".join(f"{c:>16s}" for c in cols)
    print(header)
    for row_key in sorted(matrix):
        cells = "".join(f"{matrix[row_key].get(c, 0):>16d}" for c in cols)
        print(f"{row_key:<18s}{cells}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=Path("results/mako"))
    p.add_argument("--out-dir", type=Path, default=Path("results/attribution_grid"))
    p.add_argument("--prob", default="", help="Substring filter on run dir path")
    p.add_argument("--provider", default="", help="Substring filter on provider tag")
    args = p.parse_args()

    run_dirs = discover_runs(args.results_root, args.prob, args.provider)
    rows = [r for d in run_dirs if (r := extract_row(d)) is not None]
    if not rows:
        print(f"No runs found under {args.results_root} (prob={args.prob!r}, provider={args.provider!r})")
        return

    grid_a = attribution_matrix(rows)
    grid_b = surface_matrix(rows)
    methods = method_table(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "grid_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_runs": len(rows),
        "results_root": str(args.results_root),
        "attribution_matrix": grid_a,
        "surface_vs_true_matrix": grid_b,
        "method_table": methods,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Runs analyzed: {len(rows)} (root={args.results_root})")
    _print_matrix("Grid A: true layer x attributed layer (verified preferred)", grid_a, LAYERS + ["(none)"])
    _print_matrix("Grid B: surface symptom x true layer", grid_b, LAYERS)
    print("\n== Method table ==")
    print(
        f"{'mode':<14s}{'order':<9s}{'inject':<30s}{'n':>3s}"
        f"{'SSR':>7s}{'1st-prb':>8s}{'attr':>7s}{'vrf':>7s}{'tok':>10s}"
    )
    for m in methods:
        print(
            f"{m['diagnosis_mode']:<14s}{m['probe_order']:<9s}{m['inject_id']:<30s}{m['n']:>3d}"
            f"{str(m['ssr_rate']):>7s}{str(m['first_probe_rate']):>8s}"
            f"{str(m['attribution_rate']):>7s}{str(m['verified_rate']):>7s}{m['mean_tokens']:>10.0f}"
        )
    print(f"\nSaved: {args.out_dir / 'grid_rows.csv'}")
    print(f"Saved: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
