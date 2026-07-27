#!/usr/bin/env python3
"""Aggregate Exp-II baseline runs + optional Exp-I causal reference.

Primary metric (manuscript Exp-II): attribution_hit (last-comply â == a*).
Also report first_probe_hit, SSR@$K, tokens for cost/mechanism contrast.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_suffix(suffix: str) -> tuple[str, str]:
    m = re.match(r"^ii_(adv|seq)(?:_r\d+)?$", suffix)
    if m:
        mode = "adversarial" if m.group(1) == "adv" else "sequential"
        return mode, "causal"
    m = re.match(r"^(?:v\d+|full)_(causal|random|reverse)(?:_r\d+)?$", suffix)
    if m:
        return "stackelberg", m.group(1)
    if "adv" in suffix:
        return "adversarial", "causal"
    if "seq" in suffix:
        return "sequential", "causal"
    return "stackelberg", "unknown"


def _result_dir(
    provider: str,
    model: str,
    dataset: str,
    prob_name: str,
    max_retries: int,
    suffix: str,
) -> Path:
    model_tag = model.replace("/", "-").replace(":", "-")
    return (
        Path("results")
        / "mako"
        / f"{provider}_{model_tag}"
        / f"{dataset}_k{max_retries}"
        / prob_name
        / f"exp_i_pilot_{suffix}"
    )


def _load_run(path: Path) -> dict[str, Any] | None:
    er = path / "experiment_result.json"
    if not er.exists():
        return None
    return json.loads(er.read_text(encoding="utf-8"))


def _row(suffix: str, mode: str, order: str, data: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if data is None:
        return {
            "suffix": suffix,
            "diagnosis_mode": mode,
            "probe_order": order,
            "missing": True,
            "result_path": str(path),
        }
    payoff = data.get("episode_payoff") or {}
    return {
        "suffix": suffix,
        "diagnosis_mode": mode,
        "probe_order": order,
        "missing": False,
        "status": data.get("status"),
        "gurobi_status": data.get("gurobi_status"),
        "obj_value": data.get("obj_value"),
        "expected_value": data.get("expected_value"),
        "gap_percent": (
            round(
                abs(data["obj_value"] - data["expected_value"])
                / abs(data["expected_value"])
                * 100,
                4,
            )
            if data.get("obj_value") is not None
            and data.get("expected_value") not in (None, 0)
            else None
        ),
        "attributed_layer": data.get("attributed_layer") or payoff.get("attributed_layer"),
        "true_root_cause": data.get("true_root_cause") or payoff.get("true_root_cause"),
        "attribution_hit": payoff.get("attribution_hit"),
        "verified_attribution_hit": payoff.get("verified_attribution_hit"),
        "verified_attributed_layer": payoff.get("verified_attributed_layer"),
        "first_probe_layer": payoff.get("first_probe_layer"),
        "first_probe_hit": payoff.get("first_probe_hit"),
        "kill_hit": payoff.get("kill_hit"),
        "committed_omega": payoff.get("committed_omega"),
        "omega_source": payoff.get("omega_source"),
        "rank": payoff.get("rank"),
        "rank_method": payoff.get("rank_method"),
        "refutation_log_summary": payoff.get("refutation_log_summary"),
        "probe_rounds": payoff.get("probe_rounds"),
        "tokens": data.get("total_tokens") or payoff.get("tokens"),
        "duration_s": data.get("total_duration_s"),
        "u_L": payoff.get("u_L"),
        "u_F": payoff.get("u_F"),
        "S": payoff.get("S"),
        "inject_id": data.get("inject_id"),
        "result_path": str(path),
    }


def _rate(hits: list[bool | None]) -> dict[str, Any]:
    known = [h for h in hits if h is not None]
    if not known:
        return {"n": 0, "hits": 0, "rate": None}
    h = sum(1 for x in known if x)
    return {"n": len(known), "hits": h, "rate": round(h / len(known), 4)}


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("missing"):
            continue
        by_mode[r["diagnosis_mode"]].append(r)

    out: dict[str, Any] = {"by_mode": {}, "metric": "verified_attribution_hit"}
    for mode, rs in sorted(by_mode.items()):
        toks = [r.get("tokens") or 0 for r in rs]
        out["by_mode"][mode] = {
            "n": len(rs),
            "verified_attribution_hit": _rate(
                [r.get("verified_attribution_hit") for r in rs]
            ),
            "attribution_hit": _rate([r.get("attribution_hit") for r in rs]),
            "first_probe_hit": _rate([r.get("first_probe_hit") for r in rs]),
            "ssr": _rate([True if r.get("status") is True else False for r in rs]),
            "mean_tokens": round(sum(toks) / len(toks), 1) if toks else None,
            "mean_probe_rounds": round(
                sum((r.get("probe_rounds") or 0) for r in rs) / len(rs), 2
            )
            if rs
            else None,
        }
    return out


def _markdown(rows: list[dict[str, Any]], agg: dict[str, Any]) -> str:
    lines = [
        "# Exp-II Internal Baselines Summary",
        "",
        f"**Primary metric:** `{agg.get('metric')}` (verified â == a*)",
        "",
    ]
    for mode, stats in (agg.get("by_mode") or {}).items():
        lines.append(
            f"- **{mode}**: verified={stats['verified_attribution_hit']}, "
            f"attr={stats['attribution_hit']}, "
            f"first_probe={stats['first_probe_hit']}, "
            f"SSR={stats['ssr']}, mean_tokens={stats['mean_tokens']}, "
            f"mean_K={stats['mean_probe_rounds']}"
        )
    lines += [
        "",
        "| Suffix | Mode | VerifiedHit | AttrHit | FirstProbeHit | FirstProbe | â_ver | SSR | Gap% | K | Tokens |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(
                f"| {r['suffix']} | {r['diagnosis_mode']} | — | — | — | — | — | missing | — | — | — |"
            )
            continue
        lines.append(
            "| {suf} | {mode} | {vh} | {ah} | {fph} | {fp} | {attr} | {ssr} | {gap} | {pr} | {tok} |".format(
                suf=r["suffix"],
                mode=r["diagnosis_mode"],
                vh=r.get("verified_attribution_hit"),
                ah=r.get("attribution_hit"),
                fph=r.get("first_probe_hit"),
                fp=r.get("first_probe_layer"),
                attr=r.get("verified_attributed_layer") or r.get("attributed_layer"),
                ssr=r.get("status"),
                gap=r.get("gap_percent"),
                pr=r.get("probe_rounds"),
                tok=r.get("tokens"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/exp_ii"))
    p.add_argument("--provider", default="Qwen")
    p.add_argument("--model", default="qwen3.7-plus")
    p.add_argument("--dataset", default="prob_tslp_ecr_demand")
    p.add_argument("--prob-name", default="smoke_H4_Omega5")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--suffixes", default="")
    p.add_argument(
        "--include-exp-i-causal",
        action="store_true",
        help="Also load full_causal_r1..r5 from Exp-I for side-by-side contrast.",
    )
    args = p.parse_args()

    wanted = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    if args.include_exp_i_causal:
        for i in range(1, 6):
            suf = f"full_causal_r{i}"
            if suf not in wanted:
                wanted.append(suf)

    rows = []
    for suffix in wanted:
        mode, order = _parse_suffix(suffix)
        path = _result_dir(
            args.provider, args.model, args.dataset, args.prob_name, args.max_retries, suffix
        )
        rows.append(_row(suffix, mode, order, _load_run(path), path))

    agg = _agg(rows)
    payload = {"runs": rows, "aggregate": agg}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = _markdown(rows, agg)
    (args.out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
