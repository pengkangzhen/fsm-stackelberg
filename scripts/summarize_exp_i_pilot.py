#!/usr/bin/env python3
"""Aggregate Exp-I pilot result dirs into summary.json + summary.md.

Kill metric (v3+): ``kill_hit`` / ``first_probe_hit`` = whether committed ω
probed a* first. Last-comply ``attribution_hit`` is reported but secondary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# suffix -> (diagnosis_mode, probe_order) for known single-cell names
SUFFIXES = {
    "sb_causal": ("stackelberg", "causal"),
    "sb_reverse": ("stackelberg", "reverse"),
    "sb_random": ("stackelberg", "random"),
    "adv": ("adversarial", "causal"),
    "v2_sb_causal": ("stackelberg", "causal"),
    "v2_sb_random": ("stackelberg", "random"),
}


def _parse_suffix(suffix: str) -> tuple[str, str]:
    if suffix in SUFFIXES:
        return SUFFIXES[suffix]
    # v3_causal_r1 / v3_random_r2
    m = re.match(r"^v\d+_(causal|random|reverse)(?:_r\d+)?$", suffix)
    if m:
        return "stackelberg", m.group(1)
    if "causal" in suffix:
        return "stackelberg", "causal"
    if "random" in suffix:
        return "stackelberg", "random"
    if "reverse" in suffix:
        return "stackelberg", "reverse"
    if "adv" in suffix:
        return "adversarial", "causal"
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


def _kill_flag(payoff: dict[str, Any]) -> bool | None:
    if "kill_hit" in payoff and payoff["kill_hit"] is not None:
        return payoff["kill_hit"]
    if "first_probe_hit" in payoff and payoff["first_probe_hit"] is not None:
        return payoff["first_probe_hit"]
    return payoff.get("attribution_hit")


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
        "kill_hit": _kill_flag(payoff),
        "first_probe_layer": payoff.get("first_probe_layer"),
        "first_probe_hit": payoff.get("first_probe_hit"),
        "committed_omega": payoff.get("committed_omega"),
        "true_root_rank_in_omega": payoff.get("true_root_rank_in_omega"),
        "plant_layer_complied": payoff.get("plant_layer_complied"),
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


def _kill_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by probe_order using kill_hit (first-probe) when available."""
    by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("missing"):
            continue
        by_order[r["probe_order"]].append(r)

    causal_rate = _rate([r.get("kill_hit") for r in by_order.get("causal", [])])
    random_rate = _rate([r.get("kill_hit") for r in by_order.get("random", [])])
    reverse_rate = _rate([r.get("kill_hit") for r in by_order.get("reverse", [])])

    c_rate, r_rate = causal_rate["rate"], random_rate["rate"]
    verdict = "inconclusive"
    note = ""
    if c_rate is None or (random_rate["n"] == 0 and reverse_rate["n"] == 0):
        note = "Incomplete cells; treat as inconclusive."
    elif random_rate["n"] > 0 and c_rate is not None and r_rate is not None:
        if c_rate > r_rate + 1e-9:
            verdict = "pass"
            note = (
                f"causal first-probe hit-rate {c_rate} > random {r_rate} "
                f"(n_c={causal_rate['n']}, n_r={random_rate['n']}) — proceed."
            )
        elif abs(c_rate - r_rate) < 1e-9:
            if c_rate >= 0.999:
                verdict = "kill_or_narrow"
                note = "causal ≈ random (both high) — order effect not supported; narrow claim."
            elif c_rate <= 1e-9:
                verdict = "kill_or_revise"
                note = "causal ≈ random (both ~0) — plant/ω still broken; revise before full Exp-I."
            else:
                verdict = "kill_or_narrow"
                note = f"causal ≈ random ({c_rate}) — no order advantage on this plant."
        else:
            verdict = "kill"
            note = (
                f"random first-probe hit-rate {r_rate} > causal {c_rate} — "
                "contradicts Prop.1 directionality."
            )
    elif reverse_rate["n"] > 0 and c_rate is not None and reverse_rate["rate"] is not None:
        if c_rate > reverse_rate["rate"] + 1e-9:
            verdict = "pass"
            note = f"causal > reverse ({c_rate} vs {reverse_rate['rate']})."
        else:
            verdict = "kill_or_revise"
            note = f"causal not better than reverse ({c_rate} vs {reverse_rate['rate']})."

    return {
        "verdict": verdict,
        "note": note,
        "metric": "kill_hit(=first_probe_hit)",
        "causal": causal_rate,
        "random": random_rate,
        "reverse": reverse_rate,
        # back-compat single-cell fields
        "causal_hit": (
            by_order["causal"][0].get("kill_hit") if len(by_order.get("causal", [])) == 1 else None
        ),
        "random_hit": (
            by_order["random"][0].get("kill_hit") if len(by_order.get("random", [])) == 1 else None
        ),
    }


def _markdown(rows: list[dict[str, Any]], kill: dict[str, Any]) -> str:
    lines = [
        "# Exp-I Pilot Summary",
        "",
        f"**Kill verdict:** `{kill['verdict']}` — {kill['note']}",
        f"**Kill metric:** `{kill.get('metric', 'kill_hit')}`",
        "",
        f"- causal: {kill.get('causal')}",
        f"- random: {kill.get('random')}",
        f"- reverse: {kill.get('reverse')}",
        "",
        "| Suffix | Probe | KillHit | FirstProbe | ω | Lastâ | AttrHit | SSR | Gap% | K | Tokens |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(
                f"| {r['suffix']} | {r['probe_order']} | — | — | — | — | — | missing | — | — | — |"
            )
            continue
        omega = r.get("committed_omega")
        omega_s = "→".join(omega) if isinstance(omega, list) else omega
        lines.append(
            "| {suf} | {order} | {kh} | {fp} | {om} | {attr} | {ah} | {ssr} | {gap} | {pr} | {tok} |".format(
                suf=r["suffix"],
                order=r["probe_order"],
                kh=r.get("kill_hit"),
                fp=r.get("first_probe_layer"),
                om=omega_s,
                attr=r.get("attributed_layer"),
                ah=r.get("attribution_hit"),
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
    p.add_argument("--out-dir", type=Path, default=Path("results/exp_i_pilot"))
    p.add_argument("--provider", default="Qwen")
    p.add_argument("--model", default="qwen3.7-plus")
    p.add_argument("--dataset", default="prob_tslp_ecr_demand")
    p.add_argument("--prob-name", default="smoke_H4_Omega5")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--suffixes",
        default="",
        help="Comma-separated suffix keys. Example: v3_causal_r1,v3_random_r1,...",
    )
    args = p.parse_args()

    if args.suffixes.strip():
        wanted = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    else:
        wanted = list(SUFFIXES.keys())

    rows = []
    for suffix in wanted:
        mode, order = _parse_suffix(suffix)
        path = _result_dir(
            args.provider, args.model, args.dataset, args.prob_name, args.max_retries, suffix
        )
        rows.append(_row(suffix, mode, order, _load_run(path), path))

    kill = _kill_verdict(rows)
    payload = {"runs": rows, "kill_criteria": kill}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = _markdown(rows, kill)
    (args.out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
