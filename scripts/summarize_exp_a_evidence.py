#!/usr/bin/env python3
"""Aggregate redesigned Exp-A (evidence_rank) into summary.json + summary.md.

Gates (SPEC §10):
  - first_probe / verified: causal > reverse → pass
  - else kill_or_revise
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parse_suffix(suffix: str) -> tuple[str, str]:
    m = re.match(r"^ea_(causal|random|reverse)(?:_r\d+)?$", suffix)
    if m:
        return "stackelberg", m.group(1)
    if "causal" in suffix:
        return "stackelberg", "causal"
    if "random" in suffix:
        return "stackelberg", "random"
    if "reverse" in suffix:
        return "stackelberg", "reverse"
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


def _rate(hits: list[bool | None]) -> dict[str, Any]:
    known = [h for h in hits if h is not None]
    if not known:
        return {"n": 0, "hits": 0, "rate": None}
    h = sum(1 for x in known if x)
    return {"n": len(known), "hits": h, "rate": round(h / len(known), 4)}


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
        "attributed_layer": data.get("attributed_layer") or payoff.get("attributed_layer"),
        "true_root_cause": data.get("true_root_cause") or payoff.get("true_root_cause"),
        "attribution_hit": payoff.get("attribution_hit"),
        "verified_attribution_hit": payoff.get("verified_attribution_hit"),
        "verified_attributed_layer": payoff.get("verified_attributed_layer"),
        "kill_hit": payoff.get("kill_hit"),
        "first_probe_layer": payoff.get("first_probe_layer"),
        "first_probe_hit": payoff.get("first_probe_hit"),
        "committed_omega": payoff.get("committed_omega"),
        "omega_source": payoff.get("omega_source"),
        "rank": payoff.get("rank"),
        "rank_method": payoff.get("rank_method"),
        "refutation_log_summary": payoff.get("refutation_log_summary"),
        "probe_rounds": payoff.get("probe_rounds"),
        "tokens": data.get("total_tokens") or payoff.get("tokens"),
        "duration_s": data.get("total_duration_s"),
        "S": payoff.get("S"),
        "inject_id": data.get("inject_id"),
        "result_path": str(path),
    }


def _gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("missing"):
            continue
        by_order[r["probe_order"]].append(r)

    fp = {
        o: _rate([r.get("first_probe_hit") for r in rs])
        for o, rs in by_order.items()
    }
    vh = {
        o: _rate([r.get("verified_attribution_hit") for r in rs])
        for o, rs in by_order.items()
    }
    ssr = {
        o: _rate([True if r.get("status") is True else False for r in rs])
        for o, rs in by_order.items()
    }

    c_fp = (fp.get("causal") or {}).get("rate")
    r_fp = (fp.get("reverse") or {}).get("rate")
    c_vh = (vh.get("causal") or {}).get("rate")
    r_vh = (vh.get("reverse") or {}).get("rate")

    verdict = "inconclusive"
    note = "need causal + reverse cells"
    # SPEC: causal > reverse on first-probe / verified
    if c_fp is not None and r_fp is not None:
        if c_fp > r_fp + 1e-9 or (
            c_vh is not None and r_vh is not None and c_vh > r_vh + 1e-9
        ):
            verdict = "pass"
            note = (
                f"causal>reverse on first_probe ({c_fp} vs {r_fp})"
                + (
                    f" and/or verified ({c_vh} vs {r_vh})"
                    if c_vh is not None and r_vh is not None
                    else ""
                )
            )
        else:
            verdict = "kill_or_revise"
            note = (
                f"causal not better than reverse: first_probe {c_fp} vs {r_fp}; "
                f"verified {c_vh} vs {r_vh}"
            )

    return {
        "verdict": verdict,
        "note": note,
        "first_probe_hit": fp,
        "verified_attribution_hit": vh,
        "ssr": ssr,
        "metric": "first_probe_hit|verified_attribution_hit",
    }


def _markdown(rows: list[dict[str, Any]], gate: dict[str, Any]) -> str:
    lines = [
        "# Redesigned Exp-A (evidence_rank) Summary",
        "",
        f"**Gate verdict:** `{gate['verdict']}` — {gate['note']}",
        f"**Metrics:** `{gate.get('metric')}`",
        "",
        f"- first_probe_hit: {gate.get('first_probe_hit')}",
        f"- verified_attribution_hit: {gate.get('verified_attribution_hit')}",
        f"- SSR: {gate.get('ssr')}",
        "",
        "| Suffix | Order | FirstProbeHit | VerifiedHit | FirstProbe | ω | â_ver | SSR | K | Tokens | omega_source |",
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
            "| {suf} | {order} | {fph} | {vh} | {fp} | {om} | {attr} | {ssr} | {pr} | {tok} | {os} |".format(
                suf=r["suffix"],
                order=r["probe_order"],
                fph=r.get("first_probe_hit"),
                vh=r.get("verified_attribution_hit"),
                fp=r.get("first_probe_layer"),
                om=omega_s,
                attr=r.get("verified_attributed_layer") or r.get("attributed_layer"),
                ssr=r.get("status"),
                pr=r.get("probe_rounds"),
                tok=r.get("tokens"),
                os=r.get("omega_source"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/exp_a_evidence"))
    p.add_argument("--provider", default="DashScope")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--dataset", default="prob_tslp_ecr_demand")
    p.add_argument("--prob-name", default="smoke_H4_Omega5")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--suffixes", default="")
    args = p.parse_args()

    wanted = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    if not wanted:
        wanted = [f"ea_{o}_r{i}" for i in range(1, 4) for o in ("causal", "reverse", "random")]

    rows = []
    for suffix in wanted:
        mode, order = _parse_suffix(suffix)
        path = _result_dir(
            args.provider, args.model, args.dataset, args.prob_name, args.max_retries, suffix
        )
        rows.append(_row(suffix, mode, order, _load_run(path), path))

    gate = _gate(rows)
    payload = {"rows": rows, "gate": gate, "protocol": "evidence_informed_exp_a"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = _markdown(rows, gate)
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
