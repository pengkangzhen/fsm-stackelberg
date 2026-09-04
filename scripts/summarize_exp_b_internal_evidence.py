#!/usr/bin/env python3
"""Aggregate redesigned Exp-B internal + Exp-A causal SB for verified-attr gate.

Gate (SPEC §10): verified-attr(SB causal) ≥ verified-attr(adversarial).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def _parse_mode(suffix: str) -> str:
    if suffix.startswith("eb_adv") or "adv" in suffix:
        return "adversarial"
    if suffix.startswith("eb_seq") or "seq" in suffix:
        return "sequential"
    if suffix.startswith("ea_causal") or "causal" in suffix:
        return "stackelberg"
    return "unknown"


def _row(suffix: str, data: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    mode = _parse_mode(suffix)
    if data is None:
        return {
            "suffix": suffix,
            "diagnosis_mode": mode,
            "missing": True,
            "result_path": str(path),
        }
    payoff = data.get("episode_payoff") or {}
    return {
        "suffix": suffix,
        "diagnosis_mode": mode,
        "missing": False,
        "status": data.get("status"),
        "attributed_layer": data.get("attributed_layer") or payoff.get("attributed_layer"),
        "verified_attributed_layer": payoff.get("verified_attributed_layer"),
        "verified_attribution_hit": payoff.get("verified_attribution_hit"),
        "attribution_hit": payoff.get("attribution_hit"),
        "first_probe_hit": payoff.get("first_probe_hit"),
        "committed_omega": payoff.get("committed_omega"),
        "omega_source": payoff.get("omega_source"),
        "probe_rounds": payoff.get("probe_rounds"),
        "tokens": data.get("total_tokens") or payoff.get("tokens"),
        "S": payoff.get("S"),
        "result_path": str(path),
    }


def _gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("missing"):
            continue
        by_mode[r["diagnosis_mode"]].append(r)

    stats = {}
    for mode, rs in by_mode.items():
        toks = [r.get("tokens") or 0 for r in rs]
        stats[mode] = {
            "n": len(rs),
            "verified_attribution_hit": _rate([r.get("verified_attribution_hit") for r in rs]),
            "attribution_hit": _rate([r.get("attribution_hit") for r in rs]),
            "first_probe_hit": _rate([r.get("first_probe_hit") for r in rs]),
            "ssr": _rate([True if r.get("status") is True else False for r in rs]),
            "mean_tokens": round(sum(toks) / len(toks), 1) if toks else None,
        }

    sb = (stats.get("stackelberg") or {}).get("verified_attribution_hit") or {}
    adv = (stats.get("adversarial") or {}).get("verified_attribution_hit") or {}
    sb_r, adv_r = sb.get("rate"), adv.get("rate")

    verdict = "inconclusive"
    note = "need stackelberg causal + adversarial cells with verified fields"
    if sb_r is not None and adv_r is not None:
        if sb_r + 1e-9 >= adv_r:
            verdict = "pass"
            note = f"verified-attr(SB)={sb_r} ≥ adversarial={adv_r}"
        else:
            verdict = "fail"
            note = (
                f"verified-attr(SB)={sb_r} < adversarial={adv_r} — "
                "revise rank prompt / verified rule; do NOT run Debate"
            )

    return {
        "verdict": verdict,
        "note": note,
        "by_mode": stats,
        "metric": "verified_attribution_hit",
    }


def _markdown(rows: list[dict[str, Any]], gate: dict[str, Any]) -> str:
    lines = [
        "# Redesigned Exp-B Internal (evidence protocol) Summary",
        "",
        f"**Gate verdict:** `{gate['verdict']}` — {gate['note']}",
        f"**Primary metric:** `{gate.get('metric')}`",
        "",
    ]
    for mode, stats in (gate.get("by_mode") or {}).items():
        lines.append(
            f"- **{mode}**: verified={stats['verified_attribution_hit']}, "
            f"attr={stats['attribution_hit']}, first_probe={stats['first_probe_hit']}, "
            f"SSR={stats['ssr']}, mean_tokens={stats['mean_tokens']}"
        )
    lines += [
        "",
        "| Suffix | Mode | VerifiedHit | AttrHit | FirstProbeHit | â_ver | SSR | K | Tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(
                f"| {r['suffix']} | {r['diagnosis_mode']} | — | — | — | — | missing | — | — |"
            )
            continue
        lines.append(
            "| {suf} | {mode} | {vh} | {ah} | {fph} | {attr} | {ssr} | {pr} | {tok} |".format(
                suf=r["suffix"],
                mode=r["diagnosis_mode"],
                vh=r.get("verified_attribution_hit"),
                ah=r.get("attribution_hit"),
                fph=r.get("first_probe_hit"),
                attr=r.get("verified_attributed_layer") or r.get("attributed_layer"),
                ssr=r.get("status"),
                pr=r.get("probe_rounds"),
                tok=r.get("tokens"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/exp_b_internal_evidence"))
    p.add_argument("--provider", default="DashScope")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--dataset", default="prob_tslp_ecr_demand")
    p.add_argument("--prob-name", default="smoke_H4_Omega5")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--suffixes", default="")
    p.add_argument(
        "--ea-causal-suffixes",
        default="ea_causal_r1,ea_causal_r2,ea_causal_r3",
        help="Exp-A causal SB suffixes to include for gate comparison",
    )
    args = p.parse_args()

    wanted = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    for s in args.ea_causal_suffixes.split(","):
        s = s.strip()
        if s and s not in wanted:
            wanted.append(s)

    rows = []
    for suffix in wanted:
        path = _result_dir(
            args.provider, args.model, args.dataset, args.prob_name, args.max_retries, suffix
        )
        rows.append(_row(suffix, _load_run(path), path))

    gate = _gate(rows)
    payload = {"rows": rows, "gate": gate, "protocol": "evidence_informed_exp_b_internal"}
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
