#!/usr/bin/env python3
"""Finalize overnight Exp-II: MORNING.md + HANDOVER snippets + manuscript table."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_summary() -> dict[str, Any] | None:
    p = ROOT / "results/exp_ii/summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt_rate(d: dict[str, Any] | None) -> str:
    if not d or d.get("rate") is None:
        return "—"
    return f"{d['hits']}/{d['n']} ({d['rate']:.2f})"


def write_morning(status: str, spend: dict[str, str], agg: dict[str, Any] | None) -> None:
    lines = [
        "# Morning checklist — overnight autonomous run",
        "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Status",
        f"- Overnight Exp-II finalize status: **`{status}`**",
        f"- Exp-II DONE: `{(ROOT / 'results/exp_ii/DONE').exists()}`",
        f"- Tokens spent (Stage1 file): `{spend.get('TOKENS_SPENT', '?')}`",
        f"- Est. RMB: `{spend.get('EST_RMB', '?')}` (hard cap ¥10)",
        f"- API calls: `{spend.get('API_CALLS', '?')}`",
        "",
        "## Exp-I (already done before overnight)",
        "- `results/exp_i_full/summary.md` — causal 0.80 > random 0.60 > reverse 0.00 (`pass`)",
        "",
        "## Exp-II aggregate",
        "",
    ]
    if agg and agg.get("by_mode"):
        for mode, stats in agg["by_mode"].items():
            lines.append(
                f"- **{mode}**: attr={_fmt_rate(stats.get('attribution_hit'))}, "
                f"first_probe={_fmt_rate(stats.get('first_probe_hit'))}, "
                f"SSR={_fmt_rate(stats.get('ssr'))}, "
                f"mean_tokens={stats.get('mean_tokens')}"
            )
    else:
        lines.append("- (no aggregate yet — check logs)")
    lines += [
        "",
        "## Read these first",
        "```bash",
        "cat results/exp_ii/DONE",
        "cat results/exp_ii/TOKEN_SPEND.txt",
        "cat results/exp_ii/summary.md",
        "cat results/exp_i_full/summary.md",
        "```",
        "",
        "## Resume if incomplete",
        "```bash",
        "STAGE=1 SKIP_EXISTING=1 bash scripts/launch_exp_ii.sh",
        "```",
        "",
    ]
    (ROOT / "results/MORNING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_handover(agg: dict[str, Any] | None, spend: dict[str, str], status: str) -> None:
    path = ROOT / "HANDOVER.md"
    text = path.read_text(encoding="utf-8")
    block_lines = [
        "  - **Exp-II overnight** "
        f"({datetime.now().date().isoformat()}, status=`{status}`):",
        "    adversarial + sequential, plant `me_force_zero_sea`, n=5, K=3,",
        "    Qwen/`qwen3.7-plus`. Primary `attribution_hit`.",
    ]
    if agg and agg.get("by_mode"):
        for mode, stats in agg["by_mode"].items():
            block_lines.append(
                f"    - {mode}: attr {_fmt_rate(stats.get('attribution_hit'))}, "
                f"first_probe {_fmt_rate(stats.get('first_probe_hit'))}, "
                f"SSR {_fmt_rate(stats.get('ssr'))}, "
                f"mean_tok={stats.get('mean_tokens')}"
            )
    block_lines += [
        f"    Spend ≈ ¥{spend.get('EST_RMB', '?')} "
        f"({spend.get('TOKENS_SPENT', '?')} tokens). "
        "Summary: `results/exp_ii/summary.md`.",
        "    Runners: `scripts/run_exp_ii_overnight.sh`, `run_exp_ii.sh`, "
        "`summarize_exp_ii.py`.",
    ]
    block = "\n".join(block_lines) + "\n"

    marker = "  - Manuscript: Exp-I n=5 table drafted; Exp-II–IV still open."
    if "Exp-II overnight" in text:
        # Replace previous overnight block roughly between Exp-II overnight and Manuscript
        text = re.sub(
            r"  - \*\*Exp-II overnight\*\*.*?(?=  - Manuscript:)",
            block,
            text,
            count=1,
            flags=re.S,
        )
    elif marker in text:
        text = text.replace(marker, block + marker, 1)
    else:
        text = block + "\n" + text

    # Update Exp-II row / next task if present
    text = text.replace(
        "| **Exp-II** | Internal baselines — **OPEN** |",
        f"| **Exp-II** | Internal baselines — **{status.upper()}** |",
    )
    if status == "ok":
        text = text.replace(
            r"**Immediate next task:** **Exp-II** (adversarial + sequential) on the same"
            "\nplant / blackboard / \\(K\\). Prefer Qwen until MiMo/DeepSeek keys are"
            "\nrestored. Exp-I n=5 supports order directionality but is not a final Prop.~1"
            "\nclaim; expand \\(n\\) only with budget approval.",
            "**Immediate next task:** Implement Debate + Reflexion for **Exp-III**;\n"
            "then Exp-IV diagnostics. Prefer Qwen until MiMo/DeepSeek keys are restored.",
        )
    path.write_text(text, encoding="utf-8")


def patch_manuscript(agg: dict[str, Any] | None) -> None:
    if not agg or not agg.get("by_mode"):
        return
    path = ROOT / "els-cas-templates/manuscript.tex"
    text = path.read_text(encoding="utf-8")
    if "tab:exp_ii" in text:
        return

    order_modes = ["stackelberg", "adversarial", "sequential"]
    rows_tex = []
    for mode in order_modes:
        if mode not in agg["by_mode"]:
            continue
        s = agg["by_mode"][mode]
        ah = s.get("attribution_hit") or {}
        fph = s.get("first_probe_hit") or {}
        ssr = s.get("ssr") or {}
        mt = s.get("mean_tokens")
        ah_tex = (
            f"{ah['hits']}/{ah['n']} ({ah['rate']:.2f})" if ah.get("n") else "—"
        )
        fph_tex = (
            f"{fph['hits']}/{fph['n']} ({fph['rate']:.2f})" if fph.get("n") else "—"
        )
        ssr_tex = f"{ssr['hits']}/{ssr['n']}" if ssr.get("n") else "—"
        mt_tex = f"${mt/1e5:.2f}{{\\times}}10^{{5}}$" if mt else "—"
        label = {
            "stackelberg": r"stackelberg (causal; Exp-I)",
            "adversarial": "adversarial",
            "sequential": "sequential",
        }[mode]
        rows_tex.append(
            f"{label} & {ah_tex} & {fph_tex} & {ssr_tex} & {mt_tex} \\\\"
        )

    table = r"""
\paragraph{Results (n=5; same plant as Exp-I).}
Table~\ref{tab:exp_ii} contrasts causal-order stackelberg inspection with the
internal baselines under identical $K$, blackboard, and repair path. Primary
contrast is last-comply attribution; first-probe hit and SSR@$K$ are secondary.

\begin{table}[t]
\centering
\caption{Exp-II internal baselines ($n{=}5$; $a^\star{=}$\texttt{model\_expert}).
Stackelberg causal cells are reused from Exp-I.}
\label{tab:exp_ii}
\begin{tabular}{@{}l c c c c@{}}
\toprule
Method & Attr.\ hit & First-probe hit & SSR@$K$ & Mean tokens \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    needle = r"\subsection{Exp-III: External verification baselines}"
    if needle not in text:
        return
    # Insert before Exp-III, after Exp-II section body
    # Find Exp-II section end: before Exp-III
    text = text.replace(needle, table + "\n" + needle, 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="partial")
    ap.add_argument("--orch-log", default="")
    args = ap.parse_args()

    spend: dict[str, str] = {}
    spend_path = ROOT / "results/exp_ii/TOKEN_SPEND.txt"
    if spend_path.exists():
        for line in spend_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                spend[k.strip()] = v.strip()

    summary = _load_summary()
    agg = (summary or {}).get("aggregate")
    write_morning(args.status, spend, agg)
    patch_handover(agg, spend, args.status)
    if args.status in {"ok", "partial"}:
        patch_manuscript(agg)
    print(f"finalize done status={args.status}")


if __name__ == "__main__":
    main()
