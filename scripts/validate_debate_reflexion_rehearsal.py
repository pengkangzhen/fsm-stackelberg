"""Validate subagent rehearsal replies through the real debate/reflexion machinery.

Rehearsal harness (not paper data): parses each reply file's fenced JSON
(HTML-escaped quotes unescaped first), validates against DebateOutput /
ReflexionOutput, aggregates the debate votes with the production helper,
and writes REPORT.md. Any schema drift caught here becomes a coercion
before any API money is spent.

Usage:
  uv run --no-sync python scripts/validate_debate_reflexion_rehearsal.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from fsm_stackelberg.agents.diagnosis_agent import _aggregate_debate_votes
from fsm_stackelberg.prompts import DEBATE_ROLE, REFLEXION_ROLE
from fsm_stackelberg.schemas import DebateOutput, ReflexionOutput

REPO = Path(__file__).resolve().parent.parent
DIR = REPO / "results" / "subagent_rehearsal" / "debate_reflexion"


def extract_json(text: str) -> dict | None:
    """Pull the first fenced JSON object, unescaping HTML entities."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fenced[0] if fenced else None
    if raw is None:
        m = re.search(r"\{.*\}", text, re.S)
        raw = m.group(0) if m else None
    if raw is None:
        return None
    for candidate in (raw, html.unescape(raw)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def main() -> None:
    meta = json.loads((DIR / "meta.json").read_text())
    candidates = meta["candidates"]
    a_star = meta["true_root_cause"]

    votes, drift = [], []
    for i in (1, 2, 3):
        reply_f = DIR / f"reply_debater_{i}.md"
        if not reply_f.exists():
            drift.append(f"debater_{i}: reply file missing")
            continue
        obj = extract_json(reply_f.read_text())
        if obj is None:
            drift.append(f"debater_{i}: no parsable JSON in reply")
            continue
        try:
            out = DebateOutput.model_validate(obj)
        except Exception as e:
            drift.append(f"debater_{i}: schema drift: {str(e)[:300]}")
            continue
        votes.append(out.suspected_agent)

    pick, note = _aggregate_debate_votes(votes, candidates)

    reflexion = {"status": "skipped"}
    reply_f = DIR / "reply_reflexion.md"
    if reply_f.exists():
        obj = extract_json(reply_f.read_text())
        if obj is None:
            reflexion = {"status": "drift", "issue": "no parsable JSON"}
        else:
            try:
                out = ReflexionOutput.model_validate(obj)
                reflexion = {
                    "status": "ok",
                    "final": out.suspected_agent,
                    "revised_from_initial": out.suspected_agent != "python_developer",
                    "reflection": out.reflection[:200],
                }
            except Exception as e:
                reflexion = {"status": "drift", "issue": str(e)[:300]}

    report = {
        "meta": meta,
        "debate": {"votes": votes, "pick": pick, "note": note,
                   "pick_is_a_star": pick == a_star},
        "reflexion": reflexion,
        "schema_drift": drift,
        "verdict": "PASS" if not drift else "DRIFT-CAUGHT",
    }
    (DIR / "REPORT.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
