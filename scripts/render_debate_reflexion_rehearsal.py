"""Render the real debate/reflexion prompts from a real failure snapshot.

Rehearsal harness (not paper data): reconstructs the diagnosis-side state
from a frozen me_force_zero_sea snapshot, builds the exact prompts the live
pipeline would send (same formatting helpers), and writes them to
results/subagent_rehearsal/debate_reflexion/ for subagent replay.

Usage:
  uv run --no-sync python scripts/render_debate_reflexion_rehearsal.py \
      --snapshot results/snapshots/me_force_zero_sea_s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsm_stackelberg.agents.diagnosis_agent import (
    DEBATE_LENSES, _build_diagnosis_context, _build_reflexion_memory,
    _evidence_format_kwargs, get_candidate_agents,
)
from fsm_stackelberg.prompts import DEBATE_PROMPT, REFLEXION_PROMPT

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "subagent_rehearsal" / "debate_reflexion"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="results/snapshots/me_force_zero_sea_s1")
    args = ap.parse_args()

    snap = json.loads((REPO / args.snapshot / "snapshot_state.json").read_text())
    state = {**snap, "provider": "DeepSeek", "model": "deepseek-flash"}
    gurobi_status = state.get("gurobi_status", "")
    error_category = state.get("error_category", "optimization")
    candidates = get_candidate_agents(gurobi_status)
    context = _build_diagnosis_context(state)
    kwargs = _evidence_format_kwargs(state, context, gurobi_status, candidates)

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"gurobi_status": gurobi_status, "candidates": candidates,
            "true_root_cause": state.get("true_root_cause"),
            "plant": state.get("fault_plant_id")}
    for lens_id, lens in enumerate(DEBATE_LENSES, start=1):
        task = DEBATE_PROMPT.format(lens_id=lens_id, lens=lens, **kwargs)
        (OUT / f"debater_{lens_id}.md").write_text(task)

    # Reflexion replay: round-2 scenario — round 1 accused the code layer and
    # the repair was overturned (fabricated memory, realistic shape), initial
    # attribution this round re-accuses python_developer.
    state_r2 = {
        **state,
        "backtrack_history": [{
            "retry_count": 1, "error_agent": "python_developer",
            "reason": "traceback in solver wrapper suggests code bug",
        }],
    }
    memory = _build_reflexion_memory(state_r2)
    reflexion_task = REFLEXION_PROMPT.format(
        memory=memory,
        initial_suspect="python_developer",
        initial_reason="traceback locus points at generated code",
        **kwargs,
    )
    (OUT / "reflexion.md").write_text(reflexion_task)
    meta["reflexion_scenario"] = {
        "memory": memory,
        "initial_suspect": "python_developer",
        "ideal_revision": state.get("true_root_cause"),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"rendered -> {OUT}")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
