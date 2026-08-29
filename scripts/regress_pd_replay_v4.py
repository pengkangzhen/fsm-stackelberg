#!/usr/bin/env python3
"""PD regen replay against v4's REAL trajectory seed (mechanism, not prompts).

Background
----------
The first regression (`regress_pd_regen_stability.py`) replays PD against the
*healthy* archived DE+ME and concludes PD-forward is deterministic-equivalent.
That cleared PD-forward in the *healthy* regime. But v4's failing round-2 PD
also goes through the **forward** entry (workflow.log says so), and v4's DE is
NOT the healthy DE — v4's DE uses indexed symbols (`eta[k,n,t]`,
`vessel_calls[h,t]`, …) while healthy uses bare symbols (`eta`,
`vessel_calls`). The data_access_guide therefore differs, and so does the
surface PD sees.

This script replays PD against v4's *real* per-round seeds to localize the
variance to one of:

  (A) v4 DE schema  +  stripped ME  +  forward PD  →  does gap ≈ 116% recur?
  (B) v4 DE schema  +  stripped ME  +  backward PD
      (with v4 round-1 PD code as `previous_output` and the INFEASIBLE
      `error_info`)                                    →  does gap ≈ 62% recur?

Outcomes
--------
- If **A reproduces ~116% gap**: the variance source is the *combination*
  (v4-DE schema + stripped-ME). PD-forward is then not unbiased on this
  surface; this is a structural DE-schema / data_access_guide bug, not a
  PD LLM bug.
- If **A does NOT reproduce** but v4's run did: residual variance is
  PD-LLM sampling (DashScope temperature handling), reproducible only by
  repeated draws.
- If **B reproduces ~62% gap on top of A**: the backward `error_info`
  context is *additively* destabilizing.
- If **A and B are both Practical Optimal**: the variance came from a
  state we are not capturing in this replay (probe/ranking/clear policy
  side-effects on `output_history`, ordering, etc.) — surfaced for the
  next isolation step.

Usage
-----
    python scripts/regress_pd_replay_v4.py \
        --trajectory results/mako/DashScope_deepseek-v4-flash/prob_tslp_ecr_demand_k3/smoke_H4_Omega5/verified_fix_sb_causal_er_v4 \
        --instance dataset/prob_tslp_ecr_demand/instances/smoke_H4_Omega5 \
        --provider DashScope --model deepseek-v4-flash \
        --out results/pd_regen_audit/replay_v4_<ts>.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import pathlib
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsm_stackelberg.agents.python_developer import (  # noqa: E402
    PYTHON_DEVELOPER_BACKWARD_STEP,
    PYTHON_DEVELOPER_FORWARD,
    PYTHON_DEVELOPER_ROLE,
    _build_data_access_guide,
    create_python_developer,
)
from fsm_stackelberg.agents.solver_executor import sandbox_exec_code  # noqa: E402
from fsm_stackelberg.data.auto_preprocessor import auto_preprocess  # noqa: E402
from fsm_stackelberg.knowledge.knowledge_loader import KnowledgeLoader  # noqa: E402
from fsm_stackelberg.schemas import BackwardStepOutput, DataEngineerOutput, ModelExpertOutput  # noqa: E402
from fsm_stackelberg.utils.code_sanitize import unshadow_gurobi_model_m  # noqa: E402


def _sanitize_python_code(code: str) -> str:
    """Same hygiene as python_developer_node applies (mechanism, not prompt)."""
    fixed, _ = unshadow_gurobi_model_m(code)
    return fixed

GAP_THRESHOLD_PCT = 1.0


# --------------------------------------------------------------------------- #
def _load_de(path: pathlib.Path) -> DataEngineerOutput:
    return DataEngineerOutput(**json.loads(path.read_text()))


def _load_me(path: pathlib.Path) -> ModelExpertOutput:
    return ModelExpertOutput(**json.loads(path.read_text()))


def _maybe_load_knowledge(me: ModelExpertOutput) -> str:
    if not me.knowledge_requests:
        return "(No domain knowledge provided — all parameters exist in data.)"
    loader = KnowledgeLoader()
    txt = loader.get_knowledge_by_names(list(me.knowledge_requests))
    return txt or "(Requested knowledge modules not found.)"


def _ask_pd_forward(
    me: ModelExpertOutput,
    de: DataEngineerOutput,
    data_access_guide: str,
    loaded_knowledge: str,
    provider: str,
    model: str,
) -> Tuple[str, Dict[str, Any]]:
    chain = create_python_developer(provider=provider, model=model)
    task = (
        PYTHON_DEVELOPER_FORWARD
        .replace("{model_blueprint}", me.model_dump_json(indent=2))
        .replace("{data_access_guide}", data_access_guide)
        .replace("{domain_knowledge}", loaded_knowledge)
    )
    t0 = time.time()
    resp = chain.invoke({"role": PYTHON_DEVELOPER_ROLE, "task": task})
    elapsed = time.time() - t0
    code = resp.content
    if isinstance(code, list):
        code = "\n".join(b["text"] for b in code if isinstance(b, dict) and b.get("type") == "text")
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    raw_len = len(code)
    code = _sanitize_python_code(code)
    return code, {"duration_s": round(elapsed, 3), "raw_chars": raw_len, "sanitized_chars": len(code)}


def _ask_pd_backward(
    me: ModelExpertOutput,
    de: DataEngineerOutput,
    problem_description: str,
    schema: Dict[str, Any],
    error_info: str,
    prior_code: str,
    provider: str,
    model: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Mirror python_developer_backward_step prompt assembly verbatim."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.callbacks import get_openai_callback
    from fsm_stackelberg.utils.llm_config import get_llm

    data_access_guide = _build_data_access_guide(de, schema)
    model_blueprint = me.model_dump_json(indent=2)
    task = PYTHON_DEVELOPER_BACKWARD_STEP.format(
        problem_description=problem_description,
        model_expert_output=model_blueprint,
        data_access_guide=json.dumps(data_access_guide, indent=2),
        error_info=error_info,
        previous_output=prior_code,
    )
    llm = get_llm(provider=provider, model=model, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", "{role}"), ("human", "{task}")])
    chain = prompt | llm.with_structured_output(BackwardStepOutput, method="json_mode")

    t0 = time.time()
    with get_openai_callback() as cb:
        result: BackwardStepOutput = chain.invoke({"role": PYTHON_DEVELOPER_ROLE, "task": task})
    elapsed = time.time() - t0

    info = {
        "duration_s": round(elapsed, 3),
        "is_caused_by_you": bool(result.is_caused_by_you),
        "refined_present": bool(result.refined_result),
        "reason_excerpt": (result.reason or "")[:300],
        "total_tokens": cb.total_tokens,
    }
    if not (result.is_caused_by_you and result.refined_result):
        return None, info

    refined = result.refined_result
    if isinstance(refined, list):
        refined = "\n".join(b["text"] for b in refined if isinstance(b, dict) and b.get("type") == "text")
    if isinstance(refined, str):
        if "```python" in refined:
            refined = refined.split("```python")[1].split("```")[0].strip()
        elif "```" in refined:
            refined = refined.split("```")[1].split("```")[0].strip()
        refined = _sanitize_python_code(refined)
    return refined, info


def _execute(data: Dict[str, Any], code: str, expected: float) -> Dict[str, Any]:
    report = sandbox_exec_code(data, code)
    obj = (report.get("result") or {}).get("objective_value")
    report["objective_value"] = obj
    report["gap_pct"] = (
        round(abs(obj - expected) / abs(expected) * 100, 4)
        if obj is not None and expected else None
    )
    practical = (
        report.get("execution_successful")
        and report.get("gurobi_status") == "OPTIMAL"
        and (report.get("gap_pct") is None or report["gap_pct"] <= GAP_THRESHOLD_PCT)
    )
    report["practical_optimal"] = bool(practical)
    return report


def _trim(code: str) -> str:
    if len(code) <= 4000:
        return code
    return code[:1500] + f"\n... [{len(code) - 3000} chars trimmed] ...\n" + code[-1500:]


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", required=True, type=pathlib.Path,
                    help="v4/v5 run dir (with DataEngineer_roundN.txt, ModelExpert_roundN.txt, PyDeveloper_roundN.txt)")
    ap.add_argument("--instance", required=True, type=pathlib.Path)
    ap.add_argument("--provider", default="DashScope")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--out", default=None, type=pathlib.Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    traj = args.trajectory
    sample = json.loads((args.instance / "sample.json").read_text())
    expected = json.loads((args.instance / "optimal.json").read_text()).get("objective")
    print(f"[setup] expected obj = {expected:.4f}")

    # v4 trajectory seeds
    de = _load_de(traj / "DataEngineer_round1.txt")
    me_stripped = _load_me(traj / "ModelExpert_round2.txt")  # post-strip ME
    pd_round1 = (traj / "PyDeveloper_round1.txt").read_text()  # PD code that crashed INFEASIBLE
    print(f"[setup] DE: {len(de.model_inputs.sets)} sets, {len(de.model_inputs.parameters)} params")
    print(f"[setup] stripped ME: {len(me_stripped.model_components.constraints)} constraints")
    print(f"[setup] PD round-1 code: {len(pd_round1)} chars")

    data, _ = auto_preprocess(sample)

    # Use the real loader so problem_description / schema match what v4 saw.
    from fsm_stackelberg.utils.utils import dataset_loader
    problem = dataset_loader("prob_tslp_ecr_demand", args.instance.name)
    problem_description = problem["description"]
    schema = problem.get("schema") or {}
    print(f"[setup] problem_description: {len(problem_description)} chars; schema keys: {list(schema)[:6]}")

    data_access_guide = json.dumps(_build_data_access_guide(de, schema), indent=2)
    loaded_knowledge = _maybe_load_knowledge(me_stripped)
    print(f"[setup] data_access_guide: {len(data_access_guide)} chars; knowledge: {len(loaded_knowledge)} chars")

    # The error_info that PD saw at round-3 backward in v4: gap=116% message
    # Reconstruct exactly as solver_executor builds it.
    error_info_round3 = json.dumps({
        "error_type": "SuboptimalSolution",
        "error_message": f"OPTIMAL but gap=116.14% (obj=3003399.5203669867, expected={expected})",
        "stack_trace": None,
    }, ensure_ascii=False, default=str)
    full_error_round3 = f"Gurobi Status: OPTIMAL\n{error_info_round3}"

    results: Dict[str, Any] = {}

    # ---------- A: forward PD with v4 DE + stripped ME ----------
    print("\n=== A: forward PD with v4 DE + stripped ME (replays v4 round-2) ===")
    codeA, infoA = _ask_pd_forward(me_stripped, de, data_access_guide, loaded_knowledge,
                                   args.provider, args.model)
    print(f"[A] PD generated {infoA['sanitized_chars']} chars in {infoA['duration_s']}s")
    repA = _execute(data, codeA, expected)
    print(f"[A] status={repA.get('gurobi_status')} obj={repA.get('objective_value')} "
          f"gap={repA.get('gap_pct')}% practical={repA['practical_optimal']}")
    results["A_forward_v4_seed"] = {
        **infoA,
        "report": {k: v for k, v in repA.items() if k != "result"},
        "code_excerpt": _trim(codeA),
    }

    # ---------- B: backward PD with v4 DE + stripped ME + INFEASIBLE error_info + prior code ----------
    # v4 round-3 backward was triggered by the 116% gap from round-2 forward,
    # not the INFEASIBLE. But to test whether *backward context alone*
    # destabilizes, we replay with the simplest faithful seed: round-1 PD
    # code as `previous_output` and the INFEASIBLE error_info that ME complied
    # to. This isolates "does adding error_info + prior code bias PD".
    print("\n=== B: backward PD with v4 DE + stripped ME + INFEASIBLE error_info ===")
    error_info_infeasible = json.dumps({
        "error_type": "OptimizationError",
        "error_message": "Gurobi returned status 'INFEASIBLE'",
        "result_summary": {"status": "INFEASIBLE", "objective_value": None},
    }, ensure_ascii=False, default=str)
    full_error_infeasible = f"Gurobi Status: INFEASIBLE\n{error_info_infeasible}"

    codeB, infoB = _ask_pd_backward(
        me_stripped, de, problem_description, schema,
        full_error_infeasible, pd_round1,
        args.provider, args.model,
    )
    if codeB is None:
        print(f"[B] PD deflected (is_caused_by_you={infoB['is_caused_by_you']}); no refined code")
        results["B_backward_v4_seed"] = {**infoB, "report": None, "code_excerpt": None}
    else:
        print(f"[B] PD complied; refined {len(codeB)} chars in {infoB['duration_s']}s")
        repB = _execute(data, codeB, expected)
        print(f"[B] status={repB.get('gurobi_status')} obj={repB.get('objective_value')} "
              f"gap={repB.get('gap_pct')}% practical={repB['practical_optimal']}")
        results["B_backward_v4_seed"] = {
            **infoB,
            "report": {k: v for k, v in repB.items() if k != "result"},
            "code_excerpt": _trim(codeB),
        }

    # ---------- C: backward PD with v4 DE + stripped ME + 116% gap error_info ----------
    print("\n=== C: backward PD with v4 DE + stripped ME + 116% gap error_info (replays v4 round-3) ===")
    codeC, infoC = _ask_pd_backward(
        me_stripped, de, problem_description, schema,
        full_error_round3, pd_round1,
        args.provider, args.model,
    )
    if codeC is None:
        print(f"[C] PD deflected; no refined code")
        results["C_backward_v4_gap_seed"] = {**infoC, "report": None, "code_excerpt": None}
    else:
        print(f"[C] PD complied; refined {len(codeC)} chars in {infoC['duration_s']}s")
        repC = _execute(data, codeC, expected)
        print(f"[C] status={repC.get('gurobi_status')} obj={repC.get('objective_value')} "
              f"gap={repC.get('gap_pct')}% practical={repC['practical_optimal']}")
        results["C_backward_v4_gap_seed"] = {
            **infoC,
            "report": {k: v for k, v in repC.items() if k != "result"},
            "code_excerpt": _trim(codeC),
        }

    # ---------- Verdict ----------
    a_gap = results["A_forward_v4_seed"].get("report", {}).get("gap_pct")
    a_reproduces = a_gap is not None and a_gap > 50.0  # did we get back to v4's ~116% ballpark?
    c_reproduces = False
    if results.get("C_backward_v4_gap_seed", {}).get("report"):
        c_gap = results["C_backward_v4_gap_seed"]["report"].get("gap_pct")
        c_reproduces = c_gap is not None and c_gap > 30.0

    verdict = {
        "A_reproduces_v4_116pct": bool(a_reproduces),
        "C_reproduces_v4_62pct": bool(c_reproduces),
        "any_reproduces": bool(a_reproduces or c_reproduces),
    }
    verdict["summary"] = (
        "REPRODUCED" if verdict["any_reproduces"] else "NOT_REPRODUCED"
    )
    print(f"\n[verdict] {verdict['summary']}  A_repro={verdict['A_reproduces_v4_116pct']}  C_repro={verdict['C_reproduces_v4_62pct']}")

    out_dir = pathlib.Path("results/pd_regen_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / f"replay_v4_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    artifact = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "trajectory": str(traj),
        "instance_dir": str(args.instance),
        "provider": args.provider,
        "model": args.model,
        "expected_objective": expected,
        "gap_threshold_pct": GAP_THRESHOLD_PCT,
        "v4_reference": {"round2_forward_gap_pct": 116.14, "round3_backward_gap_pct": 62.13},
        "results": results,
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str))
    print(f"[artifact] {out_path}")
    return 0 if not verdict["any_reproduces"] else 2  # 0=cleared, 2=reproduced (still success exit)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
