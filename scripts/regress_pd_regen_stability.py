#!/usr/bin/env python3
"""PD-regen stability regression (mechanism, not prompts).

Isolates PythonDeveloper code-generation variance from the inspection game by
replaying PD against *fixed* DE+ME seeds from a known-healthy run, both with
the original ME and after deterministically stripping one constraint family.

Procedure
---------
1. Load DE / ME outputs from a healthy archived run (no plant, Practical
   Optimal first-pass).
2. Auto-preprocess the same instance to get the sandbox data dict.
3. **Round R0** (forward): invoke PD with the original ME; execute the
   generated code in the real sandbox; assert OPTIMAL & gap <= 1%.
4. **Round R1** (regen after strip): apply `strip_force_zero_sea_constraints`
   to the ME (a real, mechanism-level edit the diagnosis loop performs), then
   ask PD to regenerate from scratch; execute; assert OPTIMAL & gap <= 1%.

Why this is not a prompt-patch test
-----------------------------------
- Inputs to PD (DE, data, knowledge, data_access_guide) are byte-identical
  between R0 and R1; the only controlled change is ME.constraints.
- The ME edit is a *real* diagnosis-loop operation, not a contrived mutation.
- No prompt is modified based on observed failures; we only assert the
  contract the paper already makes (Practical Optimal = OPTIMAL & gap<=1%).

Usage
-----
    python scripts/regress_pd_regen_stability.py \
        --seed results/mako/DashScope_deepseek-v4-flash/prob_tslp_ecr_demand_k3/smoke_H4_Omega5/healthy_ds_v4flash_r1 \
        --instance dataset/prob_tslp_ecr_demand/instances/smoke_H4_Omega5 \
        --provider DashScope --model deepseek-v4-flash \
        --out results/pd_regen_audit/regression_<timestamp>.json

Exit code 0 = both rounds Practical Optimal. Non-zero = PD-regen variance
surfaced; the JSON artifact has the full evidence for root-causing.
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
from typing import Any, Dict, Tuple

# Make the package importable when run from the repo root.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsm_stackelberg.agents.model_expert import strip_force_zero_sea_constraints  # noqa: E402
from fsm_stackelberg.agents.python_developer import (  # noqa: E402
    PYTHON_DEVELOPER_FORWARD,
    PYTHON_DEVELOPER_ROLE,
    _sanitize_python_code,
    create_python_developer,
)
from fsm_stackelberg.agents.solver_executor import sandbox_exec_code  # noqa: E402
from fsm_stackelberg.data.auto_preprocessor import auto_preprocess  # noqa: E402
from fsm_stackelberg.knowledge.knowledge_loader import KnowledgeLoader  # noqa: E402
from fsm_stackelberg.schemas import DataEngineerOutput, ModelExpertOutput  # noqa: E402

GAP_THRESHOLD_PCT = 1.0
EXPECTED_OBJECTIVE = 1_389_581.084067007  # GT for smoke_H4_Omega5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_seed(seed_dir: pathlib.Path) -> Tuple[DataEngineerOutput, ModelExpertOutput]:
    de = DataEngineerOutput(**json.loads((seed_dir / "DataEngineer.txt").read_text()))
    me = ModelExpertOutput(**json.loads((seed_dir / "ModelExpert.txt").read_text()))
    return de, me


def _build_pd_task(
    me: ModelExpertOutput,
    de: DataEngineerOutput,
    data_access_guide: str,
    loaded_knowledge: str,
) -> str:
    """Reproduce python_developer_node's prompt assembly verbatim."""
    model_blueprint = me.model_dump_json(indent=2)
    return (
        PYTHON_DEVELOPER_FORWARD
        .replace("{model_blueprint}", model_blueprint)
        .replace("{data_access_guide}", data_access_guide)
        .replace("{domain_knowledge}", loaded_knowledge)
    )


def _ask_pd(
    me: ModelExpertOutput,
    de: DataEngineerOutput,
    data_access_guide: str,
    loaded_knowledge: str,
    provider: str,
    model: str,
) -> Tuple[str, Dict[str, Any]]:
    """Invoke the real PD chain. Returns (code, token-usage dict)."""
    chain = create_python_developer(provider=provider, model=model)
    task = _build_pd_task(me, de, data_access_guide, loaded_knowledge)
    t0 = time.time()
    resp = chain.invoke({"role": PYTHON_DEVELOPER_ROLE, "task": task})
    elapsed = time.time() - t0

    code = resp.content
    if isinstance(code, list):  # Anthropic content blocks
        code = "\n".join(b["text"] for b in code if isinstance(b, dict) and b.get("type") == "text")
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    raw_len = len(code)
    code = _sanitize_python_code(code)

    usage = {
        "duration_s": round(elapsed, 3),
        "raw_chars": raw_len,
        "sanitized_chars": len(code),
        "total_tokens": getattr(resp, "usage_metadata", {}).get("total_tokens") if hasattr(resp, "usage_metadata") else None,
    }
    return code, usage


def _execute(data: Dict[str, Any], code: str) -> Dict[str, Any]:
    """Run the real sandbox solver; attach gap classification."""
    report = sandbox_exec_code(data, code)
    obj = (report.get("result") or {}).get("objective_value")
    report["objective_value"] = obj
    if obj is not None and EXPECTED_OBJECTIVE:
        report["gap_pct"] = round(abs(obj - EXPECTED_OBJECTIVE) / abs(EXPECTED_OBJECTIVE) * 100, 4)
    else:
        report["gap_pct"] = None

    practical = (
        report.get("execution_successful")
        and report.get("gurobi_status") == "OPTIMAL"
        and (report.get("gap_pct") is None or report["gap_pct"] <= GAP_THRESHOLD_PCT)
    )
    report["practical_optimal"] = bool(practical)
    return report


def _maybe_load_knowledge(me: ModelExpertOutput) -> str:
    """Resolve any knowledge_requests in the ME against the TSLP catalog.

    Mirrors what `knowledge_loader_node` puts into state['loaded_knowledge'].
    """
    if not me.knowledge_requests:
        return "(No domain knowledge provided — all parameters exist in data.)"
    loader = KnowledgeLoader()
    loaded = loader.get_knowledge_by_names(list(me.knowledge_requests))
    if not loaded:
        return "(Requested knowledge modules not found; falling back to no text.)"
    return loaded


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", required=True, type=pathlib.Path,
                    help="Directory with healthy DataEngineer.txt + ModelExpert.txt")
    ap.add_argument("--instance", required=True, type=pathlib.Path,
                    help="Instance dir (sample.json + optimal.json)")
    ap.add_argument("--provider", default="DashScope")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--out", default=None, type=pathlib.Path,
                    help="JSON artifact path (default: results/pd_regen_audit/regression_<ts>.json)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sample = json.loads((args.instance / "sample.json").read_text())
    expected = json.loads((args.instance / "optimal.json").read_text()).get("objective")
    if expected is None:
        raise SystemExit(f"No objective in {args.instance}/optimal.json")
    print(f"[setup] GT objective = {expected:.4f}")

    de, me = _load_seed(args.seed)
    print(f"[setup] DE: {len(de.model_inputs.sets)} sets, {len(de.model_inputs.parameters)} params")
    print(f"[setup] ME: {len(me.model_components.constraints)} constraints")

    data, data_access_guide = auto_preprocess(sample)
    print(f"[setup] preprocessed data: {len(data)} fields; guide {len(data_access_guide)} chars")
    loaded_knowledge = _maybe_load_knowledge(me)

    # ---------------- R0: forward with original ME ----------------
    print("\n=== R0: PD forward with ORIGINAL ME ===")
    code0, usage0 = _ask_pd(me, de, data_access_guide, loaded_knowledge,
                            args.provider, args.model)
    print(f"[r0] PD generated {usage0['sanitized_chars']} chars in {usage0['duration_s']}s")
    rep0 = _execute(data, code0)
    print(f"[r0] status={rep0.get('gurobi_status')} obj={rep0.get('objective_value')} "
          f"gap={rep0.get('gap_pct')}% practical={rep0['practical_optimal']}")

    # ---------------- R1: regen after deterministic ME strip ----------------
    me_stripped, dropped = strip_force_zero_sea_constraints(me)
    print(f"\n=== R1: PD regen after ME strip (dropped: {dropped}) ===")
    if not dropped:
        # healthy ME has no force-zero constraint; synthesize a benign strip
        # by dropping the LAST constraint so we still test regen-after-edit.
        kept = me.model_components.constraints[:-1]
        last = me.model_components.constraints[-1].name
        me_stripped = me.model_copy(update={
            "model_components": me.model_components.model_copy(update={"constraints": kept})
        })
        dropped = [f"(synthetic drop of '{last}')"]
        print(f"[r1] healthy ME had no force-zero; synthetic drop applied: {dropped}")

    code1, usage1 = _ask_pd(me_stripped, de, data_access_guide, loaded_knowledge,
                            args.provider, args.model)
    print(f"[r1] PD generated {usage1['sanitized_chars']} chars in {usage1['duration_s']}s")
    rep1 = _execute(data, code1)
    print(f"[r1] status={rep1.get('gurobi_status')} obj={rep1.get('objective_value')} "
          f"gap={rep1.get('gap_pct')}% practical={rep1['practical_optimal']}")

    # ---------------- Verdict ----------------
    ok = rep0["practical_optimal"] and rep1["practical_optimal"]
    verdict = "PASS" if ok else "FAIL"

    out_dir = pathlib.Path("results/pd_regen_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / f"regression_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Trim full code (it can be 10+ KB); keep first/last 2 KB for forensics.
    def _trim(code: str) -> str:
        if len(code) <= 6000:
            return code
        return code[:2000] + f"\n... [{len(code) - 4000} chars trimmed] ...\n" + code[-2000:]

    artifact = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed_dir": str(args.seed),
        "instance_dir": str(args.instance),
        "provider": args.provider,
        "model": args.model,
        "expected_objective": expected,
        "gap_threshold_pct": GAP_THRESHOLD_PCT,
        "me_constraints_before": len(me.model_components.constraints),
        "me_constraints_after": len(me_stripped.model_components.constraints),
        "stripped": dropped,
        "r0": {**usage0, "report": {k: v for k, v in rep0.items() if k != "result"}, "code_excerpt": _trim(code0)},
        "r1": {**usage1, "report": {k: v for k, v in rep1.items() if k != "result"}, "code_excerpt": _trim(code1)},
        "verdict": verdict,
        "ok": ok,
    }
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str))
    print(f"\n[verdict] {verdict}  →  {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
