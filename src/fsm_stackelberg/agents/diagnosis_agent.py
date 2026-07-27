"""DiagnosisAgent - LangChain implementation.

Diagnoses execution failures and drives repair. Supports three modes:
- stackelberg: Inspection game — DiagnosisAgent commits a causal-order probing
  policy with executed refutation; accused agents are inspectees (default).
- adversarial: Single-judge LLM accusation + accused-agent self-check (baseline).
- sequential: Reverse-order backward_step chain handled by workflow routing (baseline).
"""

import json
import logging
import random
import re
import time
from typing import Dict, List, Optional, Any, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback

from ..schemas import DiagnosisOutput
from ..prompts import (
    DIAGNOSIS_ROLE,
    DIAGNOSIS_PROMPT,
    get_diagnosis_prompt,
)
from ..utils.llm_config import get_llm, DEFAULT_PROVIDER, DEFAULT_MODEL
from ..utils.run_log import record_step_event

logger = logging.getLogger(__name__)


def parse_error_key(gurobi_status: str) -> Optional[List[Any]]:
    """解析错误消息中的键.

    支持解析以下格式:
    - ('Dalian Port', 'Dalian Port', 'truck', 0) → ['Dalian Port', 'Dalian Port', 'truck', 0]
    - ('$\\delta_{pq}$',) → ['$\\delta_{pq}$']

    Args:
        gurobi_status: Gurobi 状态字符串，可能包含错误键

    Returns:
        解析后的键列表，或 None 如果无法解析
    """
    if not gurobi_status:
        return None

    # 匹配 ERROR: (key1, key2, ...) 格式
    match = re.search(r"ERROR:\s*\((.+)\)$", gurobi_status)
    if not match:
        return None

    key_str = match.group(1)

    # 解析键的各个元素
    elements = []
    # 匹配字符串（单引号或双引号包裹）和数字
    pattern = r"'([^']*)'|\"([^\"]*)\"|(\d+)"
    for m in re.finditer(pattern, key_str):
        if m.group(1) is not None:
            elements.append(m.group(1))
        elif m.group(2) is not None:
            elements.append(m.group(2))
        elif m.group(3) is not None:
            elements.append(int(m.group(3)))

    return elements if elements else None


def extract_error_context(
    stack_trace: Optional[str],
    python_code: str,
    context_lines: int = 10
) -> Dict[str, Any]:
    """从 stack trace 提取错误上下文.

    Args:
        stack_trace: 完整的 Python traceback
        python_code: 生成的完整 Python 代码
        context_lines: 提取的上下文行数

    Returns:
        {
            "available": bool,            # 是否成功提取
            "error_line_number": int,     # 出错行号
            "error_function": str,        # 出错函数
            "error_line_content": str,    # 出错行内容
            "context_before": List[str],  # 出错前的代码
            "context_after": List[str],   # 出错后的代码
        }
    """
    if not stack_trace or not python_code:
        return {"available": False}

    # 解析 traceback 中的行号
    # 格式: File "...", line 123, in function_name
    match = re.search(r'File ".*", line (\d+), in (\w+)', stack_trace)
    if not match:
        return {"available": False}

    line_num = int(match.group(1))
    func_name = match.group(2)

    # 提取代码上下文
    code_lines = python_code.split('\n')

    # 验证行号有效
    if line_num < 1 or line_num > len(code_lines):
        return {"available": False}

    start = max(0, line_num - context_lines - 1)
    end = min(len(code_lines), line_num + context_lines)

    return {
        "available": True,
        "error_line_number": line_num,
        "error_function": func_name,
        "error_line_content": code_lines[line_num - 1].strip() if line_num <= len(code_lines) else None,
        "context_before": [f"{i+1}: {code_lines[i]}" for i in range(start, line_num - 1)],
        "context_after": [f"{i+1}: {code_lines[i]}" for i in range(line_num, end)],
    }


def analyze_error_key(error_key: Optional[List[Any]]) -> Dict[str, Any]:
    """分析错误键的语义特征.

    Args:
        error_key: 解析后的键列表

    Returns:
        包含分析结果的字典
    """
    if not error_key:
        return {"parsed": False}

    analysis = {
        "parsed": True,
        "key": error_key,
        "length": len(error_key),
    }

    # 检测自环（前两个元素相同）
    if len(error_key) >= 2:
        first, second = error_key[0], error_key[1]
        if first == second:
            analysis["is_self_loop"] = True
            analysis["self_loop_nodes"] = first
        else:
            analysis["is_self_loop"] = False

    # 检测 LaTeX 符号
    key_str = str(error_key)
    if "$" in key_str or "\\delta" in key_str or "\\text" in key_str:
        analysis["contains_latex"] = True

    return analysis


def get_data_keys_summary(sample: Dict) -> Dict[str, Any]:
    """获取数据结构的摘要信息.

    Args:
        sample: sample.json 数据

    Returns:
        数据结构摘要
    """
    if not sample:
        return {"available": False}

    summary = {
        "available": True,
        "sets": list(sample.get("sets", {}).keys()) if "sets" in sample else [],
        "parameters": list(sample.get("_meta", {}).get("parameters", {}).keys()) if "_meta" in sample else [],
    }

    # 检查 allowed_transport 的键格式（用于判断有效路径）
    if "allowed_transport" in sample.get("_meta", {}).get("parameters", {}):
        summary["allowed_transport_dims"] = sample["_meta"]["parameters"]["allowed_transport"].get("dims", [])

    return summary


# Causal layering: data ▹ model ▹ code (manuscript §stackelberg)
CAUSAL_LAYERS: Tuple[str, ...] = (
    "data_engineer",
    "model_expert",
    "python_developer",
)

VERIFICATION_RULE = "executed_refutation"  # re-solve under strict-success rule


def get_candidate_agents(gurobi_status: str = "") -> List[str]:
    """根据求解器状态确定候选 Agent（按先验可能性排序）.

    在 2D 归因框架中，管线任何阶段的错误都可能传播到下游，
    因此大部分场景下三个 Agent 都是候选人。

    Args:
        gurobi_status: Gurobi 求解状态（空字符串表示求解器未触达）

    Returns:
        候选 Agent 列表，按先验可能性排序
    """
    if not gurobi_status:
        # CRASH: 代码没跑到求解器，PD 最可能，其次 ME、DE
        return ["python_developer", "model_expert", "data_engineer"]
    if gurobi_status == "OPTIMAL":
        # WRONG_OBJ: 公式缺陷最可能 → ME 优先，其次 PD、DE
        return ["model_expert", "python_developer", "data_engineer"]
    # INFEASIBLE / UNBOUNDED / ERROR
    return ["model_expert", "python_developer", "data_engineer"]


def build_probe_order(
    gurobi_status: str = "",
    probe_order: str = "causal",
    *,
    probe_seed: int | None = None,
) -> List[str]:
    """Commit the inspector's probing order ω.

    - causal: data ⊳ model ⊳ code, rotated so Prior(status) is first (contribution).
    - reverse: pure code ⊳ model ⊳ data (no Prior rotation — ablation must differ).
    - random: full shuffle of the three layers (no Prior rotation). Optional
      ``probe_seed`` makes the shuffle reproducible across runs.
    """
    if probe_order == "reverse":
        return list(reversed(CAUSAL_LAYERS))

    if probe_order == "random":
        omega = list(CAUSAL_LAYERS)
        rng = random.Random(probe_seed) if probe_seed is not None else random.Random()
        rng.shuffle(omega)
        return omega

    # causal (default): Prior(status) seeds the start; remainder follows causal cycle.
    omega = list(CAUSAL_LAYERS)
    seed = get_candidate_agents(gurobi_status)[0]
    if seed in omega:
        i = omega.index(seed)
        return omega[i:] + omega[:i]
    return omega


def next_unclear_layer(probe_queue: List[str], cleared_layers: List[str]) -> Optional[str]:
    """NextCausalLayer: first layer in the committed queue not yet cleared."""
    cleared = set(cleared_layers or [])
    for layer in probe_queue:
        if layer not in cleared:
            return layer
    return None


def create_diagnosis_agent(provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL):
    """Create a DiagnosisAgent for error diagnosis.

    Args:
        provider: LLM provider (default: DeepSeek)
        model: The model to use (default: deepseek-chat)

    Returns:
        A LangChain chain that produces DiagnosisOutput
    """
    llm = get_llm(provider=provider, model=model, temperature=0)

    prompt = get_diagnosis_prompt()

    chain = prompt | llm.with_structured_output(DiagnosisOutput, method="json_mode")

    return chain


def diagnosis_agent_node(state: Dict) -> Dict:
    """LangGraph node function for DiagnosisAgent.

    Analyzes execution errors and determines which agent is responsible.

    For stackelberg mode: Inspector commits σ=(ω, ν) and probes the next
                        uncleared causal layer (no single-judge LLM call).
    For adversarial mode: Uses LLM to accuse the most likely responsible agent.
    For sequential mode: Sets error_agent to python_developer (first in chain).

    Args:
        state: The current graph state containing error_info and agent outputs

    Returns:
        Updated state with error_agent identified and retry_count incremented
    """
    logger.info("DiagnosisAgent: Analyzing error...")

    # Get current retry count
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count >= max_retries:
        logger.warning(f"Max retries ({max_retries}) exceeded, cannot diagnose further")
        return {
            **state,
            "error_agent": None,
        }

    # 获取诊断模式
    diagnosis_mode = state.get("diagnosis_mode", "stackelberg")
    logger.info(f"DiagnosisAgent: mode={diagnosis_mode}")

    # 获取错误上下文
    error_category = state.get("error_category", "execution")
    gurobi_status = state.get("gurobi_status", "")
    error_info = state.get("error_info", "Unknown error")

    logger.info(f"DiagnosisAgent: error_category={error_category}, gurobi_status={gurobi_status}")

    # Stackelberg inspection fields (may be updated by _stackelberg_diagnosis)
    inspection_policy = state.get("inspection_policy")
    probe_queue = list(state.get("probe_queue") or [])
    cleared_layers = list(state.get("cleared_layers") or [])
    refutation_log = list(state.get("refutation_log") or [])

    # 根据模式选择诊断方式
    if diagnosis_mode == "sequential":
        # Sequential mode: Start from python_developer, workflow will handle the chain
        error_agent = "python_developer"
        confidence = 0.0
        reason = "Sequential diagnosis: starting from python_developer"
        metrics = {"total_tokens": 0, "duration_s": 0.0}
    elif diagnosis_mode == "stackelberg":
        (
            error_agent,
            confidence,
            reason,
            metrics,
            inspection_policy,
            probe_queue,
            cleared_layers,
            refutation_log,
        ) = _stackelberg_diagnosis(
            state, error_category, gurobi_status, error_info
        )
    else:
        # Adversarial baseline: single-judge LLM accusation
        error_agent, confidence, reason, metrics = _adversarial_diagnosis(
            state, error_category, gurobi_status, error_info
        )

    logger.info(f"DiagnosisAgent result: suspected={error_agent}, "
                f"confidence={confidence:.2f}, reason={reason}")

    # Record diagnosis event
    diagnosis_history = state.get("backtrack_history", [])
    diagnosis_history.append({
        "retry_count": retry_count + 1,
        "diagnosis_mode": diagnosis_mode,
        "error_category": error_category,
        "gurobi_status": gurobi_status,
        "error_info": error_info[:200] if error_info else None,
        "error_agent": error_agent,
        "confidence": confidence,
        "reason": reason,
        "inspection_policy": inspection_policy,
        "probe_queue": probe_queue,
        "cleared_layers": cleared_layers,
    })

    artifact_payload = {
        "retry_count": retry_count + 1,
        "diagnosis_mode": diagnosis_mode,
        "probed_agent": error_agent,
        "suspected_agent": error_agent,
        "confidence": confidence,
        "reason": reason,
        "committed_omega": (inspection_policy or {}).get("omega") or probe_queue,
        "inspection_policy": inspection_policy,
        "probe_queue": probe_queue,
        "cleared_layers": cleared_layers,
        "error_category": error_category,
        "gurobi_status": gurobi_status,
    }

    # Update metrics
    last_outcome = None
    if state.get("error_agent"):
        last_outcome = "deflect" if not state.get("error_resolved") else "refuted_comply"
    step_metrics = record_step_event(
        state,
        node="diagnosis_agent",
        step_type="diagnosis",
        duration_s=metrics["duration_s"],
        prompt_tokens=metrics.get("prompt_tokens", 0),
        completion_tokens=metrics.get("completion_tokens", 0),
        total_tokens=metrics["total_tokens"],
        artifact=artifact_payload,
        artifact_name=f"diagnosis_r{retry_count + 1}",
        extra={
            "diagnosis_mode": diagnosis_mode,
            "probed_agent": error_agent,
            "committed_omega": (inspection_policy or {}).get("omega") or probe_queue,
            "cleared_layers": list(cleared_layers),
            "verdict": reason[:300] if reason else None,
            "last_probe_outcome": last_outcome,
            "error_category": error_category,
            "gurobi_status": gurobi_status,
            "confidence": confidence,
        },
    )

    node_metrics = state.get("node_metrics", {})
    if "diagnosis_agent" not in node_metrics:
        node_metrics["diagnosis_agent"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    node_metrics["diagnosis_agent"]["total_duration_s"] += metrics["duration_s"]
    node_metrics["diagnosis_agent"]["total_tokens"] += metrics["total_tokens"]
    node_metrics["diagnosis_agent"]["num_calls"] += 1

    agent_metrics = state.get("agent_metrics", {})
    if "diagnosis_agent" not in agent_metrics:
        agent_metrics["diagnosis_agent"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    agent_metrics["diagnosis_agent"]["total_duration_s"] += metrics["duration_s"]
    agent_metrics["diagnosis_agent"]["total_tokens"] += metrics["total_tokens"]
    agent_metrics["diagnosis_agent"]["num_calls"] += 1

    # Increment round counter for next iteration
    current_round = state.get("current_round", 1)

    return {
        **state,
        "error_agent": error_agent,
        "retry_count": retry_count + 1,
        "current_round": current_round + 1,  # New round starts after diagnosis
        "backtrack_history": diagnosis_history,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": state.get("total_tokens", 0) + metrics["total_tokens"],
        "total_duration_s": state.get("total_duration_s", 0.0) + metrics["duration_s"],
        "inspection_policy": inspection_policy,
        "probe_queue": probe_queue,
        "cleared_layers": cleared_layers,
        "refutation_log": refutation_log,
    }


def _stackelberg_diagnosis(
    state: Dict,
    error_category: str,
    gurobi_status: str,
    error_info: str,
) -> tuple:
    """Stackelberg inspection: commit σ and select the next layer to probe.

    Evidence-informed default: rank layers from stack+history, align ω with
    probe_order ablation, commit once, then probe uncleared layers. Legacy
    ``omega_source=status_prior`` keeps Prior(status) seeding for A/B only.

    Returns:
        (error_agent, confidence, reason, metrics,
         inspection_policy, probe_queue, cleared_layers, refutation_log)
    """
    from ..game.ranking import align_omega, rank_layers
    from ..game.verified import make_refutation_entry

    start_time = time.time()
    metrics = {"total_tokens": 0, "duration_s": 0.0}

    probe_order_mode = state.get("probe_order", "causal")
    omega_source = state.get("omega_source") or "evidence_rank"
    rank_method = state.get("rank_method") or "llm_rank"
    probe_queue = list(state.get("probe_queue") or [])
    cleared_layers = list(state.get("cleared_layers") or [])
    inspection_policy = state.get("inspection_policy")
    refutation_log = list(state.get("refutation_log") or [])

    # Syntax errors are localized to code generation — probe PD directly.
    if error_category == "syntax" or (gurobi_status == "" and error_category == "syntax"):
        logger.info("DiagnosisAgent (stackelberg): syntax error → probe python_developer")
        if not inspection_policy:
            inspection_policy = {
                "omega": ["python_developer"],
                "nu": VERIFICATION_RULE,
                "probe_order": probe_order_mode,
                "omega_source": omega_source,
                "rank": ["python_developer", "model_expert", "data_engineer"],
                "rank_method": "syntax_shortcut",
                "rank_rationale": "syntax errors localize to python_developer",
                "seed": "python_developer",
                "committed_at_round": state.get("current_round"),
            }
            probe_queue = ["python_developer"]
        metrics["duration_s"] = time.time() - start_time
        state["refutation_log"] = refutation_log
        return (
            "python_developer",
            1.0,
            "Committed probe: syntax errors localize to python_developer",
            metrics,
            inspection_policy,
            probe_queue,
            cleared_layers,
            refutation_log,
        )

    # First entry into this failure episode: commit inspection policy σ.
    if not inspection_policy or not probe_queue:
        probe_seed = state.get("probe_seed")
        rank_result = None
        rank_list: List[str] = []
        rank_rationale = ""
        rank_method_used = rank_method

        if omega_source == "evidence_rank" and probe_order_mode == "causal":
            # Rank only needed for causal tip; reverse/random ignore rank (Exp-A).
            rank_result = rank_layers(state, method=rank_method)
            rank_list = list(rank_result.rank)
            rank_rationale = rank_result.rationale
            rank_method_used = rank_result.method
            metrics["total_tokens"] = int(rank_result.tokens or 0)
        elif omega_source == "evidence_rank":
            # reverse/random: still record a placeholder rank for auditability
            # without spending LLM tokens (pure order ablation).
            rank_list = list(CAUSAL_LAYERS)
            rank_method_used = "ablation_ignored"
            rank_rationale = (
                f"probe_order={probe_order_mode} ignores rank tip (Exp-A freeze)"
            )
        else:
            # status_prior: optional heuristic rank for logs only
            rank_list = get_candidate_agents(gurobi_status)
            rank_method_used = "status_prior"
            rank_rationale = "legacy Prior(status) seed; no evidence rank"

        probe_queue = align_omega(
            probe_order=probe_order_mode,
            rank=rank_list,
            gurobi_status=gurobi_status,
            omega_source=omega_source,
            probe_seed=probe_seed,
        )
        seed = probe_queue[0] if probe_queue else "model_expert"
        inspection_policy = {
            "omega": list(probe_queue),
            "nu": VERIFICATION_RULE,
            "probe_order": probe_order_mode,
            "omega_source": omega_source,
            "rank": list(rank_list),
            "rank_method": rank_method_used,
            "rank_rationale": rank_rationale,
            "seed": seed,
            "probe_seed": probe_seed,
            "committed_at_round": state.get("current_round"),
            "prior_candidates": get_candidate_agents(gurobi_status),
            "prior_rotated": (
                probe_order_mode == "causal" and omega_source == "status_prior"
            ),
        }
        if rank_result is not None:
            inspection_policy["rank_raw"] = rank_result.raw
        cleared_layers = []
        logger.info(
            "DiagnosisAgent (stackelberg): committed σ omega=%s nu=%s seed=%s "
            "probe_order=%s omega_source=%s rank=%s rank_method=%s probe_seed=%s",
            probe_queue,
            VERIFICATION_RULE,
            seed,
            probe_order_mode,
            omega_source,
            rank_list,
            rank_method_used,
            probe_seed,
        )
    else:
        # Re-entry: previous probe was settled.
        # - comply whose re-solve still failed → false confession (cleared)
        # - deflect → symptom-scoped clear (advance ω); may reopen later if the
        #   surface symptom changes while formulation evidence remains
        #   (see game.clear_policy).
        from ..game.clear_policy import (
            reopen_symptom_scoped_deflects,
            symptom_fingerprint,
        )

        last_agent = state.get("error_agent")
        if last_agent and last_agent not in cleared_layers:
            cleared_layers.append(last_agent)
            complied = bool(state.get("error_resolved"))
            action = "comply" if complied else "deflect"
            # Re-entering diagnosis ⇒ re-solve did not reach Practical Optimal
            # (or no re-solve on deflect).
            entry = make_refutation_entry(
                layer=last_agent,
                action=action,
                error_resolved=complied,
                re_solve_strict_success=False,
                retry_index=int(state.get("retry_count") or 0),
                overturn=complied,  # claimed repair but still failing
                symptom_fingerprint=symptom_fingerprint(state),
            )
            refutation_log.append(entry)
            state["refutation_log"] = refutation_log
            logger.info(
                "DiagnosisAgent (stackelberg): cleared layer=%s action=%s "
                "overturn=%s symptom=%s",
                last_agent,
                action,
                entry["overturn"],
                entry.get("symptom_fingerprint"),
            )

        cleared_layers = reopen_symptom_scoped_deflects(
            cleared_layers, refutation_log, state
        )

    error_agent = next_unclear_layer(probe_queue, cleared_layers)
    if error_agent is None:
        logger.warning("DiagnosisAgent (stackelberg): all layers cleared, no probe left")
        metrics["duration_s"] = time.time() - start_time
        state["refutation_log"] = refutation_log
        return (
            None,
            0.0,
            "Inspection exhausted: all causal layers cleared without confirmation",
            metrics,
            inspection_policy,
            probe_queue,
            cleared_layers,
            refutation_log,
        )

    reason = (
        f"Inspector probe under committed σ: layer={error_agent}, "
        f"omega={probe_queue}, cleared={cleared_layers}, nu={VERIFICATION_RULE}, "
        f"omega_source={omega_source}"
    )
    metrics["duration_s"] = time.time() - start_time
    logger.info("DiagnosisAgent (stackelberg): probing %s", error_agent)
    state["refutation_log"] = refutation_log
    return (
        error_agent,
        1.0,
        reason,
        metrics,
        inspection_policy,
        probe_queue,
        cleared_layers,
        refutation_log,
    )


def _adversarial_diagnosis(
    state: Dict,
    error_category: str,
    gurobi_status: str,
    error_info: str,
) -> tuple:
    """Adversarial diagnosis: use LLM to accuse the most likely responsible agent.

    Args:
        state: 当前状态
        error_category: 错误类型
        gurobi_status: Gurobi 求解状态
        error_info: 错误详情

    Returns:
        (error_agent, confidence, reason, metrics_dict)
    """
    start_time = time.time()
    metrics = {"total_tokens": 0, "duration_s": 0.0}

    # 根据求解器状态确定候选 Agent
    candidate_agents = get_candidate_agents(gurobi_status)
    logger.info(f"DiagnosisAgent (adversarial): candidate_agents={candidate_agents}")

    # 如果是语法错误，直接返回 python_developer（无需 LLM 诊断）
    if error_category == "syntax" or (gurobi_status == "" and error_category == "syntax"):
        logger.info("DiagnosisAgent: Syntax error detected, directly assigning to python_developer")
        metrics["duration_s"] = time.time() - start_time
        return "python_developer", 1.0, "Syntax errors are always caused by code generation issues", metrics

    # 构建完整的诊断上下文
    context = _build_diagnosis_context(state)

    # 记录错误键分析结果
    if context.get("error_key_analysis", {}).get("parsed"):
        logger.info(f"DiagnosisAgent: error_key_analysis={context['error_key_analysis']}")

    # 需要 LLM 诊断
    chain = create_diagnosis_agent(
        provider=state.get("provider"),
        model=state.get("model"),
    )

    # Helper function to escape curly braces for .format()
    def _escape_braces(s: str) -> str:
        """Escape curly braces in string to prevent .format() placeholder errors."""
        return s.replace("{", "{{").replace("}", "}}")

    # Format the prompt with enhanced context
    task = DIAGNOSIS_PROMPT.format(
        gurobi_status=gurobi_status,
        stack_trace=state.get("stack_trace", "Not available"),
        error_location=_escape_braces(json.dumps(context.get("error_location", {}), ensure_ascii=False, indent=2)),
        python_code=context.get("agent_outputs", {}).get("python_developer", "Not available"),
        model_expert_output=context.get("agent_outputs", {}).get("model_expert", "Not available"),
        data_access_guide=_escape_braces(json.dumps(context.get("data_access_guide", {}), ensure_ascii=False, indent=2)),
        error_key_analysis=_escape_braces(json.dumps(context.get("error_key_analysis", {}), ensure_ascii=False, indent=2)),
        candidate_agents=candidate_agents,
    )

    # Execute diagnosis with metrics tracking
    try:
        with get_openai_callback() as cb:
            diagnosis: DiagnosisOutput = chain.invoke({
                "role": DIAGNOSIS_ROLE,
                "task": task,
            })
        metrics["total_tokens"] = cb.total_tokens

        error_agent = diagnosis.suspected_agent
        confidence = diagnosis.confidence
        reason = diagnosis.reason

        # Validate suspected agent is in candidates
        if error_agent not in candidate_agents:
            logger.warning(f"LLM returned invalid agent '{error_agent}', using first candidate")
            error_agent = candidate_agents[0]
            confidence = 0.5
            reason = f"Invalid diagnosis, defaulting to {error_agent}"

    except Exception as e:
        logger.error(f"DiagnosisAgent diagnosis failed: {e}")
        error_agent = candidate_agents[0]
        confidence = 0.5
        reason = f"Diagnosis failed: {str(e)}"

    metrics["duration_s"] = time.time() - start_time
    logger.info(f"DiagnosisAgent: tokens={metrics['total_tokens']}, duration={metrics['duration_s']:.2f}s")

    return error_agent, confidence, reason, metrics


def _build_agent_outputs(state: Dict) -> Dict:
    """构建所有 Agent 的输出字典（完整传递，不截断）.

    Args:
        state: 当前状态

    Returns:
        Agent 输出字典
    """
    agent_outputs = {}

    # Python 代码: 完整传递，不再截断
    if state.get("python_code"):
        agent_outputs["python_developer"] = state["python_code"]

    # ModelExpert 输出: 完整传递
    if state.get("model_expert_output"):
        me_output = state["model_expert_output"]
        if hasattr(me_output, 'model_dump_json'):
            agent_outputs["model_expert"] = me_output.model_dump_json(indent=2)
        else:
            agent_outputs["model_expert"] = json.dumps(me_output, ensure_ascii=False, indent=2)

    # DataEngineer 输出: 完整传递
    if state.get("data_engineer_output"):
        de_output = state["data_engineer_output"]
        if hasattr(de_output, 'model_dump_json'):
            agent_outputs["data_engineer"] = de_output.model_dump_json(indent=2)
        else:
            agent_outputs["data_engineer"] = json.dumps(de_output, ensure_ascii=False, indent=2)

    return agent_outputs


def _build_diagnosis_context(state: Dict) -> Dict[str, Any]:
    """构建完整的诊断上下文，包含错误分析和数据结构信息.

    Args:
        state: 当前状态

    Returns:
        诊断上下文字典
    """
    context = {}

    # 1. 解析并分析错误键
    gurobi_status = state.get("gurobi_status", "")
    error_key = parse_error_key(gurobi_status)
    context["error_key_analysis"] = analyze_error_key(error_key)

    # 2. 获取数据结构摘要
    sample = state.get("sample", {})
    context["data_summary"] = get_data_keys_summary(sample)

    # 3. 完整数据访问指南（和 PythonDeveloper 看到的一样）
    if state.get("data_access_guide"):
        context["data_access_guide"] = state["data_access_guide"]

    # 4. Agent 输出
    context["agent_outputs"] = _build_agent_outputs(state)

    # 5. 代码级错误定位
    stack_trace = state.get("stack_trace")
    python_code = state.get("python_code", "")
    context["error_location"] = extract_error_context(stack_trace, python_code)

    return context


def _get_agent_output(state: Dict, agent_name: str) -> Optional[str]:
    """获取指定 Agent 的输出.

    Args:
        state: 当前状态
        agent_name: Agent 名称

    Returns:
        Agent 输出字符串
    """
    if agent_name == "data_engineer":
        output = state.get("data_engineer_output")
        if output and hasattr(output, 'model_dump_json'):
            return output.model_dump_json(indent=2)
        return str(output) if output else None

    elif agent_name == "model_expert":
        output = state.get("model_expert_output")
        if output and hasattr(output, 'model_dump_json'):
            return output.model_dump_json(indent=2)
        return str(output) if output else None

    elif agent_name == "python_developer":
        return state.get("python_code")

    return None
