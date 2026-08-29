"""PythonDeveloper Agent - LangChain implementation.

Generates Gurobi optimization code from mathematical model specifications.
"""

import logging
import time
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback

from ..schemas import DataEngineerOutput, ModelExpertOutput, BackwardStepOutput
from ..utils.llm_config import DEFAULT_PROVIDER, DEFAULT_MODEL
from ..prompts import (
    PYTHON_DEVELOPER_ROLE,
    PYTHON_DEVELOPER_FORWARD,
    PYTHON_DEVELOPER_BACKWARD_STEP,
    get_python_developer_prompt,
)
from ..utils.llm_config import get_llm
from ..utils.utils import record_agent_output, get_last_forward_output
from ..utils.workflow_failure import WorkflowNodeError, build_failure_state
from ..utils.run_log import record_step_event, save_backward_artifact
from ..utils.code_sanitize import unshadow_gurobi_model_m
from ..data.data_contract import (
    build_data_catalog,
    build_pd_data_access_context,
    build_symbol_access_guide,
)

logger = logging.getLogger(__name__)


def _sanitize_python_code(python_code: str) -> str:
    """Apply deterministic fixes to LLM-generated Gurobi code."""
    fixed, rewritten = unshadow_gurobi_model_m(python_code)
    if rewritten:
        logger.warning(
            "PythonDeveloper: rewrote shadowed Gurobi model `m` → `model` "
            "(mode index `m` in for-loop would otherwise break addVar/dispose)"
        )
    return fixed


def _build_data_access_guide(
    data_engineer_output: DataEngineerOutput,
    schema: Dict = None,
    catalog: Dict = None,
) -> Dict:
    """Compatibility wrapper for the deterministic symbol access guide."""
    resolved_catalog = catalog or build_data_catalog(schema or {})
    return build_symbol_access_guide(data_engineer_output, resolved_catalog)


def _resolved_pd_data_access_context(state: Dict) -> str:
    """Build the shared resolved data contract for PD forward and backward."""
    data_engineer_output = state.get("data_engineer_output")
    physical_context = state.get("data_access_guide") or ""
    if not data_engineer_output:
        return physical_context or "No data access guide available"

    catalog = state.get("data_catalog") or {}
    if not catalog:
        raw_sample = state.get("sample") or state.get("schema") or {}
        catalog = build_data_catalog(raw_sample)
    if not catalog:
        return physical_context or "No data access guide available"

    return build_pd_data_access_context(
        data_engineer_output,
        catalog,
        physical_context,
    )


def create_python_developer(provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Create a PythonDeveloper agent.

    Note: This agent outputs raw Python code, not structured JSON.

    Args:
        provider: LLM provider (default: DeepSeek)
        model: The model to use (default: deepseek-chat)

    Returns:
        A LangChain chain that produces Python code
    """
    llm = get_llm(provider=provider, model=model, temperature=0)

    prompt = get_python_developer_prompt()

    chain = prompt | llm

    return chain


def python_developer_node(state: Dict) -> Dict:
    """LangGraph node function for PythonDeveloper.

    Args:
        state: The current graph state containing data_engineer_output and model_expert_output

    Returns:
        Updated state with python_code
    """
    logger.info("PythonDeveloper: Starting forward step...")
    start_time = time.time()

    # Extract inputs from state
    data_engineer_output = state.get("data_engineer_output")
    model_expert_output = state.get("model_expert_output")
    if not model_expert_output:
        raise ValueError("Missing required output from ModelExpert")

    # Resolve DE aliases through the deterministic catalog. Schema+Compact is
    # retained only as supplementary physical context.
    data_access_guide = _resolved_pd_data_access_context(state)

    # Format model blueprint
    model_blueprint = model_expert_output.model_dump_json(indent=2)

    # Inject domain knowledge (e.g., data derivation rules with helper functions)
    loaded_knowledge = state.get("loaded_knowledge") or ""
    domain_knowledge = loaded_knowledge if loaded_knowledge else "(No domain knowledge provided — all parameters exist in data.)"

    # Create the chain
    chain = create_python_developer(
        provider=state.get("provider"),
        model=state.get("model"),
    )

    # Format the prompt — use replace() instead of .format() because the knowledge
    # module contains Python code with {dict} literals that would break .format()
    task = (
        PYTHON_DEVELOPER_FORWARD
        .replace("{model_blueprint}", model_blueprint)
        .replace("{data_access_guide}", data_access_guide)
        .replace("{domain_knowledge}", domain_knowledge)
    )

    # Execute with metrics tracking
    cb = None
    try:
        with get_openai_callback() as cb:
            result = chain.invoke({
                "role": PYTHON_DEVELOPER_ROLE,
                "task": task,
            })
    except Exception as exc:
        logger.exception("PythonDeveloper failed during forward step.")
        raise WorkflowNodeError(
            "PythonDeveloper forward step failed",
            build_failure_state(
                state,
                "python_developer",
                exc,
                start_time,
                prompt_tokens=cb.prompt_tokens if cb else 0,
                completion_tokens=cb.completion_tokens if cb else 0,
                total_tokens=cb.total_tokens if cb else 0,
            ),
        ) from exc

    duration = time.time() - start_time

    # Extract code from result
    python_code = result.content
    if isinstance(python_code, list):
        # Anthropic models return list of content blocks (thinking + text)
        text_parts = [b["text"] for b in python_code if isinstance(b, dict) and b.get("type") == "text"]
        python_code = "\n".join(text_parts)

    # Strip markdown fences if present
    if "```python" in python_code:
        python_code = python_code.split("```python")[1].split("```")[0].strip()
    elif "```" in python_code:
        python_code = python_code.split("```")[1].split("```")[0].strip()

    python_code = _sanitize_python_code(python_code)

    logger.info(f"PythonDeveloper: Generated code ({len(python_code)} chars)")
    logger.info(f"PythonDeveloper: tokens={cb.total_tokens}, duration={duration:.2f}s")

    # Update metrics
    step_metrics = record_step_event(
        state,
        node="python_developer",
        step_type="forward",
        duration_s=duration,
        prompt_tokens=cb.prompt_tokens,
        completion_tokens=cb.completion_tokens,
        total_tokens=cb.total_tokens,
        prompt_text=task,
        artifact=python_code,
        artifact_name=f"python_developer_forward_r{state.get('current_round', 1)}",
    )

    node_metrics = state.get("node_metrics", {})
    if "python_developer" not in node_metrics:
        node_metrics["python_developer"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    node_metrics["python_developer"]["total_duration_s"] += duration
    node_metrics["python_developer"]["total_tokens"] += cb.total_tokens
    node_metrics["python_developer"]["num_calls"] += 1

    agent_metrics = state.get("agent_metrics", {})
    if "python_developer" not in agent_metrics:
        agent_metrics["python_developer"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    agent_metrics["python_developer"]["total_duration_s"] += duration
    agent_metrics["python_developer"]["total_tokens"] += cb.total_tokens
    agent_metrics["python_developer"]["num_calls"] += 1

    # Record output to history
    current_round = state.get("current_round", 1)
    output_history = record_agent_output(
        output_history=state.get("output_history", []),
        agent_name="python_developer",
        output=python_code,
        current_round=current_round,
        step_type="forward"
    )

    return {
        "python_code": python_code,
        "data_engineer_output": data_engineer_output,
        # Preserve the original Schema+Compact context in state. Both PD paths
        # reconstruct the same resolved symbol guide from DE + catalog.
        "data_access_guide": state.get("data_access_guide"),
        "model_expert_output": model_expert_output,
        "output_history": output_history,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
        "retry_count": state.get("retry_count", 0),
        "current_round": current_round,
    }


def python_developer_backward_step(state: Dict) -> Dict:
    """Execute backward step: judge error and provide correction.

    Used by both adversarial and sequential diagnosis modes.
    The agent analyzes the error, determines if it caused the error,
    and provides corrected code if responsible.

    Args:
        state: Current graph state with error_info, gurobi_status, output_history

    Returns:
        Updated state with:
        - error_resolved: True if agent admitted fault and provided correction
        - python_code: Corrected code if error_resolved
        - backward_reason: Agent's explanation
    """
    logger.info("PythonDeveloper: Starting backward step...")
    start_time = time.time()

    # Extract error context
    error_info = state.get("error_info", "")
    gurobi_status = state.get("gurobi_status", "")
    full_error = f"Gurobi Status: {gurobi_status}\n{error_info}" if gurobi_status else error_info

    # Get previous outputs
    problem_description = state.get("problem_description", "")
    model_expert_output = state.get("model_expert_output")
    data_engineer_output = state.get("data_engineer_output")
    model_blueprint = model_expert_output.model_dump_json(indent=2) if model_expert_output else "N/A"
    data_access_guide_text = _resolved_pd_data_access_context(state)

    output_history = state.get("output_history", [])
    previous_output = get_last_forward_output(output_history, "python_developer")

    # Create chain for backward step
    llm = get_llm(provider=state.get("provider"), model=state.get("model"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{role}"),
        ("human", "{task}"),
    ])
    chain = prompt | llm.with_structured_output(BackwardStepOutput, method="json_mode")

    # Format the prompt
    task = PYTHON_DEVELOPER_BACKWARD_STEP.format(
        problem_description=problem_description,
        model_expert_output=model_blueprint,
        data_access_guide=data_access_guide_text,
        error_info=full_error,
        previous_output=previous_output,
    )

    # Execute
    with get_openai_callback() as cb:
        result: BackwardStepOutput = chain.invoke({
            "role": PYTHON_DEVELOPER_ROLE,
            "task": task,
        })

    duration = time.time() - start_time
    logger.info(f"PythonDeveloper backward_step: is_caused_by_you={result.is_caused_by_you}, tokens={cb.total_tokens}")

    error_resolved_preview = bool(result.is_caused_by_you and result.refined_result)
    action = "comply" if error_resolved_preview else "deflect"
    step_metrics = record_step_event(
        state,
        node="python_developer",
        step_type="backward",
        duration_s=duration,
        prompt_tokens=cb.prompt_tokens,
        completion_tokens=cb.completion_tokens,
        total_tokens=cb.total_tokens,
        prompt_text=task,
        extra={
            "is_caused_by_you": result.is_caused_by_you,
            "error_resolved": error_resolved_preview,
            "action": action,
            "backward_reason": (result.reason or "")[:500],
        },
    )
    save_backward_artifact(
        state,
        agent="python_developer",
        is_caused_by_you=bool(result.is_caused_by_you),
        error_resolved=error_resolved_preview,
        reason=result.reason or "",
        refined_present=bool(result.refined_result),
    )

    node_metrics = state.get("node_metrics", {})
    if "python_developer" not in node_metrics:
        node_metrics["python_developer"] = {"total_duration_s": 0.0, "total_tokens": 0, "num_calls": 0}
    node_metrics["python_developer"]["total_duration_s"] += duration
    node_metrics["python_developer"]["total_tokens"] += cb.total_tokens
    node_metrics["python_developer"]["num_calls"] += 1

    agent_metrics = state.get("agent_metrics", {})
    if "python_developer" not in agent_metrics:
        agent_metrics["python_developer"] = {"total_duration_s": 0.0, "total_tokens": 0, "num_calls": 0}
    agent_metrics["python_developer"]["total_duration_s"] += duration
    agent_metrics["python_developer"]["total_tokens"] += cb.total_tokens
    agent_metrics["python_developer"]["num_calls"] += 1

    # Handle result
    if result.is_caused_by_you and result.refined_result:
        # Agent admitted fault and provided correction
        logger.info(f"PythonDeveloper backward_step: Agent admitted fault. Reason: {result.reason}")

        # Extract code from refined_result
        refined_code = result.refined_result
        if isinstance(refined_code, list):
            # Anthropic models may return list of content blocks
            text_parts = [b["text"] for b in refined_code if isinstance(b, dict) and b.get("type") == "text"]
            refined_code = "\n".join(text_parts)
        if isinstance(refined_code, str):
            # Strip markdown fences if present
            if "```python" in refined_code:
                refined_code = refined_code.split("```python")[1].split("```")[0].strip()
            elif "```" in refined_code:
                refined_code = refined_code.split("```")[1].split("```")[0].strip()
        else:
            logger.error("PythonDeveloper backward_step: refined_result is not a string")
            return {
                "error_resolved": False,
                "backward_reason": result.reason,
                "step_metrics": step_metrics,
                "node_metrics": node_metrics,
                "agent_metrics": agent_metrics,
                "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
                "total_duration_s": state.get("total_duration_s", 0.0) + duration,
            }

        # Record the correction
        current_round = state.get("current_round", 1)
        output_history = record_agent_output(
            output_history=output_history,
            agent_name="python_developer",
            output=refined_code,
            current_round=current_round,
            step_type="backward"
        )

        return {
            "python_code": refined_code,
            "data_engineer_output": data_engineer_output,
            "model_expert_output": model_expert_output,
            "error_resolved": True,
            "backward_reason": result.reason,
            "output_history": output_history,
            "step_metrics": step_metrics,
            "node_metrics": node_metrics,
            "agent_metrics": agent_metrics,
            "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
            "total_duration_s": state.get("total_duration_s", 0.0) + duration,
            "current_round": current_round,
        }

    # Agent doesn't admit fault or no correction provided
    logger.info(f"PythonDeveloper backward_step: Agent denied fault. Reason: {result.reason}")
    return {
        "error_resolved": False,
        "backward_reason": result.reason,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
    }
