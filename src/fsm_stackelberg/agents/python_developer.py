"""PythonDeveloper Agent - LangChain implementation.

Generates Gurobi optimization code from mathematical model specifications.
"""

import json
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

logger = logging.getLogger(__name__)


def _build_data_access_guide(data_engineer_output: DataEngineerOutput, schema: Dict = None) -> Dict:
    """Build the data access guide from DataEngineer output.

    Key change: Use symbol as the key for direct lookup by PythonDeveloper.

    This guide tells the Python developer how to access data without
    exposing actual data values.
    """
    guide = {
        "sets": {},
        "parameters": {},
        "indexing_convention": {},  # NEW: Pass through _meta indexing info
    }

    # Extract indexing convention from schema._meta if available
    if schema and "_meta" in schema:
        meta = schema["_meta"]
        # Extract period indexing info
        if "sets" in meta and "periods" in meta["sets"]:
            periods_meta = meta["sets"]["periods"]
            guide["indexing_convention"]["periods"] = {
                "start": periods_meta.get("start", 1),
                "description": periods_meta.get("desc", ""),
            }
        # Extract initial_inventory time reference
        if "parameters" in meta and "initial_inventory" in meta["parameters"]:
            inv_meta = meta["parameters"]["initial_inventory"]
            guide["indexing_convention"]["initial_inventory"] = {
                "time_reference": inv_meta.get("time_reference", 0),
                "description": inv_meta.get("desc", ""),
            }
        # Extract transit_time boundary rule
        if "parameters" in meta and "transit_time_matrix" in meta["parameters"]:
            tt_meta = meta["parameters"]["transit_time_matrix"]
            guide["indexing_convention"]["transit_time_matrix"] = {
                "description": tt_meta.get("desc", ""),
            }

    # Add sets - use symbol as key for direct lookup
    for set_def in data_engineer_output.model_inputs.sets:
        guide["sets"][set_def.symbol] = {
            "source": set_def.source,
            "description": set_def.description,
            "access_pattern": f"data['{set_def.source}']",
            "index": set_def.index,
        }

    # Add parameters - use symbol as key for direct lookup
    for param_def in data_engineer_output.model_inputs.parameters:
        source_key = param_def.source or param_def.symbol
        indices = param_def.indices

        # Build access pattern with tuple key format for multi-dimensional data
        if len(indices) == 0:
            access_pattern = f"data['{source_key}']"
        elif len(indices) == 1:
            idx_var = indices[0].symbol
            access_pattern = f"data['{source_key}'][{idx_var}]"
        else:
            # Multi-dimensional: use tuple key format data[key][(i, j, k)]
            idx_vars = ", ".join(idx.symbol for idx in indices)
            access_pattern = f"data['{source_key}'][({idx_vars})]"

        guide["parameters"][param_def.symbol] = {
            "source": source_key,
            "description": param_def.description,
            "indices": [{"symbol": idx.symbol, "set": idx.set} for idx in indices],
            "access_pattern": access_pattern,
            "sparse": param_def.sparse,
            "access_hint": param_def.access_hint,
        }

    return guide


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
    schema = state.get("schema", {})  # Get schema for _meta info

    if not model_expert_output:
        raise ValueError("Missing required output from ModelExpert")

    # Use auto-generated data_access_guide from state (preferred),
    # fall back to DE-derived guide for backward compatibility
    data_access_guide = state.get("data_access_guide")
    if not data_access_guide:
        if data_engineer_output:
            data_access_guide = json.dumps(_build_data_access_guide(data_engineer_output, schema), indent=2)
        else:
            data_access_guide = "No data access guide available"

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

    logger.info(f"PythonDeveloper: Generated code ({len(python_code)} chars)")
    logger.info(f"PythonDeveloper: tokens={cb.total_tokens}, duration={duration:.2f}s")

    # Update metrics
    step_metrics = state.get("step_metrics", [])
    step_metrics.append({
        "node": "python_developer",
        "step_type": "forward",
        "duration_s": round(duration, 3),
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
    })

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
        "data_access_guide": data_access_guide,  # 传递给后续节点
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
    schema = state.get("schema", {})  # Get schema for _meta info

    model_blueprint = model_expert_output.model_dump_json(indent=2) if model_expert_output else "N/A"
    data_access_guide = _build_data_access_guide(data_engineer_output, schema) if data_engineer_output else {}

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
        data_access_guide=json.dumps(data_access_guide, indent=2),
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

    # Update metrics
    step_metrics = state.get("step_metrics", [])
    step_metrics.append({
        "node": "python_developer",
        "step_type": "backward",
        "duration_s": round(duration, 3),
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
    })

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
