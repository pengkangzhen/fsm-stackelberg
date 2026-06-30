"""DataEngineer Agent - LangChain implementation.

Prepares analysis-ready datasets for optimization modeling.
"""

import json
import logging
import time
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback

from ..schemas import DataEngineerOutput, BackwardStepOutput
from ..utils.llm_config import DEFAULT_PROVIDER, DEFAULT_MODEL
from ..prompts import (
    DATA_ENGINEER_ROLE,
    DATA_ENGINEER_FORWARD,
    DATA_ENGINEER_BACKWARD_STEP,
    get_data_engineer_prompt,
)
from ..utils.llm_config import get_llm
from ..utils.utils import record_agent_output, get_last_forward_output
from ..utils.workflow_failure import WorkflowNodeError, build_failure_state

logger = logging.getLogger(__name__)


def create_data_engineer(provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Create a DataEngineer agent with structured output.

    Args:
        provider: LLM provider (default: DeepSeek)
        model: The model to use (default: deepseek-chat)

    Returns:
        A LangChain chain that produces DataEngineerOutput
    """
    llm = get_llm(provider=provider, model=model, temperature=0)

    prompt = get_data_engineer_prompt()

    chain = prompt | llm.with_structured_output(DataEngineerOutput, method="json_mode")

    return chain


def data_engineer_node(state: Dict) -> Dict:
    """LangGraph node function for DataEngineer.

    Args:
        state: The current graph state containing problem_description and schema

    Returns:
        Updated state with data_engineer_output
    """
    logger.info("DataEngineer: Starting forward step...")
    start_time = time.time()

    # Extract inputs from state
    problem_description = state["problem_description"]
    data_access_guide = state.get("data_access_guide", "")

    # Create the chain
    chain = create_data_engineer(
        provider=state.get("provider"),
        model=state.get("model"),
    )

    # Format the prompt — DE now receives data_access_guide instead of raw schema
    task = DATA_ENGINEER_FORWARD.format(
        problem_description=problem_description,
        data_access_guide=data_access_guide,
    )

    # Execute with metrics tracking
    cb = None
    try:
        with get_openai_callback() as cb:
            result: DataEngineerOutput = chain.invoke({
                "role": DATA_ENGINEER_ROLE,
                "task": task,
            })
    except Exception as exc:
        logger.exception("DataEngineer failed during forward step.")
        raise WorkflowNodeError(
            "DataEngineer forward step failed",
            build_failure_state(
                state,
                "data_engineer",
                exc,
                start_time,
                prompt_tokens=cb.prompt_tokens if cb else 0,
                completion_tokens=cb.completion_tokens if cb else 0,
                total_tokens=cb.total_tokens if cb else 0,
            ),
        ) from exc

    duration = time.time() - start_time

    logger.info(f"DataEngineer: Produced {len(result.model_inputs.sets)} sets, "
                f"{len(result.model_inputs.parameters)} parameters")
    logger.info(f"DataEngineer: tokens={cb.total_tokens}, duration={duration:.2f}s")

    # Update metrics
    step_metrics = state.get("step_metrics", [])
    step_metrics.append({
        "node": "data_engineer",
        "step_type": "forward",
        "duration_s": round(duration, 3),
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
    })

    node_metrics = state.get("node_metrics", {})
    if "data_engineer" not in node_metrics:
        node_metrics["data_engineer"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    node_metrics["data_engineer"]["total_duration_s"] += duration
    node_metrics["data_engineer"]["total_tokens"] += cb.total_tokens
    node_metrics["data_engineer"]["num_calls"] += 1

    agent_metrics = state.get("agent_metrics", {})
    if "data_engineer" not in agent_metrics:
        agent_metrics["data_engineer"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    agent_metrics["data_engineer"]["total_duration_s"] += duration
    agent_metrics["data_engineer"]["total_tokens"] += cb.total_tokens
    agent_metrics["data_engineer"]["num_calls"] += 1

    # Record output to history
    current_round = state.get("current_round", 1)
    output_history = record_agent_output(
        output_history=state.get("output_history", []),
        agent_name="data_engineer",
        output=result,
        current_round=current_round,
        step_type="forward"
    )

    return {
        "data_engineer_output": result,
        "output_history": output_history,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
        "retry_count": state.get("retry_count", 0),
        "current_round": current_round,
    }


def data_engineer_backward_step(state: Dict) -> Dict:
    """Execute backward step: judge error and provide correction.

    Used by both adversarial and sequential diagnosis modes.
    The agent analyzes the error, determines if it caused the error,
    and provides corrected data mapping if responsible.

    Args:
        state: Current graph state with error_info, gurobi_status, output_history

    Returns:
        Updated state with:
        - error_resolved: True if agent admitted fault and provided correction
        - data_engineer_output: Corrected output if error_resolved
        - backward_reason: Agent's explanation
    """
    logger.info("DataEngineer: Starting backward step...")
    start_time = time.time()

    # Extract error context
    error_info = state.get("error_info", "")
    gurobi_status = state.get("gurobi_status", "")
    full_error = f"Gurobi Status: {gurobi_status}\n{error_info}" if gurobi_status else error_info

    # Get previous outputs
    problem_description = state.get("problem_description", "")
    schema = state.get("schema", state.get("sample", {}))

    output_history = state.get("output_history", [])
    previous_output = get_last_forward_output(output_history, "data_engineer")

    # Create chain for backward step
    llm = get_llm(provider=state.get("provider"), model=state.get("model"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{role}"),
        ("human", "{task}"),
    ])
    chain = prompt | llm.with_structured_output(BackwardStepOutput, method="json_mode")

    # Format the prompt
    task = DATA_ENGINEER_BACKWARD_STEP.format(
        problem_description=problem_description,
        schema=json.dumps(schema, indent=2, ensure_ascii=False) if isinstance(schema, dict) else str(schema),
        error_info=full_error,
        previous_output=previous_output,
    )

    # Execute
    with get_openai_callback() as cb:
        result: BackwardStepOutput = chain.invoke({
            "role": DATA_ENGINEER_ROLE,
            "task": task,
        })

    duration = time.time() - start_time
    logger.info(f"DataEngineer backward_step: is_caused_by_you={result.is_caused_by_you}, tokens={cb.total_tokens}")

    # Update metrics
    step_metrics = state.get("step_metrics", [])
    step_metrics.append({
        "node": "data_engineer",
        "step_type": "backward",
        "duration_s": round(duration, 3),
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
    })

    node_metrics = state.get("node_metrics", {})
    if "data_engineer" not in node_metrics:
        node_metrics["data_engineer"] = {"total_duration_s": 0.0, "total_tokens": 0, "num_calls": 0}
    node_metrics["data_engineer"]["total_duration_s"] += duration
    node_metrics["data_engineer"]["total_tokens"] += cb.total_tokens
    node_metrics["data_engineer"]["num_calls"] += 1

    agent_metrics = state.get("agent_metrics", {})
    if "data_engineer" not in agent_metrics:
        agent_metrics["data_engineer"] = {"total_duration_s": 0.0, "total_tokens": 0, "num_calls": 0}
    agent_metrics["data_engineer"]["total_duration_s"] += duration
    agent_metrics["data_engineer"]["total_tokens"] += cb.total_tokens
    agent_metrics["data_engineer"]["num_calls"] += 1

    # Handle result
    if result.is_caused_by_you and result.refined_result:
        # Agent admitted fault and provided correction
        logger.info(f"DataEngineer backward_step: Agent admitted fault. Reason: {result.reason}")

        # Parse refined result
        if isinstance(result.refined_result, dict):
            refined_output = DataEngineerOutput(**result.refined_result)
        elif isinstance(result.refined_result, str):
            try:
                refined_dict = json.loads(result.refined_result)
                refined_output = DataEngineerOutput(**refined_dict)
            except json.JSONDecodeError:
                logger.error("DataEngineer backward_step: Failed to parse refined_result as JSON")
                return {
                    "error_resolved": False,
                    "backward_reason": result.reason,
                    "step_metrics": step_metrics,
                    "node_metrics": node_metrics,
                    "agent_metrics": agent_metrics,
                    "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
                    "total_duration_s": state.get("total_duration_s", 0.0) + duration,
                }
        else:
            refined_output = result.refined_result

        # Record the correction
        current_round = state.get("current_round", 1)
        output_history = record_agent_output(
            output_history=output_history,
            agent_name="data_engineer",
            output=refined_output,
            current_round=current_round,
            step_type="backward"
        )

        return {
            "data_engineer_output": refined_output,
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
    logger.info(f"DataEngineer backward_step: Agent denied fault. Reason: {result.reason}")
    return {
        "error_resolved": False,
        "backward_reason": result.reason,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
    }
