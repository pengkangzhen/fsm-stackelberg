"""ModelExpert Agent - LangChain implementation.

Translates business problems into formal mathematical optimization models.
"""

import json
import logging
import time
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback

from ..schemas import ModelExpertOutput, BackwardStepOutput
from ..utils.llm_config import DEFAULT_PROVIDER, DEFAULT_MODEL
from ..prompts import (
    MODEL_EXPERT_ROLE,
    MODEL_EXPERT_FORWARD,
    MODEL_EXPERT_BACKWARD_STEP,
    get_model_expert_prompt,
)
from ..utils.llm_config import get_llm
from ..utils.utils import record_agent_output, get_last_forward_output
from ..utils.workflow_failure import WorkflowNodeError, build_failure_state

logger = logging.getLogger(__name__)


def _strip_satisfied_knowledge_requests(result: ModelExpertOutput, state: Dict) -> ModelExpertOutput:
    """Keep only knowledge requests that are not already loaded in state."""
    if not result.knowledge_requests:
        return result

    loaded_modules = set(state.get("loaded_knowledge_modules", []))
    pending_requests = [name for name in result.knowledge_requests if name not in loaded_modules]
    if pending_requests == result.knowledge_requests:
        return result

    logger.info(
        "ModelExpert: filtered already-loaded knowledge requests %s -> %s",
        result.knowledge_requests,
        pending_requests,
    )
    return result.model_copy(update={"knowledge_requests": pending_requests})


def create_model_expert(provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Create a ModelExpert agent with structured output.

    Args:
        provider: LLM provider (default: DeepSeek)
        model: The model to use (default: deepseek-chat)

    Returns:
        A LangChain chain that produces ModelExpertOutput
    """
    llm = get_llm(provider=provider, model=model, temperature=0)

    prompt = get_model_expert_prompt()

    chain = prompt | llm.with_structured_output(ModelExpertOutput, method="json_mode")

    return chain


def model_expert_node(state: Dict) -> Dict:
    """LangGraph node function for ModelExpert.

    Args:
        state: The current graph state containing problem_description and data_engineer_output

    Returns:
        Updated state with model_expert_output
    """
    logger.info("ModelExpert: Starting forward step...")
    start_time = time.time()

    # Extract inputs from state
    problem_description = state["problem_description"]
    data_engineer_output = state.get("data_engineer_output")

    # Format data engineer output for the prompt
    if data_engineer_output:
        de_output_str = data_engineer_output.model_dump_json(indent=2)
    else:
        de_output_str = "No data engineer output available."

    # Knowledge catalog and pre-loaded knowledge
    knowledge_catalog = state.get("knowledge_catalog", "No additional knowledge modules available.")
    loaded_knowledge = state.get("loaded_knowledge")

    # Format loaded knowledge section
    if loaded_knowledge:
        loaded_knowledge_section = f"""---

## Pre-loaded Domain Knowledge

**IMPORTANT**: The following domain knowledge has been pre-loaded for this problem. **You MUST apply these rules when defining constraints and variables.**

{loaded_knowledge}

---"""
    else:
        loaded_knowledge_section = ""

    # Create the chain
    chain = create_model_expert(
        provider=state.get("provider"),
        model=state.get("model"),
    )

    # Format the prompt
    task = MODEL_EXPERT_FORWARD.format(
        problem_description=problem_description,
        data_engineer_output=de_output_str,
        knowledge_catalog=knowledge_catalog,
        loaded_knowledge_section=loaded_knowledge_section,
    )

    # Execute with metrics tracking
    cb = None
    try:
        with get_openai_callback() as cb:
            result: ModelExpertOutput = chain.invoke({
                "role": MODEL_EXPERT_ROLE,
                "task": task,
            })
    except Exception as exc:
        logger.exception("ModelExpert failed during forward step.")
        raise WorkflowNodeError(
            "ModelExpert forward step failed",
            build_failure_state(
                state,
                "model_expert",
                exc,
                start_time,
                prompt_tokens=cb.prompt_tokens if cb else 0,
                completion_tokens=cb.completion_tokens if cb else 0,
                total_tokens=cb.total_tokens if cb else 0,
            ),
        ) from exc
    result = _strip_satisfied_knowledge_requests(result, state)

    duration = time.time() - start_time

    logger.info(f"ModelExpert: Produced model with {len(result.model_components.decision_variables)} variables, "
                f"{len(result.model_components.constraints)} constraints")
    logger.info(f"ModelExpert: tokens={cb.total_tokens}, duration={duration:.2f}s")

    # Check for knowledge requests
    if result.knowledge_requests:
        logger.info(f"ModelExpert: Requesting knowledge modules: {result.knowledge_requests}")

    # Update metrics
    step_metrics = state.get("step_metrics", [])
    step_metrics.append({
        "node": "model_expert",
        "step_type": "forward",
        "duration_s": round(duration, 3),
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
    })

    node_metrics = state.get("node_metrics", {})
    if "model_expert" not in node_metrics:
        node_metrics["model_expert"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    node_metrics["model_expert"]["total_duration_s"] += duration
    node_metrics["model_expert"]["total_tokens"] += cb.total_tokens
    node_metrics["model_expert"]["num_calls"] += 1

    agent_metrics = state.get("agent_metrics", {})
    if "model_expert" not in agent_metrics:
        agent_metrics["model_expert"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    agent_metrics["model_expert"]["total_duration_s"] += duration
    agent_metrics["model_expert"]["total_tokens"] += cb.total_tokens
    agent_metrics["model_expert"]["num_calls"] += 1

    # Record output to history
    current_round = state.get("current_round", 1)
    output_history = record_agent_output(
        output_history=state.get("output_history", []),
        agent_name="model_expert",
        output=result,
        current_round=current_round,
        step_type="forward"
    )

    return {
        "model_expert_output": result,
        "data_engineer_output": data_engineer_output,  # Preserve previous output
        "output_history": output_history,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
        "retry_count": state.get("retry_count", 0),
        "current_round": current_round,
    }


def model_expert_backward_step(state: Dict) -> Dict:
    """Execute backward step: judge error and provide correction.

    Used by both adversarial and sequential diagnosis modes.
    The agent analyzes the error, determines if it caused the error,
    and provides a corrected model definition if responsible.

    Args:
        state: Current graph state with error_info, gurobi_status, output_history

    Returns:
        Updated state with:
        - error_resolved: True if agent admitted fault and provided correction
        - model_expert_output: Corrected output if error_resolved
        - backward_reason: Agent's explanation
    """
    logger.info("ModelExpert: Starting backward step...")
    start_time = time.time()

    # Extract error context
    error_info = state.get("error_info", "")
    gurobi_status = state.get("gurobi_status", "")
    full_error = f"Gurobi Status: {gurobi_status}\n{error_info}" if gurobi_status else error_info

    # Get previous outputs
    problem_description = state.get("problem_description", "")
    data_engineer_output = state.get("data_engineer_output")
    de_output_str = data_engineer_output.model_dump_json(indent=2) if data_engineer_output else "N/A"

    output_history = state.get("output_history", [])
    previous_output = get_last_forward_output(output_history, "model_expert")

    # Create chain for backward step
    llm = get_llm(provider=state.get("provider"), model=state.get("model"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{role}"),
        ("human", "{task}"),
    ])
    chain = prompt | llm.with_structured_output(BackwardStepOutput, method="json_mode")

    # Format the prompt
    task = MODEL_EXPERT_BACKWARD_STEP.format(
        problem_description=problem_description,
        data_engineer_output=de_output_str,
        error_info=full_error,
        previous_output=previous_output,
    )

    # Execute
    with get_openai_callback() as cb:
        result: BackwardStepOutput = chain.invoke({
            "role": MODEL_EXPERT_ROLE,
            "task": task,
        })

    duration = time.time() - start_time
    logger.info(f"ModelExpert backward_step: is_caused_by_you={result.is_caused_by_you}, tokens={cb.total_tokens}")

    # Update metrics
    step_metrics = state.get("step_metrics", [])
    step_metrics.append({
        "node": "model_expert",
        "step_type": "backward",
        "duration_s": round(duration, 3),
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_tokens": cb.total_tokens,
    })

    node_metrics = state.get("node_metrics", {})
    if "model_expert" not in node_metrics:
        node_metrics["model_expert"] = {"total_duration_s": 0.0, "total_tokens": 0, "num_calls": 0}
    node_metrics["model_expert"]["total_duration_s"] += duration
    node_metrics["model_expert"]["total_tokens"] += cb.total_tokens
    node_metrics["model_expert"]["num_calls"] += 1

    agent_metrics = state.get("agent_metrics", {})
    if "model_expert" not in agent_metrics:
        agent_metrics["model_expert"] = {"total_duration_s": 0.0, "total_tokens": 0, "num_calls": 0}
    agent_metrics["model_expert"]["total_duration_s"] += duration
    agent_metrics["model_expert"]["total_tokens"] += cb.total_tokens
    agent_metrics["model_expert"]["num_calls"] += 1

    # Handle result
    if result.is_caused_by_you and result.refined_result:
        # Agent admitted fault and provided correction
        logger.info(f"ModelExpert backward_step: Agent admitted fault. Reason: {result.reason}")

        # Parse refined result
        if isinstance(result.refined_result, dict):
            refined_dict = result.refined_result
        elif isinstance(result.refined_result, str):
            # Try to parse as JSON
            try:
                refined_dict = json.loads(result.refined_result)
            except json.JSONDecodeError:
                logger.error("ModelExpert backward_step: Failed to parse refined_result as JSON")
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
            refined_dict = result.refined_result.model_dump()

        # Normalize flat LLM output to nested ModelExpertOutput structure
        if "model_components" not in refined_dict:
            model_comp = {}
            if "variables" in refined_dict:
                model_comp["decision_variables"] = refined_dict.pop("variables")
            if "objective" in refined_dict:
                model_comp["objective_function"] = refined_dict.pop("objective")
            else:
                for key in list(refined_dict.keys()):
                    if "objective" in key.lower():
                        model_comp["objective_function"] = refined_dict.pop(key)
                        break
            if "constraints" in refined_dict:
                model_comp["constraints"] = refined_dict.pop("constraints")
            if model_comp:
                refined_dict["model_components"] = model_comp

        try:
            refined_output = ModelExpertOutput(**refined_dict)
        except Exception as e:
            logger.error(f"ModelExpert backward_step: Failed to parse refined_result: {e}")
            return {
                "error_resolved": False,
                "backward_reason": result.reason,
                "step_metrics": step_metrics,
                "node_metrics": node_metrics,
                "agent_metrics": agent_metrics,
                "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
                "total_duration_s": state.get("total_duration_s", 0.0) + duration,
            }
        refined_output = _strip_satisfied_knowledge_requests(refined_output, state)

        # Record the correction
        current_round = state.get("current_round", 1)
        output_history = record_agent_output(
            output_history=output_history,
            agent_name="model_expert",
            output=refined_output,
            current_round=current_round,
            step_type="backward"
        )

        return {
            "model_expert_output": refined_output,
            "data_engineer_output": data_engineer_output,
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
    logger.info(f"ModelExpert backward_step: Agent denied fault. Reason: {result.reason}")
    return {
        "error_resolved": False,
        "backward_reason": result.reason,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": agent_metrics,
        "total_tokens": state.get("total_tokens", 0) + cb.total_tokens,
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
    }
