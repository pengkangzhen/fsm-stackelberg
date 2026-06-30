"""Knowledge loader node for LangGraph workflow.

Handles knowledge requests from ModelExpert and loads domain knowledge modules.
"""

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

MAX_KNOWLEDGE_ROUNDS = 2  # Maximum successful knowledge-loading refinements allowed


def knowledge_loader_node(state: Dict) -> Dict:
    """Load requested knowledge modules and update state.

    This node is called when ModelExpert requests domain knowledge.
    It loads the requested modules and stores them in state for the next
    ModelExpert invocation.

    Args:
        state: Current state with:
            - model_expert_output: Contains knowledge_requests list
            - knowledge_loader: KnowledgeLoader instance
            - knowledge_round: Current round number

    Returns:
        Updated state with:
            - loaded_knowledge: Content of loaded modules
            - knowledge_round: Incremented round counter
    """
    start_time = time.time()
    model_expert_output = state.get("model_expert_output")
    knowledge_loader = state.get("knowledge_loader")
    loaded_modules = state.get("loaded_knowledge_modules", [])
    excluded_modules = set(state.get("knowledge_excluded_modules", []))

    # Guard: skip when knowledge is disabled
    if knowledge_loader is None:
        logger.info("Knowledge injection disabled, skipping loader node")
        return {
            "knowledge_loader_loaded": False,
            "step_metrics": state.get("step_metrics", []),
            "node_metrics": state.get("node_metrics", {}),
            "agent_metrics": state.get("agent_metrics", {}),
            "total_tokens": state.get("total_tokens", 0),
            "total_duration_s": state.get("total_duration_s", 0.0),
        }

    # Check if there are knowledge requests
    if not model_expert_output or not model_expert_output.knowledge_requests:
        logger.info("No knowledge requests from ModelExpert")
        return {
            "knowledge_loader_loaded": False,
            "step_metrics": state.get("step_metrics", []),
            "node_metrics": state.get("node_metrics", {}),
            "agent_metrics": state.get("agent_metrics", {}),
            "total_tokens": state.get("total_tokens", 0),
            "total_duration_s": state.get("total_duration_s", 0.0),
        }

    # Check round limit
    knowledge_round = state.get("knowledge_round", 0) + 1
    if knowledge_round > MAX_KNOWLEDGE_ROUNDS:
        logger.warning(f"Max knowledge rounds ({MAX_KNOWLEDGE_ROUNDS}) reached, stopping refinement")
        step_metrics = state.get("step_metrics", [])
        step_metrics.append({
            "node": "knowledge_loader",
            "step_type": "guardrail",
            "duration_s": round(time.time() - start_time, 3),
            "total_tokens": 0,
        })
        node_metrics = state.get("node_metrics", {})
        if "knowledge_loader" not in node_metrics:
            node_metrics["knowledge_loader"] = {
                "total_duration_s": 0.0,
                "total_tokens": 0,
                "num_calls": 0,
            }
        node_metrics["knowledge_loader"]["total_duration_s"] += time.time() - start_time
        node_metrics["knowledge_loader"]["num_calls"] += 1
        return {
            "knowledge_round": knowledge_round,
            "knowledge_loader_loaded": False,
            "step_metrics": step_metrics,
            "node_metrics": node_metrics,
            "agent_metrics": state.get("agent_metrics", {}),
            "total_tokens": state.get("total_tokens", 0),
            "total_duration_s": state.get("total_duration_s", 0.0) + (time.time() - start_time),
        }

    requested_modules = model_expert_output.knowledge_requests
    blocked_modules = [name for name in requested_modules if name in excluded_modules]
    if blocked_modules:
        logger.info("Knowledge requests blocked by ablation profile: %s", blocked_modules)
    pending_modules = [
        name for name in requested_modules
        if name not in loaded_modules and name not in excluded_modules
    ]
    if not pending_modules:
        if blocked_modules:
            logger.info("No loadable knowledge requests remain after ablation filtering: %s", requested_modules)
        else:
            logger.info("All requested knowledge modules are already loaded: %s", requested_modules)
        step_metrics = state.get("step_metrics", [])
        step_metrics.append({
            "node": "knowledge_loader",
            "step_type": "guardrail",
            "duration_s": round(time.time() - start_time, 3),
            "total_tokens": 0,
        })
        node_metrics = state.get("node_metrics", {})
        if "knowledge_loader" not in node_metrics:
            node_metrics["knowledge_loader"] = {
                "total_duration_s": 0.0,
                "total_tokens": 0,
                "num_calls": 0,
            }
        node_metrics["knowledge_loader"]["total_duration_s"] += time.time() - start_time
        node_metrics["knowledge_loader"]["num_calls"] += 1
        return {
            "knowledge_loader_loaded": False,
            "step_metrics": step_metrics,
            "node_metrics": node_metrics,
            "agent_metrics": state.get("agent_metrics", {}),
            "total_tokens": state.get("total_tokens", 0),
            "total_duration_s": state.get("total_duration_s", 0.0) + (time.time() - start_time),
        }

    logger.info(f"Loading knowledge modules (round {knowledge_round}): {pending_modules}")

    # Load knowledge content
    knowledge_content = knowledge_loader.get_knowledge_by_names(
        pending_modules,
        excluded_modules=excluded_modules,
    )

    if knowledge_content:
        logger.info(f"Successfully loaded {len(pending_modules)} knowledge module(s)")
    else:
        logger.warning(f"Failed to load requested modules: {pending_modules}")

    existing_knowledge = state.get("loaded_knowledge")
    if existing_knowledge and knowledge_content:
        merged_knowledge = f"{existing_knowledge}\n\n---\n\n{knowledge_content}"
    else:
        merged_knowledge = knowledge_content or existing_knowledge

    duration = time.time() - start_time
    step_metrics = state.get("step_metrics", [])
    step_metrics.append({
        "node": "knowledge_loader",
        "step_type": "guardrail",
        "duration_s": round(duration, 3),
        "total_tokens": 0,
    })
    node_metrics = state.get("node_metrics", {})
    if "knowledge_loader" not in node_metrics:
        node_metrics["knowledge_loader"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    node_metrics["knowledge_loader"]["total_duration_s"] += duration
    node_metrics["knowledge_loader"]["num_calls"] += 1

    loaded_module_names = loaded_modules + pending_modules if knowledge_content else loaded_modules

    return {
        "loaded_knowledge": merged_knowledge,
        "loaded_knowledge_modules": loaded_module_names,
        "knowledge_round": knowledge_round,
        "knowledge_loader_loaded": bool(knowledge_content),
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": state.get("agent_metrics", {}),
        "total_tokens": state.get("total_tokens", 0),
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
    }
