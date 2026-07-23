"""Knowledge loader node for LangGraph workflow.

Handles on-demand knowledge requests from ModelExpert (progressive injection).
"""

import logging
import time
from typing import Dict

from ..knowledge.progressive import DEFAULT_MAX_KNOWLEDGE_ROUNDS, ProgressiveKnowledgeInjection
from ..utils.run_log import record_step_event

logger = logging.getLogger(__name__)


def knowledge_loader_node(state: Dict) -> Dict:
    """Load requested knowledge modules and update state.

    Called when ModelExpert lists catalog module names in ``knowledge_requests``.
    Injects full module bodies and returns control to ModelExpert for refinement.
    """
    start_time = time.time()
    model_expert_output = state.get("model_expert_output")
    knowledge_loader = state.get("knowledge_loader")
    loaded_modules = list(state.get("loaded_knowledge_modules") or [])
    excluded_modules = set(state.get("knowledge_excluded_modules") or [])
    max_rounds = int(state.get("knowledge_max_rounds") or DEFAULT_MAX_KNOWLEDGE_ROUNDS)

    empty_metrics = {
        "step_metrics": state.get("step_metrics", []),
        "node_metrics": state.get("node_metrics", {}),
        "agent_metrics": state.get("agent_metrics", {}),
        "total_tokens": state.get("total_tokens", 0),
        "total_duration_s": state.get("total_duration_s", 0.0),
    }

    if knowledge_loader is None:
        logger.info("Knowledge injection disabled, skipping loader node")
        return {"knowledge_loader_loaded": False, **empty_metrics}

    if not model_expert_output or not model_expert_output.knowledge_requests:
        logger.info("No knowledge requests from ModelExpert")
        return {"knowledge_loader_loaded": False, **empty_metrics}

    knowledge_round = state.get("knowledge_round", 0) + 1
    if knowledge_round > max_rounds:
        logger.warning(
            "Max knowledge rounds (%s) reached, stopping progressive refinement",
            max_rounds,
        )
        return _with_loader_metrics(
            state,
            start_time,
            {
                "knowledge_round": knowledge_round,
                "knowledge_loader_loaded": False,
            },
        )

    requested_modules = list(model_expert_output.knowledge_requests)
    blocked_modules = [name for name in requested_modules if name in excluded_modules]
    if blocked_modules:
        logger.info("Knowledge requests blocked by ablation profile: %s", blocked_modules)

    pending_modules = ProgressiveKnowledgeInjection.pending_requests(
        requested_modules, loaded_modules, excluded_modules
    )
    if not pending_modules:
        if blocked_modules:
            logger.info(
                "No loadable knowledge requests remain after ablation filtering: %s",
                requested_modules,
            )
        else:
            logger.info(
                "All requested knowledge modules are already loaded: %s",
                requested_modules,
            )
        return _with_loader_metrics(
            state,
            start_time,
            {"knowledge_loader_loaded": False},
        )

    logger.info(
        "Progressive knowledge load (round %s): %s",
        knowledge_round,
        pending_modules,
    )

    knowledge_content = knowledge_loader.get_knowledge_by_names(
        pending_modules,
        excluded_modules=excluded_modules,
    )

    if knowledge_content:
        logger.info("Successfully loaded %d knowledge module(s)", len(pending_modules))
    else:
        logger.warning("Failed to load requested modules: %s", pending_modules)

    existing_knowledge = state.get("loaded_knowledge")
    if existing_knowledge and knowledge_content:
        merged_knowledge = f"{existing_knowledge}\n\n---\n\n{knowledge_content}"
    else:
        merged_knowledge = knowledge_content or existing_knowledge

    loaded_module_names = (
        loaded_modules + pending_modules if knowledge_content else loaded_modules
    )

    return _with_loader_metrics(
        state,
        start_time,
        {
            "loaded_knowledge": merged_knowledge,
            "loaded_knowledge_modules": loaded_module_names,
            "knowledge_round": knowledge_round,
            "knowledge_loader_loaded": bool(knowledge_content),
        },
    )


def _with_loader_metrics(state: Dict, start_time: float, payload: Dict) -> Dict:
    duration = time.time() - start_time
    requested = []
    me_out = state.get("model_expert_output")
    if me_out is not None and getattr(me_out, "knowledge_requests", None):
        requested = list(me_out.knowledge_requests)
    loaded_after = list(
        payload.get("loaded_knowledge_modules", state.get("loaded_knowledge_modules") or [])
    )
    step_metrics = record_step_event(
        state,
        node="knowledge_loader",
        step_type="knowledge",
        duration_s=duration,
        total_tokens=0,
        extra={
            "requested_modules": requested,
            "loaded_modules": loaded_after,
            "catalog_only": not bool(payload.get("knowledge_loader_loaded")),
            "knowledge_loader_loaded": bool(payload.get("knowledge_loader_loaded")),
            "knowledge_round": payload.get("knowledge_round", state.get("knowledge_round")),
        },
    )
    node_metrics = state.get("node_metrics", {})
    if "knowledge_loader" not in node_metrics:
        node_metrics["knowledge_loader"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    node_metrics["knowledge_loader"]["total_duration_s"] += duration
    node_metrics["knowledge_loader"]["num_calls"] += 1
    return {
        **payload,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "agent_metrics": state.get("agent_metrics", {}),
        "total_tokens": state.get("total_tokens", 0),
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
    }
