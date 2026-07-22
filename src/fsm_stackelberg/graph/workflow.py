"""LangGraph workflow for MAKO multi-agent optimization system."""

import json
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .state import AgentState
from ..utils.llm_config import DEFAULT_PROVIDER, DEFAULT_MODEL
from ..agents.data_engineer import data_engineer_node, data_engineer_backward_step
from ..agents.knowledge_loader import knowledge_loader_node
from ..agents.model_expert import model_expert_node, model_expert_backward_step
from ..agents.python_developer import python_developer_node, python_developer_backward_step
from ..agents.solver_executor import solver_executor_node
from ..agents.diagnosis_agent import diagnosis_agent_node
from ..knowledge.knowledge_loader import KnowledgeLoader

logger = logging.getLogger(__name__)


def _excluded_modules_for_mode(knowledge_mode: str) -> list[str]:
    return []


def _has_pending_knowledge_requests(state: AgentState) -> bool:
    """Return True if ModelExpert requested modules not already loaded."""
    model_expert_output = state.get("model_expert_output")
    if not model_expert_output or not model_expert_output.knowledge_requests:
        return False

    loaded_modules = set(state.get("loaded_knowledge_modules", []))
    excluded_modules = set(state.get("knowledge_excluded_modules", []))
    return any(
        name not in loaded_modules and name not in excluded_modules
        for name in model_expert_output.knowledge_requests
    )


# =============================================================================
# Routing Functions
# =============================================================================

def should_diagnose(state: AgentState) -> Literal["success", "diagnose"]:
    """Determine if diagnosis is needed after SolverExecutor execution.

    Args:
        state: Current workflow state

    Returns:
        "success" if optimization succeeded, "diagnose" if diagnosis needed
    """
    execution_result = state.get("execution_result", {})
    diagnosis_required = execution_result.get("diagnosis_required", False)

    if not diagnosis_required:
        logger.info("Execution successful, ending workflow.")
        return "success"

    # Check retry limit
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count >= max_retries:
        logger.warning(f"Max retries ({max_retries}) exceeded, ending workflow.")
        return "success"  # End with failure

    logger.info(f"Diagnosis required, routing to DiagnosisAgent (retry {retry_count + 1}/{max_retries})")
    return "diagnose"


def route_after_diagnosis(state: AgentState) -> str:
    """Route after DiagnosisAgent based on diagnosis_mode.

    Stackelberg / adversarial: Route to the probed (accused) agent's backward_step
    Sequential: Start sequential backward_step chain from python_developer

    Args:
        state: Current workflow state with error_agent field

    Returns:
        Name of the next node
    """
    diagnosis_mode = state.get("diagnosis_mode", "stackelberg")
    error_agent = state.get("error_agent")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count > max_retries:
        logger.warning(f"Max retries exceeded, ending workflow")
        return "end"

    if error_agent is None:
        logger.warning("No error_agent from diagnosis, ending workflow")
        return "end"

    # Stackelberg inspection and adversarial baseline: probe/accuse one agent
    if diagnosis_mode in ("stackelberg", "adversarial"):
        if error_agent == "data_engineer":
            return "data_engineer_backward"
        elif error_agent == "model_expert":
            return "model_expert_backward"
        elif error_agent == "python_developer":
            return "python_developer_backward"

    # Sequential mode: start from python_developer (reverse order)
    return "python_developer_backward"


def route_after_backward(state: AgentState) -> str:
    """Route after backward_step execution.

    If error_resolved (inspectee complied), resume downstream — the subsequent
    solver run is the executed refutation of the repair.
    If not resolved (inspectee deflected):
      - stackelberg / adversarial: re-enter diagnosis (next causal probe / re-accuse)
      - sequential: try the next upstream agent in the reverse chain

    Args:
        state: Current workflow state with error_resolved field

    Returns:
        Name of the next node
    """
    error_resolved = state.get("error_resolved", False)
    error_agent = state.get("error_agent", "")
    diagnosis_mode = state.get("diagnosis_mode", "stackelberg")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if error_resolved:
        logger.info(f"Backward step resolved error from {error_agent}")

        # Determine which downstream agents need to re-run
        # After DataEngineer fix: re-run ModelExpert, guardrail, PythonDeveloper, SolverExecutor
        # After ModelExpert fix: re-run guardrail, PythonDeveloper, SolverExecutor
        # After PythonDeveloper fix: re-run SolverExecutor
        if error_agent == "data_engineer":
            return "model_expert"  # Re-run from ModelExpert
        elif error_agent == "model_expert":
            if _has_pending_knowledge_requests(state):
                return "knowledge_loader"
            return "python_developer"
        elif error_agent == "python_developer":
            return "solver_executor"  # Re-run SolverExecutor
        else:
            return "solver_executor"

    # Error not resolved (deflection / rebuttal)
    if diagnosis_mode in ("stackelberg", "adversarial"):
        # Re-enter diagnosis: Stackelberg picks NextCausalLayer; adversarial re-accuses
        if retry_count >= max_retries:
            logger.warning("Max retries exceeded after failed backward step")
            return "end"
        return "diagnosis_agent"

    # Sequential mode: try next agent in chain
    if error_agent == "python_developer":
        return "model_expert_backward"
    elif error_agent == "model_expert":
        return "data_engineer_backward"
    else:
        # All agents checked, no one admitted fault
        if retry_count >= max_retries:
            return "end"
        return "diagnosis_agent"


def route_after_model_expert(state: AgentState) -> Literal["knowledge_loader", "python_developer"]:
    """Route based on whether ModelExpert requested new knowledge modules."""
    if _has_pending_knowledge_requests(state):
        pending_modules = [
            name for name in state["model_expert_output"].knowledge_requests
            if name not in set(state.get("loaded_knowledge_modules", []))
        ]
        logger.info("ModelExpert requested new knowledge modules: %s", pending_modules)
        return "knowledge_loader"

    if state.get("model_expert_output") and state["model_expert_output"].knowledge_requests:
        logger.info("ModelExpert requested only already-loaded knowledge; proceeding to code generation.")
    return "python_developer"


def route_after_knowledge_loader(state: AgentState) -> Literal["model_expert", "python_developer"]:
    """Route after knowledge loading based on whether new knowledge was added."""
    if state.get("knowledge_loader_loaded", False):
        return "model_expert"

    logger.info("Knowledge loader added no new knowledge; proceeding to code generation.")
    return "python_developer"


# =============================================================================
# Graph Creation
# =============================================================================

def create_mako_graph() -> StateGraph:
    """Create the MAKO workflow graph.

    The workflow supports two diagnosis modes:
    - Adversarial: DiagnosisAgent accuses one error_agent, then backward_step verifies and corrects if accepted
    - Sequential: Each agent's backward_step is called in order until one admits fault

    Flow (Adversarial mode):
    ```
    DataEngineer → ModelExpert → {KnowledgeLoader} → PythonDeveloper → SolverExecutor
                                ↓                                              ↓
                      ModelExpert backward                              [should_diagnose]
                                                                                ↓
                                                                         DiagnosisAgent
                                                                                ↓
                                                                     [route_after_diagnosis]
                                                                                ↓
                                                                     {Agent}_backward → [route_after_backward]
                                                                                ↓
                                                                     Continue downstream or end
    ```

    Returns:
        Compiled LangGraph StateGraph
    """
    workflow = StateGraph(AgentState)

    # Add workflow nodes
    # Agent nodes
    workflow.add_node("data_engineer", data_engineer_node)
    workflow.add_node("model_expert", model_expert_node)
    workflow.add_node("knowledge_loader", knowledge_loader_node)
    workflow.add_node("python_developer", python_developer_node)
    workflow.add_node("solver_executor", solver_executor_node)
    workflow.add_node("diagnosis_agent", diagnosis_agent_node)

    # Add backward step nodes
    workflow.add_node("data_engineer_backward", data_engineer_backward_step)
    workflow.add_node("model_expert_backward", model_expert_backward_step)
    workflow.add_node("python_developer_backward", python_developer_backward_step)

    # Forward edges: linear flow
    workflow.add_edge("data_engineer", "model_expert")
    workflow.add_conditional_edges(
        "model_expert",
        route_after_model_expert,
        {
            "knowledge_loader": "knowledge_loader",
            "python_developer": "python_developer",
        }
    )
    workflow.add_conditional_edges(
        "knowledge_loader",
        route_after_knowledge_loader,
        {
            "model_expert": "model_expert",
            "python_developer": "python_developer",
        }
    )
    workflow.add_edge("python_developer", "solver_executor")

    # SolverExecutor → success or diagnose
    workflow.add_conditional_edges(
        "solver_executor",
        should_diagnose,
        {
            "success": END,
            "diagnose": "diagnosis_agent",
        }
    )

    # DiagnosisAgent → backward_step (adversarial) or sequential chain
    workflow.add_conditional_edges(
        "diagnosis_agent",
        route_after_diagnosis,
        {
            "data_engineer_backward": "data_engineer_backward",
            "model_expert_backward": "model_expert_backward",
            "python_developer_backward": "python_developer_backward",
            "end": END,
        }
    )

    # Backward step → continue downstream or re-diagnose
    workflow.add_conditional_edges(
        "data_engineer_backward",
        route_after_backward,
        {
            "model_expert": "model_expert",
            "diagnosis_agent": "diagnosis_agent",
            "end": END,
        }
    )

    workflow.add_conditional_edges(
        "model_expert_backward",
        route_after_backward,
        {
            "knowledge_loader": "knowledge_loader",
            "python_developer": "python_developer",
            "data_engineer_backward": "data_engineer_backward",
            "diagnosis_agent": "diagnosis_agent",
            "end": END,
        }
    )

    workflow.add_conditional_edges(
        "python_developer_backward",
        route_after_backward,
        {
            "solver_executor": "solver_executor",
            "model_expert_backward": "model_expert_backward",
            "diagnosis_agent": "diagnosis_agent",
            "end": END,
        }
    )

    # Set entry point
    workflow.set_entry_point("data_engineer")

    return workflow.compile()


def run_mako(
    problem_description: str,
    sample: dict,
    schema: dict = None,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    diagnosis_mode: str = "stackelberg",
    probe_order: str = "causal",
    expected_value: float = None,
    knowledge_mode: str = "enable",
    true_root_cause: str = None,
) -> dict:
    """Run the FSM-Stackelberg optimization workflow.

    Args:
        problem_description: Natural language problem description
        sample: Problem data (sample.json)
        schema: Optional schema (defaults to sample structure)
        provider: LLM provider (DeepSeek, OpenAI, etc.)
        model: LLM model to use
        max_retries: Maximum backtrack retries
        diagnosis_mode: "stackelberg" (inspection game), "adversarial"
            (single-judge baseline), or "sequential" (reverse-order baseline)
        probe_order: Commitment-order ablation for stackelberg mode:
            "causal" (default), "reverse", or "random"
        true_root_cause: Optional ground-truth failing layer for u_F
            (data_engineer | model_expert | python_developer)

    Returns:
        Final state with results (includes episode_payoff)
    """
    graph = create_mako_graph()

    # Deterministic data preprocessing — runs once, no LLM involved
    from ..data.auto_preprocessor import auto_preprocess
    preprocessed_data, data_access_guide = auto_preprocess(sample)
    logger.info(
        "Auto-preprocessed data: %d fields, guide length: %d chars",
        len(preprocessed_data),
        len(data_access_guide),
    )

    # Initialize knowledge loader and pre-load relevant knowledge
    excluded_modules = _excluded_modules_for_mode(knowledge_mode)
    if knowledge_mode == "disable":
        knowledge_loader = None
        recommended_modules = []
        knowledge_catalog = ""
        loaded_knowledge = None
        logger.info("Knowledge injection disabled")
    else:
        knowledge_loader = KnowledgeLoader()
        recommended_modules = knowledge_loader.recommend_modules(
            problem_description,
            excluded_modules=excluded_modules,
        )
        knowledge_catalog = knowledge_loader.get_knowledge_catalog_only(
            excluded_modules=excluded_modules,
        )

        # Pre-load recommended modules (Option A: simpler, more reliable)
        if recommended_modules:
            loaded_knowledge = knowledge_loader.get_knowledge_by_names(
                recommended_modules,
                excluded_modules=excluded_modules,
            )
            logger.info(f"Pre-loaded knowledge modules: {recommended_modules}")
        else:
            loaded_knowledge = None
        if excluded_modules:
            logger.info("Knowledge ablation excluded modules: %s", excluded_modules)

    initial_state: AgentState = {
        "problem_description": problem_description,
        "sample": sample,
        "schema": schema or sample,
        "provider": provider,
        "model": model,
        "preprocessed_data": preprocessed_data,
        "data_access_guide": data_access_guide,
        "retry_count": 0,
        "max_retries": max_retries,
        "diagnosis_mode": diagnosis_mode,
        "probe_order": probe_order,
        "inspection_policy": None,
        "probe_queue": [],
        "cleared_layers": [],
        "backtrack_history": [],
        # Metrics initialization
        "step_metrics": [],
        "node_metrics": {},
        "agent_metrics": {},
        "total_tokens": 0,
        "total_duration_s": 0.0,
        # Output history tracking
        "current_round": 1,
        "output_history": [],
        # Knowledge loading
        "knowledge_loader": knowledge_loader,
        "knowledge_catalog": knowledge_catalog,
        "knowledge_round": 0,
        "loaded_knowledge": loaded_knowledge,
        "loaded_knowledge_modules": recommended_modules,
        "knowledge_excluded_modules": excluded_modules,
        "knowledge_loader_loaded": False,
        "expected_value": expected_value,
        "true_root_cause": true_root_cause,
        "attributed_layer": None,
        "episode_payoff": None,
    }

    final_state = graph.invoke(initial_state)

    # Phase 2: record analysis-mode episode utilities on the finished trajectory.
    from ..game.payoff import finalize_episode_payoffs

    episode_payoff = finalize_episode_payoffs(final_state)
    final_state["attributed_layer"] = episode_payoff.get("attributed_layer")
    final_state["episode_payoff"] = episode_payoff
    logger.info(
        "Episode payoff: S=%s u_L=%s u_F=%s attributed=%s true_root=%s K=%s tokens=%s",
        episode_payoff.get("S"),
        episode_payoff.get("u_L"),
        episode_payoff.get("u_F"),
        episode_payoff.get("attributed_layer"),
        episode_payoff.get("true_root_cause"),
        episode_payoff.get("probe_rounds"),
        episode_payoff.get("tokens"),
    )

    return final_state
