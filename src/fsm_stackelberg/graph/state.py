"""AgentState definition for MAKO LangGraph workflow."""

from typing import TypedDict, Optional, Dict, Any, Literal, List
from langgraph.graph import add_messages

from ..schemas import DataEngineerOutput, ModelExpertOutput


class AgentState(TypedDict, total=False):
    """State schema for MAKO multi-agent workflow.

    This state is passed between agents in the LangGraph workflow.
    """

    # Input data
    problem_description: str
    schema: Dict[str, Any]
    sample: Dict[str, Any]
    provider: str  # LLM provider (DeepSeek, OpenAI, etc.)
    model: str  # LLM model to use
    temperature: float  # LLM temperature (default 0; logged for reproducibility)

    # Run logging (result_dir drives events.jsonl / artifacts)
    result_dir: Optional[str]
    run_id: Optional[str]
    log_prompts: bool  # when True, dump full prompts under prompts/

    # Deterministic preprocessing (set before any LLM agent runs)
    preprocessed_data: Optional[Dict[str, Any]]  # Auto-preprocessed data dict for sandbox
    data_access_guide: Optional[str]  # Compact text guide for LLM agents
    data_catalog: Optional[Dict[str, Dict[str, Any]]]  # Allowed DE source IDs + index domains

    # Agent outputs
    data_engineer_output: Optional[DataEngineerOutput]
    model_expert_output: Optional[ModelExpertOutput]
    python_code: Optional[str]
    execution_result: Optional[Dict[str, Any]]

    # Error handling - 完整错误上下文
    error_info: Optional[str]  # 错误详情 JSON 字符串
    error_category: Optional[str]  # "syntax" | "execution" | "optimization"
    gurobi_status: Optional[str]  # Gurobi 求解状态
    stack_trace: Optional[str]  # 完整 Python traceback
    error_agent: Optional[str]  # DiagnosisAgent 指控的责任 Agent，待 backward_step 验证
    retry_count: int
    max_retries: int
    diagnosis_mode: str  # "stackelberg" | "adversarial" | "sequential"
    probe_order: str  # "causal" | "reverse" | "random" (Stackelberg commitment-order ablation)

    # Stackelberg inspection-game state (DiagnosisAgent = inspector)
    inspection_policy: Optional[Dict[str, Any]]  # committed σ = (ω, ν)
    probe_queue: List[str]  # committed probing order for this failure episode
    cleared_layers: List[str]  # layers cleared by executed refutation / upheld deflection
    refutation_log: List[Dict[str, Any]]  # per-probe executed-refutation records
    omega_source: str  # "evidence_rank" (default) | "status_prior"
    rank_method: str  # "llm_rank" | "heuristic" | "hybrid"

    # Backward step results
    error_resolved: Optional[bool]  # Whether backward_step resolved the error
    backward_reason: Optional[str]  # Agent's explanation from backward_step

    # Knowledge loading (progressive injection; see knowledge/progressive.py)
    knowledge_loader: Optional[Any]  # KnowledgeLoader instance
    knowledge_catalog: Optional[str]  # name + description table only
    knowledge_round: int             # Counter for successful load rounds
    knowledge_max_rounds: int        # Cap on progressive refinements
    knowledge_injection_mode: str    # "progressive" | "disable"
    loaded_knowledge: Optional[str]  # Full text of requested modules only
    loaded_knowledge_modules: List[str]  # Names already injected
    knowledge_excluded_modules: List[str]  # Ablation exclusions
    knowledge_loader_loaded: Optional[bool]  # Whether last loader step added modules

    # Ground truth
    expected_value: Optional[float]  # Ground truth objective for gap validation
    true_root_cause: Optional[str]  # Injected layer label for attribution payoff (Phase 4)

    # Fault injection / probe reproducibility (optional; filled when CLI provides them)
    inject_id: Optional[str]
    fault_injected: bool
    fault_plant_id: Optional[str]
    probe_seed: Optional[int]

    # Failure-snapshot freeze/resume (see graph/snapshot.py)
    snapshot_dir: Optional[str]      # set to freeze the failure blackboard before diagnosis
    snapshot_written: Optional[bool]  # set by the snapshot gate once frozen

    # Episode payoffs (analysis-mode Stackelberg utilities; set at run end)
    attributed_layer: Optional[str]  # Mechanism's root-cause claim â
    episode_payoff: Optional[Dict[str, Any]]  # {u_L, u_F, S, costs, ...}

    # Backtrack history
    backtrack_history: list

    # Metrics tracking
    step_metrics: List[Dict]       # Per-node step metrics (tokens, duration)
    node_metrics: Dict[str, Dict]  # Aggregated metrics per workflow node
    agent_metrics: Dict[str, Dict] # Aggregated metrics for true agents only
    total_tokens: int
    total_duration_s: float

    # Output history tracking
    current_round: int             # 当前轮次 (默认 1)
    output_history: List[Dict]     # 历史输出记录, 每条记录包含 round, agent, output
