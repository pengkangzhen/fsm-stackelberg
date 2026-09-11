"""Prompt templates for MAKO LangChain agents.

Prompts are loaded from external Markdown files in the templates/ directory.
Each agent has its own subdirectory with role.md and forward.md files.
"""

from pathlib import Path
from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate

# 模板目录路径
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=None)
def _load_prompt(agent: str, task: str) -> str:
    """Load a prompt template from templates/<agent>/<task>.md.

    Args:
        agent: Agent name (e.g., "data_engineer", "model_expert")
        task: Task type (e.g., "role", "forward", "backward_step")

    Returns:
        The prompt template string with {placeholder} format
    """
    path = _TEMPLATES_DIR / agent / f"{task}.md"
    return path.read_text(encoding="utf-8")


# ============ DataEngineer ============

DATA_ENGINEER_ROLE = _load_prompt("data_engineer", "role")
DATA_ENGINEER_FORWARD = _load_prompt("data_engineer", "forward")
DATA_ENGINEER_BACKWARD_STEP = _load_prompt("data_engineer", "backward_step")


def get_data_engineer_prompt() -> ChatPromptTemplate:
    """Get the DataEngineer prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", "{role}"),
        ("human", "{task}"),
    ])


# ============ ModelExpert ============

MODEL_EXPERT_ROLE = _load_prompt("model_expert", "role")
MODEL_EXPERT_FORWARD = _load_prompt("model_expert", "forward")
MODEL_EXPERT_BACKWARD_STEP = _load_prompt("model_expert", "backward_step")


def get_model_expert_prompt() -> ChatPromptTemplate:
    """Get the ModelExpert prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", "{role}"),
        ("human", "{task}"),
    ])


# ============ PythonDeveloper ============

PYTHON_DEVELOPER_ROLE = _load_prompt("python_developer", "role")
PYTHON_DEVELOPER_FORWARD = _load_prompt("python_developer", "forward")
PYTHON_DEVELOPER_BACKWARD_STEP = _load_prompt("python_developer", "backward_step")


def get_python_developer_prompt() -> ChatPromptTemplate:
    """Get the PythonDeveloper prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", "{role}"),
        ("human", "{task}"),
    ])


# ============ DiagnosisAgent ============

DIAGNOSIS_ROLE = _load_prompt("diagnosis_agent", "role")
DIAGNOSIS_PROMPT = _load_prompt("diagnosis_agent", "forward")
LAYER_RANK_PROMPT = _load_prompt("diagnosis_agent", "rank")
BACKWARD_STEP_PROMPT = _load_prompt("diagnosis_agent", "backward_step")
DEBATE_ROLE = _load_prompt("diagnosis_agent", "debate_role")
DEBATE_PROMPT = _load_prompt("diagnosis_agent", "debate")
REFLEXION_ROLE = _load_prompt("diagnosis_agent", "reflexion_role")
REFLEXION_PROMPT = _load_prompt("diagnosis_agent", "reflexion")


def get_diagnosis_prompt() -> ChatPromptTemplate:
    """Get the diagnosis prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", "{role}"),
        ("human", "{task}"),
    ])


__all__ = [
    "_load_prompt",
    # DataEngineer
    "DATA_ENGINEER_ROLE",
    "DATA_ENGINEER_FORWARD",
    "DATA_ENGINEER_BACKWARD_STEP",
    "get_data_engineer_prompt",
    # ModelExpert
    "MODEL_EXPERT_ROLE",
    "MODEL_EXPERT_FORWARD",
    "MODEL_EXPERT_BACKWARD_STEP",
    "get_model_expert_prompt",
    # PythonDeveloper
    "PYTHON_DEVELOPER_ROLE",
    "PYTHON_DEVELOPER_FORWARD",
    "PYTHON_DEVELOPER_BACKWARD_STEP",
    "get_python_developer_prompt",
    # DiagnosisAgent
    "DIAGNOSIS_ROLE",
    "DIAGNOSIS_PROMPT",
    "DEBATE_ROLE",
    "DEBATE_PROMPT",
    "REFLEXION_ROLE",
    "REFLEXION_PROMPT",
    "LAYER_RANK_PROMPT",
    "BACKWARD_STEP_PROMPT",  # Legacy, kept for backward compatibility
    "get_diagnosis_prompt",
]
