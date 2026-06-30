"""CoT (Chain-of-Thought) Prompt Templates.

This module contains multi-stage prompt templates for the Chain-of-Thought method,
which breaks down problem-solving into sequential reasoning stages.

All baselines use the unified V2 data interface (auto_preprocessor).
"""

# Stage 1: Problem Analysis
COT_STAGE1_PROMPT = """You are an expert optimization analyst. Analyze the following problem:

**Problem Description:**
{problem_description}

**Problem Data:**
{data_description}

**Your task is to provide a structured analysis:**
1. Problem Type: What type of optimization problem is this?
2. Decision Variables: What decisions need to be made?
3. Parameters: What are the input parameters?
4. Objective: What is being optimized?
5. Constraints: What restrictions limit the decisions?

Provide a clear, structured analysis."""

# Stage 2: Model Formulation
COT_STAGE2_PROMPT = """You are an expert optimization modeler. Based on the following analysis, formulate a mathematical model:

**Problem Description:**
{problem_description}

**Analysis from Stage 1:**
{stage1_output}

**Formulate the mathematical model:**
1. Sets and Indices
2. Parameters
3. Decision Variables (with types)
4. Objective Function
5. Constraints

Use standard mathematical notation."""

# Stage 3: Code Implementation
COT_STAGE3_PROMPT = """You are an expert optimization programmer. Implement the following mathematical model in Python using Gurobi:

**Mathematical Formulation:**
{stage2_output}

**Problem Description:**
{problem_description}

**Problem Data:**
{data_description}

**CRITICAL REQUIREMENTS:**
1. Function MUST be named `optimize` with signature `def optimize(data: dict) -> dict:`
2. Return dict MUST have: "status", "objective_value", "variables"
3. DO NOT wrap code in markdown fences
4. Provide ONLY Python code

Write the complete implementation:"""


def get_cot_stage1_prompt(problem_description: str, data_description: str = "") -> str:
    """Get prompt for Stage 1: Problem Analysis."""
    return COT_STAGE1_PROMPT.format(
        problem_description=problem_description,
        data_description=data_description or "No data provided"
    )


def get_cot_stage2_prompt(problem_description: str, stage1_output: str) -> str:
    """Get prompt for Stage 2: Model Formulation."""
    return COT_STAGE2_PROMPT.format(
        problem_description=problem_description,
        stage1_output=stage1_output
    )


def get_cot_stage3_prompt(problem_description: str, data_description: str, stage2_output: str) -> str:
    """Get prompt for Stage 3: Code Implementation."""
    return COT_STAGE3_PROMPT.format(
        problem_description=problem_description,
        data_description=data_description or "No data provided",
        stage2_output=stage2_output
    )
