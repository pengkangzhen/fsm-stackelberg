"""SPM (Standard Prompting Method) Prompt Templates.

This module contains prompt templates for the Standard Prompting Method,
which uses a single comprehensive prompt to generate optimization solutions.
"""

# SPM System Prompt - Sets up the AI's role and capabilities
SPM_SYSTEM_PROMPT = """You are an expert optimization specialist with deep knowledge in:
- Mathematical modeling and optimization
- Operations research techniques
- Linear programming, integer programming, and mixed-integer programming
- Python programming for optimization
- Using solvers like Gurobi, CPLEX, and OR-Tools

Your task is to solve optimization problems by:
1. Understanding the problem description
2. Formulating a mathematical model
3. Implementing the solution in Python using appropriate solvers
4. Providing clear, executable code

**IMPORTANT CONSTRAINTS:**
- Provide ONLY executable Python code - no explanations outside the code
- The entire solution MUST be encapsulated within a single function named `optimize`
- The function signature MUST be `def optimize(data: dict) -> dict:`
- Include necessary imports (gurobipy, numpy, pandas, etc.)
- Use proper variable naming that reflects the problem context
- Add comments in the code to explain the model formulation
- Ensure the code runs without errors
"""

SPM_HUMAN_PROMPT = """**Problem Description:**
{problem_description}

**Problem Data:**
{data_description}

**Instructions:**
1. Analyze this optimization problem carefully
2. Formulate a mathematical model (sets, parameters, variables, objective, constraints)
3. Write a complete `optimize(data: dict) -> dict:` function
4. Unpack data from the input dictionary using the keys shown above
5. Implement the optimization model
6. Solve and return results in the specified format

**CRITICAL REQUIREMENTS:**
1. Function MUST be named `optimize` with signature `def optimize(data: dict) -> dict:`
2. Always handle non-optimal statuses (INFEASIBLE, UNBOUNDED, etc.)
3. Return dict MUST have exactly three keys: "status", "objective_value", "variables"
4. For non-optimal solutions, set objective_value and variables to None
5. Your response must start with `import gurobipy as gp` or `import` statements
6. DO NOT wrap code in markdown fences (```python or ```)
7. DO NOT include any explanatory text outside the code
8. The imports MUST be at the top, before the `optimize` function

Provide ONLY the Python code starting with imports. No markdown, no explanations."""


def get_spm_prompt(problem_description: str, data_description: str = "") -> str:
    """Get formatted SPM prompt for a problem.

    Args:
        problem_description: Natural language problem description
        data_description: Data access guide (raw JSON or Schema+Compact from auto_preprocessor)

    Returns:
        Formatted prompt string
    """
    return SPM_SYSTEM_PROMPT + "\n\n" + SPM_HUMAN_PROMPT.format(
        problem_description=problem_description,
        data_description=data_description or "No data provided"
    )
