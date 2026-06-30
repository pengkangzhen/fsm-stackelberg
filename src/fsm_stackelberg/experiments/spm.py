"""SPM (Standard Prompting Method) Implementation.

Single-shot prompting for optimization problem solving.
"""

import logging
import re
import time
from typing import Dict, Any, Optional

from langchain_community.callbacks import get_openai_callback

from fsm_stackelberg.data.auto_preprocessor import auto_preprocess
from fsm_stackelberg.utils.llm_config import get_llm, DEFAULT_PROVIDER, DEFAULT_MODEL
from fsm_stackelberg.agents.solver_executor import sandbox_exec_code
from fsm_stackelberg.prompts.spm_prompts import get_spm_prompt

logger = logging.getLogger(__name__)


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    # Remove markdown code blocks if present
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    # Find the optimize function
    if 'def optimize(' in response:
        # Find imports and function
        lines = response.strip().split('\n')
        code_lines = []
        in_function = False

        for line in lines:
            # Include import statements
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                code_lines.append(line)
            # Start of optimize function
            elif 'def optimize(' in line:
                in_function = True
                code_lines.append(line)
            # Inside function
            elif in_function:
                code_lines.append(line)

        return '\n'.join(code_lines)

    return response.strip()


def run_spm(
    problem: Dict[str, Any],
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Run SPM algorithm on a problem.

    Args:
        problem: Problem data containing:
            - description: str - Natural language problem description
            - sample: dict - Problem data
            - dataset: str - Dataset name
            - prob_name: str - Problem name
        provider: LLM provider name
        model: Model name

    Returns:
        Dict with execution results
    """
    start_time = time.time()

    description = problem.get("description", "")
    dataset = problem.get("dataset", "")
    prob_name = problem.get("prob_name", "")

    # Unified V2 data interface: auto_preprocess handles compression
    sample = problem.get("sample", {})
    processed_data, data_access_guide = auto_preprocess(sample)

    logger.info(f"Starting SPM for {dataset}/{prob_name}")

    # Initialize result
    result = {
        "algorithm": "spm",
        "dataset": dataset,
        "prob_name": prob_name,
        "provider": provider,
        "model": model,
        "status": False,
        "obj_value": None,
        "is_model_valid": None,
        "gurobi_status": "",
        "total_tokens": 0,
        "total_duration_s": 0.0,
        "error_msg": None,
        "steps": [],
    }

    try:
        # Get LLM
        llm = get_llm(provider=provider, model=model, temperature=0)

        # Generate prompt — use V2 data access guide
        prompt = get_spm_prompt(description, data_access_guide)

        # Get LLM response
        step_start = time.time()
        with get_openai_callback() as cb:
            response = llm.invoke(prompt)
        step_duration = time.time() - step_start
        step_tokens = cb.total_tokens

        result["total_tokens"] += step_tokens
        result["steps"].append({
            "step": "code_generation",
            "duration_s": round(step_duration, 3),
            "tokens": step_tokens,
        })

        logger.info(f"SPM: Generated response ({step_tokens} tokens, {step_duration:.2f}s)")

        # Extract code
        code = extract_code(response.content)

        if not code or 'def optimize' not in code:
            result["error_msg"] = "Could not extract valid optimize() function"
            logger.error(result["error_msg"])
            return result

        # Execute code using processed data
        exec_start = time.time()
        exec_report = sandbox_exec_code(processed_data, code)
        exec_duration = time.time() - exec_start

        result["steps"].append({
            "step": "code_execution",
            "duration_s": round(exec_duration, 3),
            "tokens": 0,
        })

        # Process execution result
        if exec_report and exec_report.get("diagnosis_required") is False:
            result["status"] = True
            exec_result = exec_report.get("result") or {}
            result["obj_value"] = exec_result.get("objective_value")
            result["gurobi_status"] = exec_report.get("gurobi_status", "")
            logger.info(f"SPM: Success! Objective = {result['obj_value']}")
        else:
            if exec_report:
                result["gurobi_status"] = exec_report.get("gurobi_status", "")
                error_details = exec_report.get("error_details") or {}
                gurobi_status = exec_report.get("gurobi_status", "unknown")
                if isinstance(error_details, dict):
                    result["error_msg"] = error_details.get("error_message") or f"Gurobi status: {gurobi_status}"
                else:
                    result["error_msg"] = str(error_details) if error_details else f"Gurobi status: {gurobi_status}"
            else:
                result["error_msg"] = "Execution returned no report"
            logger.error(f"SPM: Failed - {result['error_msg']}")

    except Exception as e:
        result["error_msg"] = str(e)
        logger.error(f"SPM: Exception - {e}")

    result["total_duration_s"] = round(time.time() - start_time, 3)

    return result
