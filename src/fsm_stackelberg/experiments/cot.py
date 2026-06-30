"""CoT (Chain-of-Thought) Implementation.

Multi-stage reasoning for optimization problem solving.
"""

import logging
import re
import time
from typing import Dict, Any, Optional

from langchain_community.callbacks import get_openai_callback

from fsm_stackelberg.data.auto_preprocessor import auto_preprocess
from fsm_stackelberg.utils.llm_config import get_llm, DEFAULT_PROVIDER, DEFAULT_MODEL
from fsm_stackelberg.agents.solver_executor import sandbox_exec_code
from fsm_stackelberg.prompts.cot_prompts import (
    get_cot_stage1_prompt,
    get_cot_stage2_prompt,
    get_cot_stage3_prompt,
)

logger = logging.getLogger(__name__)


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    # Remove markdown code blocks if present
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    # Find the optimize function
    if 'def optimize(' in response:
        lines = response.strip().split('\n')
        code_lines = []
        in_function = False

        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                code_lines.append(line)
            elif 'def optimize(' in line:
                in_function = True
                code_lines.append(line)
            elif in_function:
                code_lines.append(line)

        return '\n'.join(code_lines)

    return response.strip()


def run_cot(
    problem: Dict[str, Any],
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Run CoT algorithm on a problem.

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

    logger.info(f"Starting CoT for {dataset}/{prob_name}")

    # Initialize result
    result = {
        "algorithm": "cot",
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
        "stage_outputs": {},
    }

    try:
        # Get LLM
        llm = get_llm(provider=provider, model=model, temperature=0)

        # Stage 1: Problem Analysis
        logger.info("CoT: Stage 1 - Problem Analysis")
        stage1_start = time.time()
        stage1_prompt = get_cot_stage1_prompt(description, data_access_guide)
        with get_openai_callback() as cb:
            stage1_response = llm.invoke(stage1_prompt)
        stage1_duration = time.time() - stage1_start

        stage1_output = stage1_response.content
        result["total_tokens"] += cb.total_tokens
        result["steps"].append({
            "step": "stage1_analysis",
            "duration_s": round(stage1_duration, 3),
            "tokens": cb.total_tokens,
        })
        result["stage_outputs"]["stage1"] = stage1_output[:500]

        logger.info(f"CoT: Stage 1 complete ({cb.total_tokens} tokens)")

        # Stage 2: Model Formulation
        logger.info("CoT: Stage 2 - Model Formulation")
        stage2_start = time.time()
        stage2_prompt = get_cot_stage2_prompt(description, stage1_output)
        with get_openai_callback() as cb:
            stage2_response = llm.invoke(stage2_prompt)
        stage2_duration = time.time() - stage2_start

        stage2_output = stage2_response.content
        result["total_tokens"] += cb.total_tokens
        result["steps"].append({
            "step": "stage2_formulation",
            "duration_s": round(stage2_duration, 3),
            "tokens": cb.total_tokens,
        })
        result["stage_outputs"]["stage2"] = stage2_output[:500]

        logger.info(f"CoT: Stage 2 complete ({cb.total_tokens} tokens)")

        # Stage 3: Code Implementation
        logger.info("CoT: Stage 3 - Code Implementation")
        stage3_start = time.time()
        stage3_prompt = get_cot_stage3_prompt(description, data_access_guide, stage2_output)
        with get_openai_callback() as cb:
            stage3_response = llm.invoke(stage3_prompt)
        stage3_duration = time.time() - stage3_start

        code = extract_code(stage3_response.content)
        result["total_tokens"] += cb.total_tokens
        result["steps"].append({
            "step": "stage3_code",
            "duration_s": round(stage3_duration, 3),
            "tokens": cb.total_tokens,
        })

        logger.info(f"CoT: Stage 3 complete ({cb.total_tokens} tokens)")

        if not code or 'def optimize' not in code:
            result["error_msg"] = "Could not extract valid optimize() function"
            logger.error(result["error_msg"])
            return result

        # Execute code using processed data
        logger.info("CoT: Executing code")
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
            logger.info(f"CoT: Success! Objective = {result['obj_value']}")
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
            logger.error(f"CoT: Failed - {result['error_msg']}")

    except Exception as e:
        result["error_msg"] = str(e)
        logger.error(f"CoT: Exception - {e}")

    result["total_duration_s"] = round(time.time() - start_time, 3)

    return result
