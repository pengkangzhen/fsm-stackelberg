"""SolverExecutor - LangChain implementation.

Executes generated Gurobi code in a sandbox environment and validates results.
"""

import json
import traceback
import logging
import importlib
import math
import numpy as np
import os
import sys
import itertools
import functools
import collections
import copy
import random
import re
import datetime
import time
import typing
from typing import Dict

import ast

import gurobipy

from ..utils.utils import record_agent_output
from ..utils.run_log import record_step_event
from ..constants import GAP_THRESHOLD

logger = logging.getLogger(__name__)

# Error type classification
SYNTAX_ERRORS = {"SyntaxError", "IndentationError", "TabError"}

# Sandbox whitelist: allowed safe modules
SAFE_MODULES = {
    "math", "numpy", "np", "json", "os", "sys",
    "itertools", "functools", "collections", "copy",
    "random", "re", "datetime", "time", "typing",
    "gurobipy", "gp", "GRB",
}


def safe_sandbox_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Safe sandbox import function.

    Only allows importing modules from the whitelist.
    """
    if name not in SAFE_MODULES:
        raise ImportError(f"Sandbox blocked import of module: {name}")

    try:
        return importlib.__import__(name, globals, locals, fromlist, level)
    except ImportError as e:
        logger.warning(f"Sandbox import failed: {name}, error: {e}")
        raise


def validate_code(code: str) -> tuple[bool, str]:
    """Static validation of generated Python code before execution.

    Returns (ok, error_message).  ok=True means the code passed all checks.
    """
    # 1. Compile check (catches SyntaxError)
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"

    # 2. AST checks
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"AST parse failed: {e}"

    FORBIDDEN_EXCEPT = False  # toggled per-policy; for now just warn

    for node in ast.walk(tree):
        # No try/except with bare Exception in the optimize function
        if isinstance(node, ast.ExceptHandler):
            if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                if FORBIDDEN_EXCEPT:
                    return False, "bare 'except Exception' not allowed — let the sandbox handle errors"
                else:
                    logger.warning("Code contains 'except Exception' — real errors may be swallowed")

        # Check import aliases: only 'gp' is pre-injected
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gurobipy" and alias.asname and alias.asname != "gp":
                    return False, (
                        f"Use 'import gurobipy as gp' (not '... as {alias.asname}'). "
                        "The sandbox only pre-injects 'gp' into the global namespace."
                    )

    return True, ""


def _gurobi_fingerprint(model) -> str:
    """Deterministic structural hash of a Gurobi model via its LP serialization.

    Gurobi normalizes variable/constraint ordering in LP output, so the hash is
    stable across runs for the same structure (§5.5 formulation-stability analysis).
    """
    import hashlib
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".lp")
    os.close(fd)
    try:
        model.write(path)                 # Gurobi infers LP format from .lp extension
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _gurobi_structure(model) -> Dict:
    """Introspect a solved Gurobi model: (R, C, NZ) + structural fingerprint.

    Returns {} if model is None or introspection fails. Resolves the §5.5
    "28/30 No solver log" gap by recording the structure deterministically.
    """
    if model is None:
        return {}
    try:
        model.update()
        return {
            "rows": model.NumConstrs,
            "cols": model.NumVars,
            "nonzeros": model.NumNZs,
            "fingerprint": _gurobi_fingerprint(model),
        }
    except Exception as e:
        logger.warning("Gurobi structure introspection failed: %s", e)
        return {}


def sandbox_exec_code(data: Dict, code: str) -> Dict:
    """Execute Gurobi code in a sandbox environment.

    Args:
        data: Preprocessed data dictionary
        code: Python code string containing optimize() function

    Returns:
        Dict: Execution report with status, result, and error details
    """
    execution_report = {
        "execution_successful": False,
        "diagnosis_required": True,
        "result": None,
        "error_details": None,
        "gurobi_log": "Not available",
        "gurobi_status": "",
        "error_category": "none",
    }

    # --- Static code validation ---
    ok, validation_msg = validate_code(code)
    if not ok:
        execution_report["error_details"] = {
            "error_type": "CodeValidationError",
            "error_message": validation_msg,
            "stack_trace": None,
        }
        execution_report["error_category"] = "syntax"
        logger.warning("Code validation failed: %s", validation_msg)
        return execution_report

    try:
        # Prepare sandbox globals
        globals_dict = {
            "gp": gurobipy,
            "GRB": gurobipy.GRB,
            "__import__": safe_sandbox_import,
            "math": math,
            "np": np,
            "numpy": np,
            "json": json,
            "os": os,
            "sys": sys,
            "itertools": itertools,
            "functools": functools,
            "collections": collections,
            "copy": copy,
            "random": random,
            "re": re,
            "datetime": datetime,
            "time": time,
            "typing": typing,
        }

        local_vars = {}

        # ── Gurobi model capture hook (§5.5 formulation fingerprint) ──
        # Contract-based generation keeps the model inside optimize()'s local
        # scope, unreachable after return. Hook Model.optimize to capture it.
        _captured_model = {"structure": {}}
        _orig_model_optimize = gurobipy.Model.optimize

        def _capture_model_optimize(self, *a, **kw):
            r = _orig_model_optimize(self, *a, **kw)
            # Introspect immediately while the model is guaranteed alive: generated
            # code may dispose/free the model (or its Env) after optimize() returns,
            # which would raise "Model has already been freed" on deferred access.
            _captured_model["structure"] = _gurobi_structure(self)
            return r

        gurobipy.Model.optimize = _capture_model_optimize
        try:
            exec(code, globals_dict, local_vars)

            # Add local functions to globals for optimize() to access
            for key, value in local_vars.items():
                if callable(value):
                    globals_dict[key] = value

            # Validate optimize function exists
            if "optimize" not in local_vars:
                execution_report["error_details"] = {
                    "error_type": "ContractViolationError",
                    "error_message": "'optimize' function not found in generated code.",
                    "stack_trace": None,
                }
                return execution_report

            # Execute optimize function
            result: Dict = local_vars["optimize"](data)
        finally:
            gurobipy.Model.optimize = _orig_model_optimize

        # Validate result structure
        if not isinstance(result, dict):
            execution_report["error_details"] = {
                "error_type": "ContractViolationError",
                "error_message": f"optimize() returned {type(result)}, expected dict.",
                "stack_trace": None,
            }
            return execution_report

        required_keys = {"status", "objective_value", "variables"}
        if not required_keys.issubset(result.keys()):
            missing_keys = required_keys - result.keys()
            execution_report["error_details"] = {
                "error_type": "ContractViolationError",
                "error_message": f"Missing required keys: {missing_keys}.",
                "stack_trace": None,
            }
            return execution_report

        # Analyze outcome
        gurobi_status = result.get("status")
        execution_report["execution_successful"] = True
        execution_report["result"] = result

        # Accept both string ("OPTIMAL") and int (GRB.OPTIMAL = 2) status codes
        _GRB_STATUS_NAMES = {1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED"}
        if isinstance(gurobi_status, int):
            from gurobipy import GRB
            is_optimal = gurobi_status == GRB.OPTIMAL
            gurobi_status_str = _GRB_STATUS_NAMES.get(gurobi_status, str(gurobi_status))
        else:
            is_optimal = str(gurobi_status).upper() == "OPTIMAL"
            gurobi_status_str = str(gurobi_status).upper()

        if is_optimal:
            execution_report["diagnosis_required"] = False
            execution_report["gurobi_status"] = "OPTIMAL"
            execution_report["error_category"] = "none"
            execution_report["gurobi_structure"] = _captured_model.get("structure", {})
            logger.info("Execution successful and model optimal.")
        else:
            execution_report["diagnosis_required"] = True
            execution_report["gurobi_status"] = gurobi_status_str
            execution_report["error_category"] = "optimization"
            logger.warning(f"Model status is '{gurobi_status_str}'. Diagnosis required.")

        return execution_report

    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        stack_trace = traceback.format_exc()

        error_category = "syntax" if error_type in SYNTAX_ERRORS else "execution"

        execution_report["execution_successful"] = False
        execution_report["diagnosis_required"] = True
        execution_report["gurobi_status"] = ""
        execution_report["error_category"] = error_category
        execution_report["error_details"] = {
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
        }
        logger.error(f"Execution failed with {error_type}: {error_message}\n{stack_trace}")
        return execution_report


def solver_executor_node(state: Dict) -> Dict:
    """LangGraph node function for SolverExecutor.

    Args:
        state: The current graph state containing python_code and data_engineer_output

    Returns:
        Updated state with execution_result and error context
    """
    logger.info("SolverExecutor: Starting execution...")
    start_time = time.time()

    python_code = state.get("python_code")
    data_engineer_output = state.get("data_engineer_output")
    sample = state.get("sample")

    if not python_code:
        duration = time.time() - start_time
        step_metrics = record_step_event(
            state,
            node="solver_executor",
            step_type="solver",
            duration_s=duration,
            ok=False,
            error_type="MissingCode",
            error_message="No Python code provided",
            extra={"gurobi_status": "", "diagnosis_required": True},
        )
        return {
            "execution_result": {
                "execution_successful": False,
                "diagnosis_required": True,
                "error_category": "execution",
                "error_details": {"error_message": "No Python code provided"},
            },
            "error_info": "No Python code provided",
            "error_category": "execution",
            "gurobi_status": "",
            "data_engineer_output": data_engineer_output,
            "model_expert_output": state.get("model_expert_output"),
            "python_code": python_code,
            "retry_count": state.get("retry_count", 0),
            "step_metrics": step_metrics,
            "total_duration_s": state.get("total_duration_s", 0.0) + duration,
        }

    # Use preprocessed data from workflow state (produced by auto_preprocessor in run_mako)
    data = state.get("preprocessed_data")
    if data is None:
        duration = time.time() - start_time
        step_metrics = record_step_event(
            state,
            node="solver_executor",
            step_type="solver",
            duration_s=duration,
            ok=False,
            error_type="MissingData",
            error_message="preprocessed_data missing from state",
            extra={"gurobi_status": "", "diagnosis_required": True},
        )
        return {
            "execution_result": {
                "execution_successful": False,
                "diagnosis_required": True,
                "error_category": "execution",
                "error_details": {"error_message": "preprocessed_data missing from state"},
            },
            "error_info": "preprocessed_data missing from state",
            "error_category": "execution",
            "gurobi_status": "",
            "data_engineer_output": data_engineer_output,
            "model_expert_output": state.get("model_expert_output"),
            "python_code": python_code,
            "retry_count": state.get("retry_count", 0),
            "step_metrics": step_metrics,
            "total_duration_s": state.get("total_duration_s", 0.0) + duration,
        }

    # Execute code
    exec_report = sandbox_exec_code(data, python_code)

    # --- NEW: Ground truth gap validation ---
    # gap is in percent; GAP_THRESHOLD is a fraction (single source of truth).
    if exec_report.get("execution_successful") and exec_report.get("gurobi_status") == "OPTIMAL":
        expected_value = state.get("expected_value")
        obj_value = exec_report.get("result", {}).get("objective_value")
        if expected_value is not None and obj_value is not None:
            gap = abs(obj_value - expected_value) / abs(expected_value) * 100
            if gap > GAP_THRESHOLD * 100:
                logger.warning(
                    f"Solution OPTIMAL but gap={gap:.2f}% > threshold {GAP_THRESHOLD * 100}%. "
                    f"Triggering diagnosis ( obj={obj_value})"
                )
                exec_report["diagnosis_required"] = True
                exec_report["error_category"] = "optimization"
                exec_report["error_details"] = {
                    "error_type": "SuboptimalSolution",
                    "error_message": f"OPTIMAL but gap={gap:.2f}% (obj={obj_value}, expected={expected_value})",
                }

    # Build error info for backtracking
    error_info = None
    if exec_report.get("diagnosis_required"):
        error_details = exec_report.get("error_details")
        if error_details:
            # Execution/syntax error: pass full traceback + message
            error_info_parts = {
                "error_type": error_details.get("error_type"),
                "error_message": error_details.get("error_message"),
                "stack_trace": error_details.get("stack_trace"),
            }
        else:
            # Optimization error (INFEASIBLE, UNBOUNDED, etc.): build rich context
            error_info_parts = {
                "error_type": "OptimizationError",
                "error_message": f"Gurobi returned status '{exec_report.get('gurobi_status')}'",
                "result_summary": exec_report.get("result"),
            }
        error_info = json.dumps(error_info_parts, ensure_ascii=False, default=str)

    logger.info(f"SolverExecutor: Execution {'successful' if exec_report['execution_successful'] else 'failed'}")
    logger.info(f"SolverExecutor: error_category={exec_report.get('error_category')}, gurobi_status={exec_report.get('gurobi_status')}")

    duration = time.time() - start_time
    result_summary = exec_report.get("result") or {}
    obj = result_summary.get("objective_value") if isinstance(result_summary, dict) else None
    structure = exec_report.get("gurobi_structure") or {}
    err_details = exec_report.get("error_details") or {}
    step_metrics = record_step_event(
        state,
        node="solver_executor",
        step_type="solver",
        duration_s=duration,
        ok=not bool(exec_report.get("diagnosis_required")),
        error_type=err_details.get("error_type") if exec_report.get("diagnosis_required") else None,
        error_message=err_details.get("error_message") if exec_report.get("diagnosis_required") else None,
        artifact={
            "gurobi_status": exec_report.get("gurobi_status"),
            "objective_value": obj,
            "diagnosis_required": exec_report.get("diagnosis_required"),
            "error_category": exec_report.get("error_category"),
            "gurobi_structure": structure,
        },
        artifact_name=f"solver_r{state.get('current_round', 1)}",
        extra={
            "gurobi_status": exec_report.get("gurobi_status") or "",
            "obj": obj,
            "structure": structure,
            "diagnosis_required": exec_report.get("diagnosis_required"),
            "error_category": exec_report.get("error_category"),
        },
    )
    node_metrics = state.get("node_metrics", {})
    if "solver_executor" not in node_metrics:
        node_metrics["solver_executor"] = {
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "num_calls": 0,
        }
    node_metrics["solver_executor"]["total_duration_s"] += duration
    node_metrics["solver_executor"]["num_calls"] += 1

    # Record output to history
    current_round = state.get("current_round", 1)
    output_history = record_agent_output(
        output_history=state.get("output_history", []),
        agent_name="solver_executor",
        output=exec_report,
        current_round=current_round,
        step_type="forward"
    )

    # Build stack_trace for diagnosis: for execution errors use the real traceback,
    # for optimization errors build a summary so diagnosis agent has something to analyze.
    error_details = exec_report.get("error_details") or {}
    stack_trace = error_details.get("stack_trace")
    if not stack_trace and exec_report.get("diagnosis_required"):
        result_summary = exec_report.get("result") or {}
        stack_trace = json.dumps({
            "gurobi_status": exec_report.get("gurobi_status"),
            "objective_value": result_summary.get("objective_value"),
            "variable_count": len(result_summary.get("variables") or {}),
        }, ensure_ascii=False, default=str)

    return {
        "execution_result": exec_report,
        "error_info": error_info,
        "error_category": exec_report.get("error_category"),
        "gurobi_status": exec_report.get("gurobi_status"),
        "stack_trace": stack_trace,
        "data_engineer_output": data_engineer_output,
        "data_access_guide": state.get("data_access_guide"),
        "model_expert_output": state.get("model_expert_output"),
        "python_code": python_code,
        "output_history": output_history,
        "retry_count": state.get("retry_count", 0),
        "current_round": current_round,
        "step_metrics": step_metrics,
        "node_metrics": node_metrics,
        "total_duration_s": state.get("total_duration_s", 0.0) + duration,
    }
