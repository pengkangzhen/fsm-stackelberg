"""Pydantic models for structured LLM output.

Reused from mako.schemas.outputs for compatibility.
"""

import json
import re

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any, Union


def repair_json_latex_escapes(text: str) -> str:
    """Fix unescaped LaTeX backslashes so json.loads() can parse the string.

    LLMs commonly emit single backslashes inside JSON strings for LaTeX math
    (e.g. ``\\delta``, ``\\text{}``) which are invalid JSON escape sequences
    and crash the parser.  This function doubles every backslash that is NOT
    already part of a recognised JSON escape sequence (``\\\\", ``\\\"", etc.).

    The naive ``val.replace("\\\\", "\\\\\\\\")`` approach breaks already-escaped
    quotes (``\\"`` → ``\\\\"`` → parse error) and is not idempotent.  Using
    a negative-lookahead regex avoids both pitfalls.
    """
    # Match a backslash NOT followed by: another backslash, a double-quote,
    # or standard single-char JSON escapes (n t r b f u).
    return re.sub(r'\\(?![\\"ntrfbu])', r'\\\\', text)


def extract_first_json_collection(text: str) -> Any:
    """Extract the first JSON array/object by bracket-matching.

    Some models (notably GLM-5.2 under complex prompts) emit a list field
    whose string value has trailing garbage, e.g. ``'["a","b"]}'``. Standard
    ``json.loads`` fails on the trailing ``}``; this function locates the
    balanced ``[...]`` or ``{...}`` starting at the first bracket and parses
    just that substring.
    """
    start = -1
    open_ch = close_ch = ""
    for i, ch in enumerate(text):
        if ch in "[{":
            start, open_ch = i, ch
            close_ch = "]" if ch == "[" else "}"
            break
    if start < 0:
        raise json.JSONDecodeError("no opening bracket", text, 0)
    depth = 0
    in_str = False
    escape = False
    for j in range(start, len(text)):
        ch = text[j]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:j + 1])
    raise json.JSONDecodeError("unbalanced brackets", text, start)


# ============ DataEngineer Output ============


class IndexInfo(BaseModel):
    """Index symbol and its corresponding set."""

    symbol: str
    set: str


class SetDefinition(BaseModel):
    """Definition of a set with its symbol and index."""

    symbol: str
    index: str
    description: str
    source: Optional[str] = Field(
        default=None,
        description="Exact set data_id from the deterministic data catalog.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        """Tolerate common LLM output variations before field-level validation.

        * ``index`` is sometimes emitted as a JSON list (e.g. ``["i", "j"]``)
          instead of a plain string.  We join the elements with a comma.
        * ``source`` is sometimes ``null`` / ``None`` for derived sets even
          though the prompt says to leave it empty-string; we normalise to "".
        """
        if not isinstance(data, dict):
            return data
        # Coerce list index → comma-separated string
        idx = data.get("index")
        if isinstance(idx, list):
            data["index"] = ",".join(str(x) for x in idx)
        # Normalise None source → empty string
        if data.get("source") is None:
            data["source"] = ""
        return data


class ParameterDefinition(BaseModel):
    """Definition of a parameter with indices."""

    symbol: str
    indices: List[IndexInfo]
    description: str
    source: Optional[str] = Field(
        default=None,
        description=(
            "Exact parameter data_id from the deterministic data catalog. "
            "Use derived_parameters, not source=null, for computed values."
        ),
    )
    sparse: bool = False  # True if parameter is a sparse dictionary (records format)
    access_hint: Optional[str] = None  # Guidance for sparse parameter access

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        """Tolerate common LLM output variations before field-level validation.

        * ``indices`` is often emitted as a plain mapping
          ``{"i": "N", "t": "T"}`` instead of the canonical list of
          ``{symbol, set}`` pairs (observed in subagent rehearsal outputs).
          We convert the mapping, preserving insertion order.
        """
        if not isinstance(data, dict):
            return data
        idx = data.get("indices")
        if isinstance(idx, dict):
            data["indices"] = [
                {"symbol": str(k), "set": str(v)} for k, v in idx.items()
            ]
        return data


class DerivedParameter(BaseModel):
    """A parameter that must be computed from other parameters, not directly available in data."""

    symbol: str
    indices: List[IndexInfo]
    description: str
    derivation_logic: str = ""
    source_parameters: List[str] = []


class ModelInputs(BaseModel):
    """Model inputs containing sets, parameters, and derived parameters."""

    sets: List[SetDefinition]
    parameters: List[ParameterDefinition]
    derived_parameters: List[DerivedParameter] = []


class DataEngineerOutput(BaseModel):
    """Structured output for DataEngineer agent."""

    model_inputs: ModelInputs

    @model_validator(mode="before")
    @classmethod
    def repair_llm_output(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # model_inputs returned as JSON string instead of dict
        mi = data.get("model_inputs")
        if isinstance(mi, str):
            # First try strict parse, then apply LaTeX-escape repair
            for attempt in (mi, repair_json_latex_escapes(mi)):
                try:
                    data["model_inputs"] = json.loads(attempt)
                    break
                except (json.JSONDecodeError, ValueError):
                    continue
        # sets[*].source: None → "" (handled by SetDefinition._coerce_fields,
        # but also pre-patch here so the list survives Pydantic coercion)
        mi = data.get("model_inputs", {})
        if isinstance(mi, dict):
            for s in mi.get("sets", []):
                if isinstance(s, dict) and s.get("source") is None:
                    s["source"] = ""
        return data


# ============ ModelExpert Output ============


class DecisionVariable(BaseModel):
    """Definition of a decision variable."""

    symbol: str
    indices: List[str]
    shape: List[str]
    type: str  # "Continuous" | "Integer" | "Binary"
    description: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        """Tolerate common LLM output variations before field-level validation.

        * ``indices`` / ``shape`` are often omitted when the indexing is
          already visible in the symbol (e.g. ``y_in[h,t]``) or described in
          a free-text ``bounds`` field (observed in subagent rehearsal
          outputs).  We derive ``indices`` from the symbol brackets and
          default ``shape`` to the same list (shape is descriptive; the
          ModelComponents validator tolerates mismatched lengths).
        * ``type`` is sometimes lowercase (``continuous``); normalise to
          the capitalised convention used by downstream code generation.
        * ``name`` is sometimes emitted instead of ``symbol``.
        """
        if not isinstance(data, dict):
            return data
        if not data.get("symbol") and data.get("name"):
            data["symbol"] = data["name"]
        symbol = str(data.get("symbol") or "")
        if not data.get("indices"):
            start, end = symbol.find("["), symbol.rfind("]")
            if 0 <= start < end:
                inner = symbol[start + 1 : end]
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                if parts:
                    data["indices"] = parts
        if not data.get("shape") and data.get("indices"):
            data["shape"] = list(data["indices"])
        var_type = data.get("type")
        if isinstance(var_type, str) and var_type:
            data["type"] = var_type.capitalize()
        return data


class ObjectiveFunction(BaseModel):
    """Definition of the objective function."""

    direction: str  # "min" | "max"
    expression: str
    description: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        """Normalise natural-language directions to the canonical min/max.

        LLMs emit ``minimize`` / ``minimum`` / ``maximise`` etc.; the
        ModelComponents consistency check requires exactly ``min``/``max``
        (observed in subagent rehearsal outputs).
        """
        if isinstance(data, dict):
            d = data.get("direction")
            if isinstance(d, str) and d.strip():
                dl = d.strip().lower()
                if dl in {"minimize", "minimise", "min", "minimum"}:
                    data["direction"] = "min"
                elif dl in {"maximize", "maximise", "max", "maximum"}:
                    data["direction"] = "max"
        return data


class Constraint(BaseModel):
    """Definition of a constraint."""

    name: str
    expression: str
    description: str


class ModelComponents(BaseModel):
    """Components of the mathematical model."""

    decision_variables: List[DecisionVariable]
    objective_function: ObjectiveFunction
    constraints: List[Constraint]

    @model_validator(mode="after")
    def validate_consistency(self) -> "ModelComponents":
        for var in self.decision_variables:
            if len(var.indices) != len(var.shape):
                # Relax: shape is descriptive; warn but don't reject
                import logging
                logging.getLogger(__name__).warning(
                    "Decision variable %s has %d indices but %d shape entries — "
                    "proceeding (shape is informational only).",
                    var.symbol, len(var.indices), len(var.shape),
                )
        if self.objective_function.direction not in {"min", "max"}:
            raise ValueError(
                f"Objective direction must be 'min' or 'max', "
                f"got {self.objective_function.direction!r}."
            )
        seen: set[str] = set()
        for constraint in self.constraints:
            if constraint.name in seen:
                raise ValueError(f"Duplicate constraint name: {constraint.name}")
            seen.add(constraint.name)
        return self


class ModelExpertOutput(BaseModel):
    """Structured output for ModelExpert agent."""

    knowledge_requests: List[str] = []
    model_components: ModelComponents
    assumptions_made: List[str] = []
    ambiguous_points: List[str] = []

    @model_validator(mode="before")
    @classmethod
    def repair_llm_output(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # String fields that should be dict/list — MiMo sometimes serializes them
        _str_to_dict_fields = {"model_components"}
        _str_to_list_fields = {"assumptions_made", "ambiguous_points"}
        for field in _str_to_dict_fields | _str_to_list_fields:
            val = data.get(field)
            if isinstance(val, str):
                # Try: strict parse → LaTeX escape repair → bracket-match extraction
                parsed = None
                for attempt_fn in (
                    lambda v: json.loads(v),
                    lambda v: json.loads(repair_json_latex_escapes(v)),
                    lambda v: extract_first_json_collection(v),
                    lambda v: extract_first_json_collection(repair_json_latex_escapes(v)),
                ):
                    try:
                        parsed = attempt_fn(val)
                        break
                    except (json.JSONDecodeError, ValueError):
                        continue
                if parsed is not None:
                    data[field] = parsed
        # Nesting drift: LLMs sometimes hoist objective_function / constraints
        # to the top level as siblings of model_components (first seen live on
        # fam_H4_Omega10, deepseek-flash) — lift them back in.
        mc = data.get("model_components")
        if isinstance(mc, dict):
            for stray in ("objective_function", "constraints"):
                if stray not in mc and stray in data:
                    mc[stray] = data.pop(stray)
        # Flatten nested lists in both list fields (e.g. [["a","b"]] -> ["a","b"])
        for list_field in ("ambiguous_points", "assumptions_made"):
            val = data.get(list_field)
            if isinstance(val, list):
                flattened: list = []
                for item in val:
                    if isinstance(item, list):
                        flattened.extend(str(x) for x in item)
                    else:
                        flattened.append(item)
                data[list_field] = flattened
        return data


# ============ Diagnosis Output ============


class DiagnosisOutput(BaseModel):
    """Structured output for adversarial diagnosis."""

    suspected_agent: str
    confidence: float
    reason: str
    evidence: str = ""  # Specific observations that support the conclusion


class LayerRankOutput(BaseModel):
    """Ordered layer ranking for evidence-informed Stackelberg commit.

    Contract: ordered list only. Do **not** use confidence/probability fields
    even if a model emits them — ranking is not a calibrated posterior.
    """

    rank: List[str] = Field(
        description="Best-first permutation of data_engineer, model_expert, python_developer"
    )
    rationale: str = Field(
        default="",
        description="Short evidence-based justification (not a probability)",
    )


class BackwardStepOutput(BaseModel):
    """Structured output for backward step.

    Used by agents to:
    1. Judge whether the accusation is valid (is_caused_by_you)
    2. Provide corrected output (refined_result) if they accept responsibility

    refined_result type varies by agent:
    - ModelExpert: dict matching ModelExpertOutput structure
    - PythonDeveloper: str containing corrected Python code
    - DataEngineer: dict matching DataEngineerOutput structure
    """

    is_caused_by_you: bool = Field(description="Whether the diagnosed error is actually caused by this agent")
    reason: str = Field(default="", description="Explanation of the analysis")
    refined_result: Optional[Any] = Field(
        default=None,
        description="Corrected output if is_caused_by_you=true. Type varies by agent."
    )
