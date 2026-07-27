"""Evidence-informed layer ranking for Stackelberg commitment.

Leader forms a layer ranking from failure evidence (stack + agent history),
then commits ω. Ranking is an ordered list — never calibrated LLM
probabilities (ignore confidence / probability fields if emitted).
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Optional

from langchain_community.callbacks import get_openai_callback

from ..prompts import DIAGNOSIS_ROLE, LAYER_RANK_PROMPT, get_diagnosis_prompt
from ..schemas import LayerRankOutput
from ..utils.llm_config import DEFAULT_MODEL, DEFAULT_PROVIDER, get_llm

logger = logging.getLogger(__name__)

Layer = Literal["data_engineer", "model_expert", "python_developer"]
# Local copy avoids circular import via game.inspection → diagnosis_agent.
LAYERS: tuple[str, ...] = (
    "data_engineer",
    "model_expert",
    "python_developer",
)


@dataclass
class RankResult:
    """Best-first layer ranking at commit time."""

    rank: list[str]  # best-first permutation of three layers
    method: str  # "llm_rank" | "heuristic" | "hybrid"
    rationale: str  # short text for logs (not a probability)
    raw: dict = field(default_factory=dict)  # debug only; never treat as P(·)
    tokens: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def repair_rank(rank: Optional[List[Any]]) -> list[str]:
    """Force a permutation of the three layers; append missing in causal order."""
    seen: list[str] = []
    for item in rank or []:
        name = str(item).strip()
        if name in LAYERS and name not in seen:
            seen.append(name)
    for layer in LAYERS:
        if layer not in seen:
            seen.append(layer)
    return seen


def align_omega(
    *,
    probe_order: str,
    rank: Optional[List[str]] = None,
    gurobi_status: str = "",
    omega_source: str = "evidence_rank",
    probe_seed: int | None = None,
) -> list[str]:
    """Align ranking with ablation mode and produce committed ω.

    Exp-A freeze (SPEC §4.1):
      - causal + evidence_rank: rotate causal cycle so rank[0] is first
      - reverse: fixed PD→ME→DE (ignore rank)
      - random: seeded shuffle of {DE,ME,PD} (ignore rank)
      - status_prior: legacy Prior(status) rotate (causal only)
    """
    # Lazy imports: diagnosis_agent owns legacy Prior(status) helpers.
    from ..agents.diagnosis_agent import build_probe_order, get_candidate_agents

    if omega_source == "status_prior":
        return build_probe_order(
            gurobi_status,
            probe_order,
            probe_seed=probe_seed,
        )

    # evidence_rank (default)
    if probe_order == "reverse":
        return list(reversed(LAYERS))

    if probe_order == "random":
        omega = list(LAYERS)
        rng = random.Random(probe_seed) if probe_seed is not None else random.Random()
        rng.shuffle(omega)
        return omega

    # causal: rotate so tip = rank[0] (fallback: status prior tip)
    tip = None
    repaired = repair_rank(rank)
    if repaired:
        tip = repaired[0]
    if tip not in LAYERS:
        tip = get_candidate_agents(gurobi_status)[0]
    omega = list(LAYERS)
    i = omega.index(tip)
    return omega[i:] + omega[:i]


def rank_layers(state: dict, *, method: str = "llm_rank") -> RankResult:
    """Produce a best-first layer ranking from the failure blackboard."""
    method = (method or "llm_rank").strip().lower()
    if method == "heuristic":
        return _heuristic_rank(state)
    if method == "hybrid":
        # Heuristic tip as prior; LLM may reorder. Fall back to heuristic tip
        # if LLM fails. Prefer heuristic tip when force-zero smell is strong.
        heur = _heuristic_rank(state)
        if heur.raw.get("force_zero_sea_smell"):
            return RankResult(
                rank=list(heur.rank),
                method="hybrid",
                rationale=f"hybrid:force_zero→heuristic; {heur.rationale}",
                raw={**heur.raw, "hybrid_mode": "force_zero_heuristic"},
                tokens=0,
                duration_s=heur.duration_s,
            )
        llm = _llm_rank(state, method_label="hybrid")
        return llm
    return _llm_rank(state, method_label="llm_rank")


def _text_blob(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _force_zero_sea_smell(text: str) -> bool:
    """True if text encodes forced y_in/y_out = 0 (plant or coded symptom)."""
    t = (text or "").lower()
    if not t:
        return False
    if "injected_force_zero" in t or "force_zero_sea" in t:
        return True
    if "force_zero_yin" in t or "force_zero_yout" in t:
        return True
    if "force" in t and "zero" in t and ("y_in" in t or "y_out" in t or "sea" in t):
        return True
    # Coded constraints: y_in[...] == 0 together with y_out[...] == 0
    if "y_in" in t and "y_out" in t and "== 0" in t.replace("==0", "== 0"):
        return True
    return False


def _heuristic_rank(state: dict) -> RankResult:
    """Cheap feature scores; ties broken by causal order. No LLM tokens."""
    from ..agents.diagnosis_agent import get_candidate_agents

    scores = {layer: 0.0 for layer in LAYERS}
    gurobi_status = (state.get("gurobi_status") or "").strip().upper()
    stack = (state.get("stack_trace") or "") + "\n" + (state.get("error_info") or "")
    stack_l = stack.lower()
    me_text = _text_blob(state.get("model_expert_output"))
    code_text = state.get("python_code") or ""
    force_zero = _force_zero_sea_smell(me_text) or _force_zero_sea_smell(code_text)

    # Status prior: tip gets +3, second +1 — but do not let CRASH prior
    # overpower force-zero / WRONG_OBJ modeling evidence (below).
    prior = get_candidate_agents(gurobi_status)
    for i, layer in enumerate(prior):
        scores[layer] = scores.get(layer, 0.0) + max(0.0, 3.0 - i)

    # Traceback / error mentions (weak signal; overridden by force-zero)
    mention_map = {
        "python_developer": (
            "traceback", "syntaxerror", "nameerror", "typeerror", "keyerror",
            "attributeerror", "python_developer", "gurobipy", "line ",
        ),
        "model_expert": (
            "model_expert", "constraint", "infeasible", "unbounded",
            "objective", "formulation", "variable",
        ),
        "data_engineer": (
            "data_engineer", "data_access", "sample.json", "keyerror",
            "missing key", "parameter",
        ),
    }
    for layer, keys in mention_map.items():
        for key in keys:
            if key in stack_l:
                scores[layer] += 1.0

    # Empty / missing stage outputs
    if not state.get("python_code"):
        scores["python_developer"] += 2.0
    if not state.get("model_expert_output"):
        scores["model_expert"] += 2.0
    if not state.get("data_engineer_output") and not state.get("data_access_guide"):
        scores["data_engineer"] += 2.0

    # Modeling keywords in ME output
    me_l = me_text.lower()
    for kw in ("y_in", "y_out", "sea", "force", "zero"):
        if kw in me_l:
            scores["model_expert"] += 0.5

    # Force-zero / empty-sea plant smell → tip ME (traceback ≠ root)
    if force_zero:
        scores["model_expert"] += 6.0
        # Downweight PD traceback-as-oracle when the plant is in ME/code.
        scores["python_developer"] -= 2.0

    # OPTIMAL / WRONG_OBJ: prefer formulation over crash locus
    if gurobi_status in ("OPTIMAL", "SUBOPTIMAL") or (
        not gurobi_status and "gap" in stack_l
    ):
        scores["model_expert"] += 1.5

    # Stable sort: higher score first; ties → causal order
    causal_index = {layer: i for i, layer in enumerate(LAYERS)}
    ranked = sorted(
        LAYERS,
        key=lambda layer: (-scores[layer], causal_index[layer]),
    )
    rationale = (
        "heuristic scores="
        + json.dumps({k: round(v, 2) for k, v in scores.items()}, ensure_ascii=False)
        + ("; force_zero_sea_smell=1" if force_zero else "")
    )
    return RankResult(
        rank=list(ranked),
        method="heuristic",
        rationale=rationale,
        raw={"scores": scores, "prior": prior, "force_zero_sea_smell": force_zero},
    )


def _llm_rank(state: dict, *, method_label: str = "llm_rank") -> RankResult:
    """One LLM call → ordered list only (same evidence budget as adversarial)."""
    start = time.time()
    tokens = 0
    # Lazy import to avoid circular imports at module load.
    from ..agents.diagnosis_agent import _build_diagnosis_context

    context = _build_diagnosis_context(state)
    gurobi_status = state.get("gurobi_status", "")
    candidate_agents = list(LAYERS)

    def _escape_braces(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    task = LAYER_RANK_PROMPT.format(
        gurobi_status=gurobi_status,
        stack_trace=state.get("stack_trace", "Not available"),
        error_location=_escape_braces(
            json.dumps(context.get("error_location", {}), ensure_ascii=False, indent=2)
        ),
        python_code=context.get("agent_outputs", {}).get("python_developer", "Not available"),
        model_expert_output=context.get("agent_outputs", {}).get("model_expert", "Not available"),
        data_access_guide=_escape_braces(
            json.dumps(context.get("data_access_guide", {}), ensure_ascii=False, indent=2)
        ),
        error_key_analysis=_escape_braces(
            json.dumps(context.get("error_key_analysis", {}), ensure_ascii=False, indent=2)
        ),
        candidate_agents=candidate_agents,
    )

    provider = state.get("provider") or DEFAULT_PROVIDER
    model = state.get("model") or DEFAULT_MODEL
    llm = get_llm(provider=provider, model=model, temperature=0)
    prompt = get_diagnosis_prompt()
    chain = prompt | llm.with_structured_output(LayerRankOutput, method="json_mode")

    raw: dict[str, Any] = {}
    try:
        with get_openai_callback() as cb:
            out: LayerRankOutput = chain.invoke({
                "role": DIAGNOSIS_ROLE,
                "task": task,
            })
        tokens = int(cb.total_tokens or 0)
        raw = out.model_dump() if hasattr(out, "model_dump") else dict(out)
        # Ignore any probability / confidence fields if present in raw dump.
        raw.pop("confidence", None)
        raw.pop("probability", None)
        raw.pop("probabilities", None)
        rank = repair_rank(getattr(out, "rank", None))
        rationale = (getattr(out, "rationale", None) or "").strip() or "llm_rank"
    except Exception as exc:
        logger.error("rank_layers(llm_rank) failed: %s; falling back to heuristic", exc)
        fallback = _heuristic_rank(state)
        fallback.method = method_label
        fallback.rationale = f"llm_rank failed ({exc}); {fallback.rationale}"
        fallback.raw = {**fallback.raw, "error": str(exc)}
        fallback.tokens = tokens
        fallback.duration_s = time.time() - start
        return fallback

    return RankResult(
        rank=rank,
        method=method_label,
        rationale=rationale,
        raw=raw,
        tokens=tokens,
        duration_s=time.time() - start,
    )
