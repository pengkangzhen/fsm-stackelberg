# HANDOVER — fsm-stackelberg (State-Machine Stackelberg Diagnosis-Repair)

> For the next AI assistant picking this up. Read fully before editing.
> Companion: [`CLAUDE.md`](./CLAUDE.md) (project rules), [`README.md`](./README.md) (quick start).

---

## 0. TL;DR

- **What this project is.** A new paper whose contribution is a **Stackelberg leader-follower diagnosis-repair mechanism defined on a state machine**, for an LLM multi-agent optimization pipeline, validated on the **Empty Container Repositioning (ECR)** problem. The two pillars are in the name: **FSM** (the LangGraph state machine = the stage-game apparatus) + **Stackelberg** (the leader-follower game).
- **Where it came from.** Forked from `mako` @ `fa2cbc8` on **2026-06-30**; package chain renamed `mako_langchain` → `maestro` → `fsm_stackelberg`. It is an **independent codebase** — do not modify `mako` for this paper, do not sync back.
- **Current state.** Phase 0 (mako smoke test) **skipped** — mako already validated end-to-end in the parent project. **Phase 1 PoC is in-tree:** default `--diagnosis_mode stackelberg` commits an inspection policy σ=(ω, ν) and probes layers in causal order; `--diagnosis_mode adversarial|sequential` remain as internal baselines. Open work: payoff logging (Phase 2), external baselines + order ablation runs (Phase 3), attribution experiments (Phase 4), proposition validation (Phase 5).

---

## 1. Why a new paper (the research gap)

MAKO's current "adversarial" diagnosis (in `src/fsm_stackelberg/agents/diagnosis_agent.py`) is, in reality:

1. `get_candidate_agents(gurobi_status)` → a **static heuristic prior** (CRASH → PythonDeveloper first; OPTIMAL-but-wrong-obj → ModelExpert first; …).
2. **One** LLM call → `{suspected_agent, confidence, reason}`.
3. The accused agent's `*_backward_step` self-verifies/repairs; `error_resolved` decides "continue downstream" vs "re-accuse".

There is **no** payoff/utility structure, no equilibrium concept, no leader-follower commitment, no strategic interaction between agents. It is a single-judge classifier with a retry loop.

**This is the gap fsm-stackelberg fills:** elevate diagnosis-repair from a single-judge heuristic to a mechanism with genuine game-theoretic structure, where the gain is **measurable** (root-cause attribution accuracy, SSR-vs-budget, token cost) and **defensible** (a proposition, not just an empirical bump).

> ⚠️ The same gap that makes this *publishable* also creates the trap: a reviewer will ask *"is your Stackelberg a real game, or a payoff gloss on a single-judge LLM?"* Everything in §4–§5 exists to make it real.

---

## 2. The thesis (the actual contribution)

**Commitment order = pipeline causal order.** The modeling pipeline has a natural layered dependency: **data ⊳ model ⊳ code** (errors propagate downstream and contaminate downstream signals). This is precisely a Stackelberg stage:

| Game element | Concretization |
|---|---|
| **Leader** (moves first, commits a strategy) | The upstream agent commits a hypothesis — e.g., ModelExpert commits a formulation fix |
| **Follower** (best-responds) | The diagnosis/verification agent searches the **cheapest refuting test** against the committed hypothesis |
| **Leader utility** | Passes the solver under the follower's strongest attack, at lowest cost |
| **Follower utility** | True root-cause localization accuracy / locks the error layer with fewest tokens |
| **Stage-game apparatus** | The LangGraph **state machine** — each node is a stage, transitions are functions of the strategy profile + shared `AgentState` |

**Why the FSM is a real pillar, not wallpaper.** The state machine is not just "we happen to use LangGraph." It is the **stage-game structure** the Stackelberg game is defined on: states = stage-game states, transitions = strategy-dependent. Treating it as a **Stackelberg Markov game** is what lets the commitment order carry *causal* meaning, and what supports a proposition (e.g., *"under layered error dependence, causal-order commitment strictly lowers the mis-attribution lower bound vs. simultaneous play"*). If you do not formalize the FSM, drop "FSM" from the contribution claim — otherwise a reviewer will say "the state machine is just LangGraph."

**Why this is non-trivial (not "debate with extra steps"):** unlike general LLM debate where any agent can refute any agent, OR/code modeling errors are **layered**. The Stackelberg commitment order maps to that causal layering, so the leader's first-mover commitment has a *causal* meaning. That proposition — even semi-formal + empirically validated — is what separates a real methods paper from a relabel.

**Stronger variant (the rigorous target):** lift the whole FSM to a **Stackelberg Markov game** (states = stage-game states, transitions depend on the strategy profile). Heavier to deliver; this is the version that justifies the "FSM" in the name and is expected at an OR-theory venue.

---

## 3. Current architecture (inherited from mako, unchanged)

### 3.1 The state machine (`src/fsm_stackelberg/graph/workflow.py`)

A LangGraph `StateGraph` — i.e., an Extended Finite State Machine:

- **States (nodes):** `data_engineer`, `model_expert`, `knowledge_loader`, `python_developer`, `solver_executor`, `diagnosis_agent`, plus three backward variants `data_engineer_backward`, `model_expert_backward`, `python_developer_backward`.
- **Shared blackboard:** `AgentState` (`src/fsm_stackelberg/graph/state.py`) — a 74-line TypedDict mixing control state (`retry_count`, `error_agent`, `error_resolved`, `diagnosis_mode`, `current_round`) with data payload (agent outputs, metrics, knowledge).
- **Transitions** = pure functions of state → next-node name:

| Router | File:line | Decides from | Branches |
|---|---|---|---|
| `route_after_model_expert` | `workflow.py:166` | `knowledge_requests` | knowledge_loader / python_developer |
| `route_after_knowledge_loader` | `workflow.py:181` | `knowledge_loader_loaded` | model_expert (loop) / python_developer |
| `should_diagnose` | `workflow.py:45` | `diagnosis_required`, `retry_count` | END / diagnosis_agent |
| `route_after_diagnosis` | `workflow.py:73` | `error_agent`, `diagnosis_mode` | one of `*_backward` / END |
| `route_after_backward` | `workflow.py:109` | `error_resolved`, `error_agent` | downstream resume / re-diagnose / END |

Entry: `data_engineer`. Termination: `END`. Retry budget `max_retries` is the FSM's transition budget.

> Note: `create_mako_graph()` / `run_mako()` retain mako-era **function names** (the rename only touched the package token). Cosmetic rename is a TODO — low priority, but do it before the codebase grows.

### 3.2 The diagnosis logic

- `src/fsm_stackelberg/agents/diagnosis_agent.py`
  - `diagnosis_agent_node` — the FSM node; dispatches by `diagnosis_mode`.
  - `_stackelberg_diagnosis` — **default**: inspector commits σ and selects next uncleared layer (no single-judge LLM).
  - `_adversarial_diagnosis` — single-judge LLM baseline (kept for comparison).
  - `get_candidate_agents` / `build_probe_order` — Prior(status) seed + committed ω.
  - `_build_diagnosis_context` — signal extraction (still used by adversarial baseline).
- `src/fsm_stackelberg/agents/{data_engineer,model_expert,python_developer}.py` — each has a `*_backward_step` node; `error_resolved` + `backward_reason` are the inspectee signals (comply vs deflect).

### 3.3 Ground truth & data

- `src/generator/ground_truth_solver.py` — provides `expected_value` for the strict-success rule (`OPTIMAL and gap_percent <= 1e-3`). This is **fsm-stackelberg's own copy** — free to modify, though you likely won't need to (single-commodity ECR).
- Instance: `dataset/prob_ecr_shipper_consignee/instances/high_demand_5-3_5` (16 nodes, 173 arcs, 5 periods, optimum **¥14,386,797**). The dir literally named `small_5-3_5` is unusable — always run on `high_demand_5-3_5`.

---

## 4. The build plan

### Phase 0 — Smoke test — SKIPPED
Mako already validated end-to-end in the parent project; do not re-run a mako-parity smoke test here. Proceed directly with the Stackelberg upgrade.

### Phase 1 — Minimal viable Stackelberg (PoC) — DONE (skeleton)
Implemented as an **inspection game** matching the manuscript (not the older HANDOVER leader=upstream sketch):
1. **Inspector (DiagnosisAgent) commits** σ=(ω, ν): probing order ω seeded by `Prior(status)`, verification rule ν = executed refutation (re-solve via resume→solver).
2. **Inspectee best-responds** in `*_backward_step`: comply-repair (`error_resolved=True`) or deflect (`False`).
3. **Route from the verdict:** comply → resume downstream (re-solve is ν); deflect / refuted comply → clear layer, `NextCausalLayer`.

CLI: `--diagnosis_mode stackelberg` (default), `--probe_order causal|reverse|random` (ablation hook). Baselines: `adversarial`, `sequential`.

Still thin vs. the full paper claim: no numeric payoff logging yet; deflection verification is “clear + descend” rather than a separate targeted isolating re-solve.

### Phase 2 — Payoff definition (make the game real)
Ground utility in signals already on `AgentState`:
- Follower payoff = whether the chosen test localizes the *true* root-cause layer (ground truth available — see §5).
- Leader payoff = solver passes after the committed fix, minus a token/round cost.
No LLM "feels" a utility — you either (a) define a numeric payoff the orchestrator optimizes, or (b) treat each LLM call as a black-box best-responder and analyze the resulting trajectory. Be explicit about which.

### Phase 3 — Baselines (fair comparison is mandatory)
The Stackelberg mechanism must beat:
- **Internal:** mako's `sequential` mode and the current `adversarial` (single-judge) mode — both already in-tree.
- **External (build fresh):** multi-agent **debate** (cf. Du et al. 2023) and **Reflexion**-style self-critique. These are the direct competitors; CoE/OptiMUS (in `src/fsm_stackelberg/baselines/`) are a *different* family (multi-agent collaboration, not verification) — useful as context, not as the head-to-head.
- **Ablation (the key differentiator):** **commitment order** — random/permuted leader-follower order vs causal order. This isolates the contribution from "just calling the LLM more times."

### Phase 4 — Experiments
Primary metric: **root-cause attribution accuracy** (does the mechanism correctly localize the failing layer?). Infrastructure for this exists in mako as `scripts/experiment_phase4/analyze_2d_attribution.py` (solver-status × origin-agent) — **port it into fsm-stackelberg**; it provides the ground-truth labels.
Secondary: SSR-vs-K curve, token cost, convergence rounds, per-error-category breakdown.

### Phase 5 — At least one proposition
Even semi-formal: e.g., under layered error dependence, causal-order Stackelberg commitment dominates simultaneous play on mis-attribution rate. Prove a bound or validate empirically with tight CIs.

---

## 5. Claim boundaries (write these into the paper)

- fsm-stackelberg targets **layered/structured-error NLO** (OR modeling with a data→model→code dependency). Do **not** claim gains for general LLM reasoning — that invites unwinnable comparisons.
- The Stackelberg framing is **domain-grounded**, not a universal agent paradigm.
- The "FSM" pillar is a contribution **only if** formalized as a Stackelberg Markov/stage game (§2). Otherwise claim only the Stackelberg mechanism.
- Human-in-the-loop framing is inherited; the planner remains specifier + final arbiter.

---

## 6. Threats to validity (keep in mind throughout)

1. **LLMs have no stable utility function.** Either make the payoff a real number the orchestrator optimizes, or treat LLMs as black-box best-responders. An OR venue will challenge any "equilibrium" claim built on vibes.
2. **Commitment-order ablation is the experiment.** If random-order performs as well as causal-order, the thesis is dead — find out early.
3. **Verification cost.** Stackelberg needs a follower best-response each round = extra LLM calls. Show accuracy gain justifies cost.
4. **"FSM = just LangGraph" critique.** The state machine is novel only via the game defined on it; preempt this in the paper.

---

## 7. Open decisions

| Decision | Default | Notes |
|---|---|---|
| Package function rename (`create_mako_graph` / `run_mako` → ?) | defer | cosmetic; do once codebase stabilizes |
| Prune `src/fsm_stackelberg/baselines/` + `experiments/` | defer | mako-paper artifacts; safe to remove for a lean core (nothing in core imports them) |
| ECR model: single- vs multi-commodity | **single** (default) | multi-commodity lives in `mako/paper-imhfc/`; port only if the paper needs it |
| Target venue | undecided | Stackelberg+FSM rigor → EJOR / Computers & OR; mechanism novelty → AAAI-Agent; application → EAAI |
| Paper system name | independent of repo name | repo = `fsm-stackelberg`; the paper may brand the system differently (or keep descriptive) |

---

## 8. Environment checklist

- [ ] `uv sync` (done — venv at `.venv/`)
- [ ] Copy `.env` from `mako` (API keys for the providers you use; `python-dotenv` loads it). Key names match whatever `mako/.env` uses.
- [ ] Gurobi license active (same as mako; `gurobipy>=12.0.2`).
- [ ] `git init` + initial commit when ready (the scaffold is not yet a git repo).

---

## 9. File map (cheat sheet)

| Purpose | Path |
|---|---|
| FSM construction | `src/fsm_stackelberg/graph/workflow.py` (`create_mako_graph` :194) |
| Shared state | `src/fsm_stackelberg/graph/state.py` (`AgentState`) |
| **Diagnosis (Stackelberg inspector)** | `src/fsm_stackelberg/agents/diagnosis_agent.py` (`_stackelberg_diagnosis`, `_adversarial_diagnosis`, `build_probe_order`) |
| Follower-side repair | `src/fsm_stackelberg/agents/{data_engineer,model_expert,python_developer}.py` (`*_backward_step`) |
| CLI entry | `src/fsm_stackelberg/main.py` |
| Ground truth | `src/generator/ground_truth_solver.py` |
| Working instance | `dataset/prob_ecr_shipper_consignee/instances/high_demand_5-3_5` |
| Knowledge modules | `src/fsm_stackelberg/knowledge/` |
