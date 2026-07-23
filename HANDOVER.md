# HANDOVER — fsm-stackelberg (State-Machine Stackelberg Diagnosis-Repair)

> For the next AI assistant picking this up. Read fully before editing.
> Companion: [`README.md`](./README.md) (quick start), manuscript
> [`els-cas-templates/manuscript.tex`](./els-cas-templates/manuscript.tex).

---

## 0. TL;DR

- **What this project is.** A paper whose contribution is a **Stackelberg
  inspection game for diagnosis–repair defined on a state machine**, for an
  LLM multi-agent optimization pipeline, validated on **demand-uncertainty
  sea–land ECR** as a **two-stage stochastic LP** ([D1]–[D2] DEP) from
  `tslp-ecr-demand`. Pillars: **FSM** (LangGraph stage-game apparatus) +
  **Stackelberg** (inspector commits \(\sigma=(\omega,\nu)\); inspectees
  comply/deflect).
- **Where it came from.** Workflow forked from `mako` @ `fa2cbc8`
  (2026-06-30); package `mako_langchain` → `maestro` → `fsm_stackelberg`.
  Application model from sibling `tslp-ecr-demand` + `ecr-shared-data`.
  **Independent codebase** — do not modify `mako` / do not sync back.
  Mako-era `prob_ecr_shipper_consignee` instances have been **removed**.
- **Repo.** `git@github.com:pengkangzhen/fsm-stackelberg.git`, branch `main`
  (includes merge `869752b` structured run logging).
- **Current state (2026-07-23).**
  - Phase 0 skipped; Phase 1 Stackelberg PoC **done**; Phase 2 analysis-mode
    payoffs **done** (`src/fsm_stackelberg/game/payoff.py`), including
    commitment diagnostics (`committed_omega`, `first_probe_hit` / `kill_hit`,
    `attribution_hit`) for Exp-I.
  - **Progressive knowledge injection** (catalog → request → full text; not
    RAG, not dump-all) is the default via
    `src/fsm_stackelberg/knowledge/progressive.py`, wired through
    `plugins.FeatureBundle`. Heuristic full-text preload has been **removed**.
  - **Structured run logging done** (`src/fsm_stackelberg/utils/run_log.py`):
    each run writes `run_manifest.json` (config + terminal metrics),
    `events.jsonl` (per-node tokens/duration/diagnosis/comply–deflect), and
    `artifacts/` (diagnosis / backward / inject). Optional `--log_prompts`.
    **Machine-readable authority = manifest + events**, not `workflow.log`
    (human mirror only). Summarize / Exp-I scripts should read those files.
  - Default dataset: `prob_tslp_ecr_demand` / `smoke_H4_Omega5`.
  - Manuscript method §inspection game drafted; Experiments protocol
    **written** (Setup → Methods → Metrics → Exp-I–IV), results empty.
  - **Next concrete work (ordered):**
    1. **Smoke E2E on TSLP** — restore a working LLM key, then run
       `smoke_H4_Omega5` with `--knowledge progressive` and confirm ME can
       request/load modules and the solve/diagnosis loop completes
       (also verify `run_manifest.json` / `events.jsonl` land under `results/`).
    2. Exp-I pilot (kill criteria) — fault injection +
       `stackelberg × {causal, reverse, random}` vs `adversarial`
       (aggregate `first_probe_hit` from manifests).
    3. Exp-II; implement Debate + Reflexion; Exp-III/IV.

---

## 1. Why a new paper (the research gap)

MAKO's current "adversarial" diagnosis (in
`src/fsm_stackelberg/agents/diagnosis_agent.py`) is, in reality:

1. `get_candidate_agents(gurobi_status)` → a **static heuristic prior**
   (CRASH → PythonDeveloper first; OPTIMAL-but-wrong-obj → ModelExpert first; …).
2. **One** LLM call → `{suspected_agent, confidence, reason}`.
3. The accused agent's `*_backward_step` self-verifies/repairs;
   `error_resolved` decides "continue downstream" vs "re-accuse".

There is **no** payoff/utility structure, no equilibrium concept, no
leader–follower commitment, no strategic interaction between agents. It is a
**single-judge** classifier with a retry loop.

**This is the gap fsm-stackelberg fills:** elevate diagnosis–repair from a
single-judge heuristic to a mechanism with genuine game-theoretic structure,
where the gain is **measurable** (root-cause attribution accuracy, SSR-vs-budget,
token cost) and **defensible** (a proposition + commitment-order ablation).

> ⚠️ Reviewer trap: *"is your Stackelberg a real game, or a payoff gloss on a
> single-judge LLM?"* Ablation + executed refutation exist to answer that.
>
> ⚠️ Strong-LLM trap: debate / full-context accusation may also attribute well.
> Do **not** rest the paper on "others cannot diagnose." Rest it on
> **causal-order commitment** improving **verifiable** attribution (and/or
> cost). If `causal ≈ random`, the thesis is dead — find out in Exp-I pilot.

**Motivation already in the manuscript:** under layered errors, stack traces /
solver surfaces mark the **symptom locus** (often code), not the certified
root-cause layer — so one cannot treat the traceback as an oracle.

---

## 2. The thesis (match the manuscript — inspection game)

**Inspector = DiagnosisAgent (leader).** Commits first and observably to
\(\sigma=(\omega,\nu)\): probing order \(\omega\) (default = causal
data ⊳ model ⊳ code, seeded by `Prior(status)`) and verification rule \(\nu\) =
**executed refutation** (real re-solve under strict success), not cheap talk.

**Inspectees = DE / ME / PD (followers).** When probed, choose
**comply** (repair) or **deflect**. Interests diverge under layered faults:
innocent downstream layers prefer deflect; guilty layers should comply when
deflection cannot survive \(\nu\).

| Game element | Concretization (current paper) |
|---|---|
| Leader / inspector | `DiagnosisAgent` commits \(\sigma=(\omega,\nu)\) |
| Followers / inspectees | Accused `data_engineer` / `model_expert` / `python_developer` |
| Leader strategy | Committed \(\omega\) + executed re-solve \(\nu\) |
| Follower strategy | `comply` \| `deflect` in `*_backward_step` |
| Payoffs | Analysis-mode \(u_L,u_F\) in `game/payoff.py` (evaluate trajectories; LLMs are black-box best-responders) |
| Stage-game apparatus | LangGraph FSM; \(\delta_{\mathrm{dg}}\) / \(\delta^{-}\) depend on the verdict |

> Older drafts (and an outdated row in early HANDOVER) cast the **upstream
> agent** as leader. **That is wrong for the current paper.** Do not revive it.

**Why not "just debate":** debate is multi-call attribution without a committed
causal probing policy or inspection payoffs. External baselines (Debate,
Reflexion) test whether gains collapse to "calling the LLM more times."

**Prop. 1 (manuscript):** under strict layered error dependence, causal-order
executed-refutation inspection reduces the viable root-cause set relative to
simultaneous / single-judge play without dropping the true layer. Empirical
spine = **commitment-order ablation**.

---

## 3. Current architecture

### 3.1 The state machine (`src/fsm_stackelberg/graph/workflow.py`)

LangGraph `StateGraph` (extended FSM):

- **Nodes:** `data_engineer`, `model_expert`, `knowledge_loader`,
  `python_developer`, `solver_executor`, `diagnosis_agent`, plus
  `*_backward` for DE/ME/PD.
- **Blackboard:** `AgentState` — control (`retry_count`, `error_agent`,
  `error_resolved`, `diagnosis_mode`, `probe_order`, `inspection_policy`,
  `probe_queue`, `cleared_layers`, …) + payload + `episode_payoff` /
  `true_root_cause` / `attributed_layer`.
- **Routers:** `should_diagnose`, `route_after_diagnosis`,
  `route_after_backward`, knowledge-load branches. Budget \(K\) =
  `max_retries`.

> Cosmetic debt: `create_mako_graph()` / `run_mako()` still use mako-era names.

### 3.2 Diagnosis logic

- `diagnosis_agent.py` + `game/inspection.py` (re-exports probe helpers)
  - `_stackelberg_diagnosis` — **default**: commit \(\sigma\), probe next
    uncleared layer (**no** single-judge LLM).
  - `_adversarial_diagnosis` — single-judge baseline.
  - `get_candidate_agents` / `build_probe_order` — Prior(status) +
    `causal|reverse|random`.
- Inspectee signals: `error_resolved` + `backward_reason` (comply vs deflect).
- **Not yet implemented:** `debate`, `reflexion` diagnosis modes (Exp-III).

### 3.2b Progressive knowledge (scaffolding, not the paper claim)

- `knowledge/progressive.py` — catalog-first on-demand injection.
- `plugins/features.py` — `FeatureBundle` composes knowledge + diagnosis mode
  without coupling their internals into the graph routers.
- Flow: ME sees **catalog only** → `knowledge_requests` →
  `knowledge_loader` node injects **full text** of named modules → ME again
  (≤ `knowledge_max_rounds`). CLI: `--knowledge progressive|enable|disable`
  (`enable` ≡ progressive).
- Domain modules under `knowledge/domains/tslp/` remain required for correct
  TSLP modeling; they are **not** marketed as the research contribution.

### 3.3 Ground truth & data

- `src/generator/` — TSLP export (`serialize.py`, `cli.py`) + DEP GT
  (`ground_truth_solver.py` → `tslp_ecr_demand.dep.solve_dep`).
- Dataset: `dataset/prob_tslp_ecr_demand/` (+ `smoke_H4_Omega5`).
- Knowledge: `src/fsm_stackelberg/knowledge/domains/tslp/`.
- **Do not** revive `prob_ecr_shipper_consignee`.

---

## 4. Build plan ↔ manuscript experiments

### Phase 0 — Smoke test — SKIPPED

### Phase 1 — Stackelberg PoC — DONE

Inspector commits \(\sigma\); inspectee comply/deflect; route on verdict.
CLI: `--diagnosis_mode stackelberg|adversarial|sequential`,
`--probe_order causal|reverse|random`.

Known thin spot vs full paper: deflect path is largely “clear + descend,” not
a separate targeted isolating re-solve for every deflection claim.

### Phase 2 — Payoffs — DONE (analysis-mode)

`src/fsm_stackelberg/game/payoff.py` → `AgentState.episode_payoff` at end of
`run_mako` / experiment JSON:

\[
u_F = \mathbf{1}\{\hat a = a^\*\} - \lambda_K K - \lambda_C (C/C_0),\quad
u_L = S - \mu_K K - \mu_C (C/C_0)
\]

Without \(a^\*\) (`--true_root_cause`), `u_F` is `null`. Tests:
`tests/test_payoff.py`.

### Phase 3 / manuscript Exp-I–III — OPEN (next)

Manuscript protocol (`§Experiments`):

| Block | Content |
|---|---|
| **Setup** | TSLP data, fixed model, budget \(K\), upstream injection / downstream symptom, fairness (same blackboard + repair path) |
| **Methods** | (A) stackelberg × {causal, reverse, random}; (B) adversarial, sequential; (C) Debate + Reflexion |
| **Metrics** | Primary: attribution \(\mathbf{1}\{\hat a=a^\*\}\); secondary: SSR@\(K\), tokens/rounds; diagnostics in Exp-IV |
| **Exp-I** | Commitment-order ablation (+ **pilot / kill criteria** — written; may delete later) |
| **Exp-II** | Internal baselines |
| **Exp-III** | Debate (majority vote, **no LLM confidence weights**) + Reflexion |
| **Exp-IV** | Deflect / overturn rates, attribution vs \(K\) |

**Immediate next task:** (1) restore a working LLM key and complete
smoke E2E on `smoke_H4_Omega5`; (2) then Exp-I pilot injection + runner.

Debate design note (agreed): debaters output `{suspected_agent, argument}`
only; aggregate by majority vote; tie-break via status prior — **do not**
trust LLM-reported confidence.

### Phase 4 — Attribution grid — OPEN

Primary metric already fixed in manuscript. Port / rebuild injection +
analysis (mako had `scripts/experiment_phase4/analyze_2d_attribution.py` —
adapt to TSLP + fsm-stackelberg).

### Phase 5 — Proposition — PARTIAL

Prop. 1 is in the manuscript; formal bound optional; **empirical validation =
Exp-I**. Do not claim the prop is validated until ablation lands.

---

## 5. Claim boundaries

- Target **layered-error NLO** (data → model → code), not general LLM reasoning.
- Stackelberg is **domain-grounded** inspection, not a universal agent paradigm.
- FSM counts as a contribution only as stage-game apparatus for the inspection
  game; otherwise drop "FSM" from the claim.
- Prefer **"verifiable attribution under causal-order commitment"** over
  **"only we can diagnose"** (strong models will falsify the latter).

---

## 6. Threats to validity

1. **LLMs have no stable utility** — analysis-mode payoffs only; no vibe equilibria.
2. **Commitment-order ablation is the experiment** — `causal ≈ random` kills the thesis; run Exp-I pilot early.
3. **Strong LLM / Debate ceiling** — may match attribution; then defend cost, verifiability, and ablation — or narrow the claim.
4. **Verification cost** — show accuracy (or auditability) justifies extra calls.
5. **"FSM = just LangGraph"** — preempt via game-on-FSM formalization in the paper.

---

## 7. Open decisions

| Decision | Default | Notes |
|---|---|---|
| ECR application model | **TSLP demand-uncertainty DEP** | Not mako MCNF |
| Package rename (`create_mako_graph` / `run_mako`) | defer | cosmetic |
| Prune CoE/OptiMUS under `baselines/` | defer | different family; not Exp-III head-to-head |
| Target venue | undecided | EJOR / C&OR vs agent venue vs EAAI |
| Paper system name | undecided | may differ from repo name |
| Exp-I pilot subsection in tex | keep for now | user OK to delete after main tables |

---

## 8. Environment checklist

- [x] `uv sync` (venv at `.venv/`)
- [ ] `.env` with API keys (`python-dotenv`; copy from `mako` as needed)
- [ ] Gurobi license active (`gurobipy>=12.0.2`)
- [x] GitHub remote in use (`origin` → `pengkangzhen/fsm-stackelberg`)

---

## 9. File map (cheat sheet)

| Purpose | Path |
|---|---|
| Manuscript | `els-cas-templates/manuscript.tex` (§stackelberg, §experiments) |
| FSM | `src/fsm_stackelberg/graph/workflow.py` |
| State | `src/fsm_stackelberg/graph/state.py` |
| Stackelberg inspector | `src/fsm_stackelberg/agents/diagnosis_agent.py` |
| Progressive knowledge | `src/fsm_stackelberg/knowledge/progressive.py` |
| Feature plug-ins | `src/fsm_stackelberg/plugins/features.py` |
| Inspection probe helpers | `src/fsm_stackelberg/game/inspection.py` |
| Episode payoffs + commitment diagnostics | `src/fsm_stackelberg/game/payoff.py` |
| Run logs (`run_manifest.json` / `events.jsonl` / `artifacts/`) | `src/fsm_stackelberg/utils/run_log.py` |
| TSLP export / GT | `src/generator/` (`cli`, `serialize`, `ground_truth_solver`) |
| Inspectee repair | `src/fsm_stackelberg/agents/{data_engineer,model_expert,python_developer}.py` |
| CLI | `src/fsm_stackelberg/main.py` (`--log_prompts`, `--temperature`, `--probe_seed`, `--inject`) |
| Smoke instance | `dataset/prob_tslp_ecr_demand/instances/smoke_H4_Omega5` |
| TSLP knowledge | `src/fsm_stackelberg/knowledge/domains/tslp/` |
| Sibling repos | `../tslp-ecr-demand`, `../ecr-shared-data` |
