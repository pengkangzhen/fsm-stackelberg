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
  (local tip includes `bf0c849` Exp-I injection/pilot tooling; push if needed).
- **Current state (2026-07-27).**
  - Phase 0 skipped; Phase 1 Stackelberg PoC **done**; Phase 2 analysis-mode
    payoffs **done** (`src/fsm_stackelberg/game/payoff.py`), with Exp-I
    diagnostics: `first_probe_hit` / `kill_hit` / `committed_omega`.
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
  - **Smoke E2E on TSLP done** (2026-07-22): `--knowledge progressive`,
    historically Qwen/`qwen3.7-plus`; **current cheap default =
    DashScope/`deepseek-v4-flash`**. Final **Practical Optimal** vs GT
    \(z^\star\approx 1.389581\times 10^{6}\) (solver `OPTIMAL` alone is not
    enough — see manuscript `tab:outcome_taxonomy`).
  - **Fault injection + Exp-I pilot infra done** (`--inject`,
    `src/fsm_stackelberg/injection/`, ME→PD one-shot `fault_injector` node).
    Default strong plant: `me_force_zero_sea` (force \(y_{\mathrm{in}}=y_{\mathrm{out}}=0\);
    keep min objective + balances; clear loaded knowledge bodies).
  - **Probe-order ablation fixed:** `random` / `reverse` no longer
    Prior-rotated onto the status seed (only `causal` rotates). Optional
    `--probe_seed` for reproducible random shuffles.
  - **Exp-I kill-criteria pilot v3 PASSED** (2026-07-23):
    causal **2/3≈0.67** > random **1/3≈0.33** → proceed.
    Summary: `results/exp_i_pilot_v3/summary.md`.
  - **Full Exp-I n=5 DONE** (2026-07-23, cost-safe staged run):
    `stackelberg × {causal, reverse, random}`, plant `me_force_zero_sea`,
    \(a^\*=\)`model_expert`, \(K=3\), Qwen/`qwen3.7-plus`. Primary
    `first_probe_hit`: causal **4/5=0.80** > random **3/5=0.60** >
    reverse **0/5=0.00** → kill verdict **`pass`**. Secondary:
    `attribution_hit` 0/5, 1/5, 0/5; SSR@$K$ 1/5, 1/5, 0/5. Reused pilot
    causal/random r1–3; new API ≈1.24M tokens (~¥3.8 est.); hard cap ¥10.
    Summary: `results/exp_i_full/summary.md`. Runners:
    `scripts/launch_exp_i_full.sh`, `run_exp_i_full.sh`.
    **Do not** treat n=5 as definitive Prop.~1 proof; optional expand n
    later if budget allows. Caveat: CRASH surfaces still Prior-rotate
    causal onto PD first (causal_r1 miss); last-comply attribution remains
    weak.
  - Earlier pilots (historical): v1/v2 `kill_or_revise`.
  - **Exp-II overnight** (2026-07-23, status=`ok`):
    adversarial + sequential, plant `me_force_zero_sea`, n=5, K=3,
    Qwen/`qwen3.7-plus`. Primary `attribution_hit`.
    - adversarial: attr 0/5 (0.00), first_probe 2/5 (0.40), SSR 0/5 (0.00), mean_tok=234844.2
    - sequential: attr 0/5 (0.00), first_probe 0/4 (0.00), SSR 2/5 (0.40), mean_tok=85609.6
    - stackelberg: attr 0/5 (0.00), first_probe 4/5 (0.80), SSR 1/5 (0.20), mean_tok=138219.0
    Spend ≈ ¥3.94 (1294730 tokens). Summary: `results/exp_ii/summary.md`.
    Runners: `scripts/run_exp_ii_overnight.sh`, `run_exp_ii.sh`, `summarize_exp_ii.py`.
  - Manuscript: `§Experiments` reorganized into **three blocks**
    (Exp-A ablation / Exp-B unified baselines / Exp-C diagnostics);
    mock main tables + status-prior preliminary tables; Exp-B merges former
    Exp-II+III narratively (execution may still stage internal before external).
  - **Claim reset (2026-07-24) — redesign before more API spend.**
    Qualitative review: status-prior Stackelberg does **not** naturally beat
    a strong LLM single-judge on naming accuracy (symmetric evidence at
    failure time). Footing = **verifiable attribution + adversarial
    robustness + cost/audit**, with accuracy **≥ / not behind** single-judge
    as a hard requirement → requires **Evidence-informed commitment**:
    Leader forms a **layer ranking** from stack+history (**no** calibrated
    LLM probabilities), **commits** \(\omega\), then probe + **executed
    refutation**; attribution = verified, not last-comply. Do **not** start
    Exp-B external (Debate/Reflexion) until redesign ships and smoke shows
    non-zero verified-attr.
    **Spec locked:** [`docs/SPEC_EVIDENCE_INFORMED_SB.md`](./docs/SPEC_EVIDENCE_INFORMED_SB.md).
  - **Evidence-informed SB — CODE DONE; ¥1 smoke structural PASS (2026-07-24).**
    - Impl: `game/ranking.py` (`rank_layers` / `align_omega`), `game/verified.py`,
      rank prompt `prompts/templates/diagnosis_agent/rank.md`, wired in
      `_stackelberg_diagnosis`; CLI `--omega_source evidence_rank|status_prior`
      (default `evidence_rank`), `--rank_method llm_rank|heuristic|hybrid`.
      Exp-A freeze: causal uses rank tip; reverse/random ignore rank.
      Primary fields: `verified_attributed_layer` / `verified_attribution_hit`
      + `refutation_log` in `episode_payoff`. Tests:
      `tests/test_ranking.py`, `tests/test_verified_attr.py` (green).
    - Smoke (Qwen/`qwen3.7-plus`, `me_force_zero_sea`, K=3):
      - SB: `results/mako/Qwen_qwen3.7-plus/prob_tslp_ecr_demand_k3/smoke_H4_Omega5/evidence_sb_causal_er/`
        — exit 0; `omega`+`rank`+`omega_source` logged; `verified_attribution_hit=False`
        (rank tipped PD on CRASH; K exhausted; ~166k tok).
      - Adv: `.../evidence_adv/` — exit 0; SSR=1 but â_ver=PD ≠ a\*=ME
        (`verified_attribution_hit=False`; ~111k tok).
      - Notes: `results/evidence_informed_smoke/SUMMARY.md`.
    - Gate: **structural pass**; **verified-attr miss** on that draw.
  - **Provider (2026-07-24+):** cost-gated redesigned runs prefer
    **DashScope / `deepseek-v4-flash`** (百炼 OpenAI-compat;
    `QWEN_*` or `DASHSCOPE_API_KEY`; `enable_thinking` default **off** —
    set `FSM_ENABLE_THINKING=1` for Bailian chat-style thinking).
    Qwen Exp-A partial aborted (~¥3.1) → `results/exp_a_evidence_qwen_partial/`.
  - **Redesigned Exp-A DONE (2026-07-24, DashScope/`deepseek-v4-flash`).**
    n=3 × {causal, reverse, random}, plant `me_force_zero_sea`, K=3,
    `--omega_source evidence_rank`. Spend **1.21M tok ≈ ¥3.69**; 9/9 CELL_OK;
    no budget stop.
    - first_probe: causal **1/3≈0.33** > reverse **0/3** > random **0/3**
      → gate verdict **`pass`** (thin; only `ea_causal_r3` tipped ME).
    - **verified_attribution_hit: 0/9**; **SSR@$K$: 0/9** — mechanism primary
      metric failed; do **not** treat as paper-ready success.
    - Failure modes (inspect trajectories):
      1. CRASH surfaces → llm_rank often tips **PD** (traceback-as-oracle);
         K burned on PD/DE overturns before ME.
      2. Even correct tip (`ea_causal_r3`: ME first, OPTIMAL-but-gap≈74%):
         ME backward saw **pre-injection** history (hid `Injected_Force_Zero`);
         LLM “fixed” unrelated `sea_outbound_floor` → overturn; never Practical
         Optimal → no verified.
    - Summary: `results/exp_a_evidence/summary.md` (+ `summary.json`).
      Results root: `results/mako/DashScope_deepseek-v4-flash/prob_tslp_ecr_demand_k3/smoke_H4_Omega5/exp_i_pilot_ea_*`.
      Runners: `scripts/launch_exp_a_evidence.sh`, `run_exp_a_evidence.sh`,
      `summarize_exp_a_evidence.py`. Exp-B runners ready but **not run**:
      `run_exp_b_internal_evidence.sh`, `summarize_exp_b_internal_evidence.py`.
  - **Verified-attr repair CODE DONE (2026-07-27); live smoke UNLOCKED verified.**
    Root cause of `ea_causal_r3` gap: ME `backward_step` used
    `get_last_forward_output` (pre-plant) + LLM rewritten wrong floors.
    Fixes shipped:
      1. `rank.md`: traceback ≠ root; force-zero / \(y_{\mathrm{in}},y_{\mathrm{out}}=0\) → tip ME.
      2. `ranking._heuristic_rank`: force-zero smell → +ME / −PD; hybrid shortcuts
         to heuristic when smell present. Tests: `tests/test_ranking.py`.
      3. ME backward: prefer **current** `model_expert_output`; on comply with
         force-zero present → **minimal strip** of plant constraint (keep rest);
         strip regex must **not** match nonnegativity `(>= y_in 0)` (v1 smoke
         false positive). Prompt `model_expert/backward_step.md` updated.
         Tests: `tests/test_me_force_zero_repair.py` (green).
      4. `fault_injector`: record planted ME into `output_history` as forward.
    **¥1 re-smoke (DashScope/`deepseek-v4-flash`, 2026-07-27):**
      - Adv `verified_fix_adv`: **`verified_attribution_hit=True`**,
        â_ver=`model_expert`, SSR=1, obj≈\(z^\star\), ~159k tok.
      - SB v2 `verified_fix_sb_causal_er_v2`: tip ME + strip OK, but PD
        regen KeyError→193% gap; verified=False (~114k tok).
      - Summary: `results/evidence_informed_smoke_v2/SUMMARY.md`.
  - **Re-Exp-A hybrid DONE (2026-07-27).** Old llm_rank cells backed up as
    `exp_i_pilot_ea_*_llmrank_20260724`. New grid n=3,
    `--rank_method hybrid`, DashScope/`deepseek-v4-flash`, ~1.05M tok ≈ ¥3.18.
    Summary: `results/exp_a_evidence_hybrid/summary.md`.
    - first_probe: causal **3/3=1.0** > reverse **0/3** > random **0/3**
    - verified: causal **1/3≈0.33** > reverse **0** > random **0**
    - SSR: causal **2/3≈0.67** > reverse/random **0**
    - Gate **`pass`**. Standout: `ea_causal_r3` verified=True, SSR=True, K=1.
  - **Exp-B internal hybrid DONE (2026-07-27).** n=3 × {adv, seq},
    ~0.74M tok ≈ ¥2.24. Summary:
    `results/exp_b_internal_evidence_hybrid/summary.md`.
    - verified: SB **0.33** ≥ adversarial **0.00** ≥ sequential **0.00**
      → gate **`pass`** (thin; SB win is 1/3 vs 0/3).
    - Caveats: `eb_seq_r1` tokens=0 (KeyError `solver_executor`); adv grid
      0/3 verified despite earlier smoke adv hit (high variance / PD regen).
  - **Offline consolidation (2026-08-29; LLM budget empty — no API runs).**
    Fixed the clear-policy test breakage shipped in `b465b62`
    (`ranking.has_force_zero_sea_smell` now public + normalizes dict/pydantic
    ME output via `_text_blob`); full suite **65 passed**. Documented the
    provenance of `results/pd_regen_audit/de_contract_{smoke,e2e}_20260730.json`
    in that SUMMARY: the "e2e" objective matches v4-replay leg A to 12
    significant digits → **v4-seed replay, not the no-plant gate**; do not
    cite it as a no-plant result. All API-calling work (no-plant E2E smoke,
    expand-n, Debate, Exp-C) is **blocked until tokens are recharged**.
  - **Next concrete work (ordered) — Debate still frozen:**
    1. ~~Evidence-informed SB code + smokes + re-Exp-A/B internal~~ **done**.
    2. ~~Audit PD-regen after ME strip~~ **done, but root cause remains
       partially localized**. Healthy-seed forward PD is stable (gap 0%).
       The v4 replay's 55%/62%/930% gaps are retained as evidence that the
       legacy DE-derived guide is unsafe, **not** as proof that DE caused the
       original v4/v5 gap: that replay manually used
       `_build_data_access_guide(DE)`, whereas the real PD-forward path uses
       `auto_preprocess()`'s deterministic `state["data_access_guide"]`.
       Also, aliases (`V`, `A`) and v4's `p` index are not inherently invalid.
       Correction: `results/pd_regen_audit/SUMMARY.md` § "v4 trajectory
       replay — correction".
    3. ~~Deterministic DE schema validation~~ **implemented 2026-07-30**
       (mechanism, not experiment-specific prompt tips):
       `data/data_contract.py` derives exact catalog IDs and index domains
       from `sample.json`; DE selects IDs while retaining mathematical
       aliases; validator rejects unknown sources/domain mismatches and gives
       one bounded retry; derived values use `derived_parameters`. PD backward
       now uses the same deterministic guide as PD forward. Tests:
       `tests/test_data_contract.py`.
    4. **PRIORITY (blocked on LLM budget since 2026-08-29):** DE-only smoke
       is structurally done (`de_contract_smoke_20260730.json`); still owed a
       no-plant E2E smoke confirming the new contract reaches Practical
       Optimal before any re-Exp-A or expanded \(n\). **Run first after
       recharge.**
    5. Exp-B external (Debate/Reflexion) + Exp-C — freeze until SB verified
       rate is stabler / PD-regen stability bounded by DE validation.
    6. ~~Manuscript tables: replace mock with hybrid Exp-A/B numbers~~ **done**
       (`els-cas-templates/manuscript.tex`: `tab:exp_ablation`,
       `tab:exp_baselines`; thin-$n$ caveat; Exp-C still mock; Debate not run).

### Next-agent prompt (copy-paste)

Paste the block below into a new chat to continue.

```markdown
# 任务：充值后先跑 no-plant E2E smoke（预算空时勿调任何 API）

DE 数据契约已落地（单测绿）；clear-policy 测试断裂已修（65 passed）。
先读：

1. `HANDOVER.md` TL;DR（DE contract + 2026-08-29 离线收尾）
2. `results/pd_regen_audit/SUMMARY.md`（含 `de_contract_*` 定性：e2e=v4-seed replay，非 no-plant）
3. `results/exp_a_evidence_hybrid/summary.md`
4. `results/exp_b_internal_evidence_hybrid/summary.md`

## 硬约束

- **不要**跑 Debate/Reflexion，除非用户明确要求。
- **预算空（2026-08-29）**：未确认充值前勿调任何 LLM API。
- Provider：**DashScope / deepseek-v4-flash**；控制预算。
- 主指标：`verified_attribution_hit`。

## 立刻做（先问用户是否已充值）

A. 已充值 → no-plant E2E smoke（唯一 gate；PASS 才能扩 n / 重跑 Exp-A）
B. 未充值 → 离线：Results prose / Phase 4 网格脚本移植

## 验收

- [ ] 有明确下一步决策 + HANDOVER 更新
- [ ] 未盲开 Debate / 未在预算空时调 API
```



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

### Phase 3 / manuscript Exp-I–III — PILOT DONE; FULL GRID OPEN

Manuscript protocol (`§Experiments`):

| Block | Content |
|---|---|
| **Setup** | TSLP data, fixed model, budget \(K\), upstream injection / downstream symptom, fairness (same blackboard + repair path) |
| **Methods** | (A) stackelberg × {causal, reverse, random}; (B) adversarial, sequential; (C) Debate + Reflexion |
| **Metrics** | Primary: verified attribution + first-probe (ablation); last-comply secondary. SSR@$K$, tokens, audit; Exp-C: vs \(K\), deflect overturn |
| **Exp-A** | Commitment-order ablation — status-prior n=5 DONE; **evidence_rank n=3 DONE** (first-probe pass, verified 0) |
| **Exp-B** | Unified baselines: internal (adversarial, sequential) + external (Debate, Reflexion) — **internal not re-run under evidence protocol yet** |
| **Exp-C** | Inspection diagnostics (attr vs \(K\), overturn) |

**Immediate next task:** Manuscript Exp-A/B tables **updated** to measured
hybrid $n{=}3$ (`tab:exp_ablation`, `tab:exp_baselines`). Next: stabilize SB
verified (PD regen) and/or expand $n$, then Results prose; Debate frozen.
Summaries: `results/exp_a_evidence_hybrid/summary.md`,
`results/exp_b_internal_evidence_hybrid/summary.md`.

Runners (gitignored results under `results/`):
- **Evidence Exp-A/B:** `scripts/launch_exp_a_evidence.sh`,
  `run_exp_a_evidence.sh`, `summarize_exp_a_evidence.py`,
  `run_exp_b_internal_evidence.sh`, `summarize_exp_b_internal_evidence.py`
- Legacy status-prior: `scripts/launch_exp_i_full.sh` / `run_exp_i_full.sh`,
  `launch_exp_i_v3.sh` / `run_exp_i_pilot_v3.sh`, `summarize_exp_i_pilot.py`
- CLI: `--inject me_force_zero_sea --true_root_cause model_expert`
  `--probe_order {causal,reverse,random} [--probe_seed N]`
  `--omega_source evidence_rank|status_prior --rank_method llm_rank|heuristic`

Debate design note (agreed): debaters output `{suspected_agent, argument}`
only; aggregate by majority vote; tie-break via status prior — **do not**
trust LLM-reported confidence.

### Phase 4 — Attribution grid — OFFLINE HALF DONE (2026-08-29)

Primary metric already fixed in manuscript. Ported from mako's
`analyze_2d_attribution.py`:

- **Analysis:** `scripts/analyze_attribution_grid.py` auto-discovers runs
  (reads `run_manifest.json` or legacy `experiment_result.json`), and emits
  true-layer × attributed-layer confusion, surface × true matrices, and
  per-method rates. Unlike mako (which *guessed* origin from the surface),
  the true layer comes from the injected plant. Debugged on 85 historical
  runs (`results/attribution_grid/`).
- **Plants:** layer grid closed — `de_swap_demand_supply_source` (DE→ME
  boundary; wrong catalog mapping under the value-free contract) and
  `pd_comment_out_balance` (PD→solver boundary; relaxed DEP, correct ME).
  `fault_injector` is now a layer-aware factory (`make_fault_injector_node`);
  global one-shot `fault_injected` flag unchanged. Tests:
  `tests/test_injection.py`.
- **Remaining:** the grid *run* itself (plant × method × reps) — blocked on
  LLM budget; run only after the no-plant E2E gate passes.

### Phase 5 — Proposition — PARTIAL

Prop. 1 is in the manuscript; formal bound optional; **empirical validation =
Exp-I**. n=5 commitment-order table landed (`pass`: causal > random >
reverse on `first_probe_hit`). Still treat as **directional evidence**, not
a definitive prop validation, until larger \(n\) / multi-plant replication.

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
   Healthy-seed PD-forward is stable, but the v4 replay did not faithfully
   reproduce the real guide path and cannot assign causality to DE. A
   deterministic DE catalog/validator now removes one contract ambiguity;
   original post-strip variance remains a residual threat until an E2E smoke.
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
| Next experiment priority | **DE contract E2E smoke** | Catalog/validator implemented; first verify one no-plant run is Practical Optimal. Do not claim the corrected v4 replay localized the original gap. |
| Exp-I pilot subsection in tex | keep for now | v3 passed; OK to shrink/delete after main Exp-I table |

---

## 8. Environment checklist

- [x] `uv sync` (venv at `.venv/`)
- [x] `.env` with API keys (`python-dotenv`; **DashScope/`deepseek-v4-flash`
      via `QWEN_*` works**; Qwen/`qwen3.7-plus` also OK; official DeepSeek
      / MiMo keys historically flaky — prefer 百炼)
- [x] Gurobi license active via `GRB_LICENSE_FILE` (academic WLS; verified in
      smoke / pilot runs)
- [x] GitHub remote in use (`origin` → `pengkangzhen/fsm-stackelberg`)

---

## 9. File map (cheat sheet)

| Purpose | Path |
|---|---|
| Manuscript | `els-cas-templates/manuscript.tex` (§stackelberg, §experiments, `tab:outcome_taxonomy`) |
| Evidence-informed SB **spec** | `docs/SPEC_EVIDENCE_INFORMED_SB.md` |
| FSM | `src/fsm_stackelberg/graph/workflow.py` |
| State | `src/fsm_stackelberg/graph/state.py` |
| Stackelberg inspector | `src/fsm_stackelberg/agents/diagnosis_agent.py` (`build_probe_order`) |
| Progressive knowledge | `src/fsm_stackelberg/knowledge/progressive.py` |
| Feature plug-ins | `src/fsm_stackelberg/plugins/features.py` |
| Deterministic DE data catalog / validator | `src/fsm_stackelberg/data/data_contract.py`, `tests/test_data_contract.py` |
| Inspection probe helpers | `src/fsm_stackelberg/game/inspection.py` |
| Episode payoffs + kill / verified | `src/fsm_stackelberg/game/payoff.py`, `game/verified.py` |
| Evidence ranking + ω align | `src/fsm_stackelberg/game/ranking.py` + `prompts/templates/diagnosis_agent/rank.md` |
| Fault plants (Exp-I; DE/ME/PD layers) | `src/fsm_stackelberg/injection/` + `agents/fault_injector.py` (layer-aware boundaries) |
| Phase-4 attribution-grid analyzer | `scripts/analyze_attribution_grid.py` |
| Exp-I full / pilot runners (legacy status-prior) | `scripts/launch_exp_i_full.sh`, `run_exp_i_full.sh`, `launch_exp_i_v3.sh`, `run_exp_i_pilot_v3.sh`, `summarize_exp_i_pilot.py` |
| Pilot summary (local, gitignored) | `results/exp_i_pilot_v3/summary.md` |
| Evidence-informed ¥1 smoke | `results/evidence_informed_smoke/SUMMARY.md` |
| PD-regen audit + deterministic regressions | `results/pd_regen_audit/SUMMARY.md` (v3–v5 audit, healthy-seed regression GREEN, v4-seed replay REPRODUCED → DE defect table) |
| PD-regen stability regression (healthy seed) | `scripts/regress_pd_regen_stability.py` |
| PD-regen replay regression (v4 seed) | `scripts/regress_pd_replay_v4.py` |
| Redesigned Exp-A (evidence_rank) | `results/exp_a_evidence/summary.md` + runners `scripts/launch_exp_a_evidence.sh`, `run_exp_a_evidence.sh`, `summarize_exp_a_evidence.py` |
| Redesigned Exp-B internal (ready, not run) | `scripts/run_exp_b_internal_evidence.sh`, `summarize_exp_b_internal_evidence.py` |
| LLM providers | `src/fsm_stackelberg/utils/llm_config.py` (DashScope + `deepseek-v4-flash`) |
| TSLP export / GT | `src/generator/` (`cli`, `serialize`, `ground_truth_solver`) |
| Inspectee repair | `src/fsm_stackelberg/agents/{data_engineer,model_expert,python_developer}.py` |
| CLI | `src/fsm_stackelberg/main.py` (`--inject`, `--probe_order`, `--probe_seed`, `--omega_source`, `--rank_method`, `--true_root_cause`) |
| Smoke instance | `dataset/prob_tslp_ecr_demand/instances/smoke_H4_Omega5` |
| TSLP knowledge | `src/fsm_stackelberg/knowledge/domains/tslp/` |
| Sibling repos | `../tslp-ecr-demand`, `../ecr-shared-data` |
