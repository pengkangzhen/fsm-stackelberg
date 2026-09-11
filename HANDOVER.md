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
  (all six `pr/*` branches merged 2026-09-05 in dependency order 1→6; only
  conflict was binary `manuscript.pdf`, resolved to the pr/6 build; suite
  72 passed on merged main).
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
  - **Provider (2026-07-24+, 历史记录——测试阶段已被 2026-09-11
    GLM-5.3-Flash 决策取代):** cost-gated redesigned runs prefer
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
  - **Offline writing + branch consolidation (2026-09-05; still no API
    budget).** Manuscript (pr/6 line, now merged to main): new
    §"Result interpretation and provenance" (`sec:exp_reporting` —
    registered expectations as the reading grid, directional-only language
    under thin n, manifest/events as number authority, DE-contract
    provenance + non-poolability of pre/post-contract cells), expanded
    status-prior preliminary narratives (Exp-A/B), new §"Failure-mode
    analysis" (`sec:exp_failure_modes` — traceback-as-oracle, stale repair
    evidence, residual regeneration variance; grounded in per-cell run
    records). Methodology decision: **keep pre-registered expectations**
    (respond to thin-n by expanding n after the gate; label exploratory
    analyses; do NOT switch to exploratory "pick-the-winner" framing).
    Stacked pr/1–pr/6 merged into main in dependency order and pushed;
    compile clean at 17 pages; deferred writing: hybrid main-table
    confirmatory prose (waits for expanded n) and Exp-C real table.
  - **Snapshot freeze/resume shipped (2026-09-05, branch
    `pr/7-snapshot-resume`; offline, no API spend).** Cost-controlled
    grids: `--snapshot_dir` freezes the failure blackboard at the
    solver→diagnosis boundary (policy-invariant prefix); `--resume_from`
    restores it and runs **only** the diagnosis–repair loop live
    (entry node = `diagnosis_agent`). Per-cell cost drops to treatment-path
    tokens only, and cells sharing a snapshot remove forward-pipeline noise
    as a confound (the Setup's "same blackboard" fairness promise becomes
    exact). Provenance: `snapshot_manifest.json` (provider/model/temp, git
    commit, plant, failure surface, forward token baseline) + run_manifest
    gains `resumed_from` / `snapshot_dir` / `snapshot_forward_tokens`.
    Tests: `tests/test_snapshot_resume.py` (10). **Also fixed latent CLI
    break:** `--probe_seed` / `--inject` were duplicated in `main.py` since
    `9a26e17` (07-23, pr/1) → argparse ArgumentError on every invocation;
    deduplicated. The 07-24~27 runs nevertheless succeeded through
    `python -m fsm_stackelberg.main`, so that tree carried an uncommitted
    local fix — commit fixes immediately; uncommitted state evaporates.
    **Design note (decided 2026-09-05):** expanded-n / re-Exp-A grids after
    recharge run in snapshot mode (freeze once per plant×seed, resume all
    arms); the no-plant E2E gate stays fully live and cannot be replayed.
    ZCode subagents may rehearse the resume path (harness validation only —
    never as paper data: workspace holds the plants/a*, no audit trail).
  - **Subagent rehearsal DONE (2026-09-10; harness-only, not paper data).**
    4 subagents replayed real prompt templates on a planted force-zero
    failure (report: `results/subagent_rehearsal/REPORT.md`, gitignored).
    Behavior matched design (rank tips ME; ME-backward complies with
    minimal strip; ME-forward requests 4 knowledge modules round 1).
    Machinery caught **3 schema-drift classes** real LLMs emit (param
    indices as dict; DecisionVariable missing indices/shape + name/type
    drift; direction "minimize") → before-validators added
    (`pr/8-schema-coercions`, stacked on pr/7) + 9 regressions; suite
    **91 passed**. Rehearsal re-confirmed: subagent transport has no raw
    byte recovery (HTML-escaped final message only) — unusable as data
    source, fine as free fixture generator.
  - **测试阶段引擎定为 GLM-5.3-Flash（2026-09-11，用户拍板）。** 依据
    9-10/11 的 provider 调研：智谱 `glm-5.3-flash` 全天平价（无峰谷）
    ¥0.8/¥2.8 每 M tokens、缓存命中 ¥0.23、Arena 盲测代码榜前十、混合
    成本 ~¥1.3/M——"能力足够 + 最便宜"的最优解（5 折已于 9-9/10 到期，
    现为原价）。`llm_config` 已支持 ZhipuAI（env `ZHIPUAI_BASE_URL` /
    `ZHIPUAI_API_KEY`；智谱走 function_calling 结构化输出，`_fc_providers`
    已含；`glm-5.3-flash` 已加入 PROVIDER_MODELS）。**`.env` 需先加
    ZHIPUAI_API_KEY**（base URL 默认
    `https://open.bigmodel.cn/api/paas/v4`）。备选引擎 DeepSeek V4.1
    Flash（`deepseek-flash`，空闲 ¥1/¥4、缓存 ¥0.02；高峰 = 工作日
    9–12/14–18，高峰翻倍——用它须排夜间/周末）。**纪律：引擎切换 =
    新战役冻结**，GLM 格子不与旧 DashScope deepseek-v4-flash n=3 混池；
    对比块内 provider/model 固定并报告。调研时点快照（2026-09-11，价格
    波动快，使用前以官方页为准）：GLM-5.3-Flash ¥0.8/¥2.8 平价 ·
    deepseek-flash ¥1/¥4 空闲（高峰×2）· 混元 3.0 ¥1.2/¥4（SWE-bench
    Verified 74.4%，首开送 1M tokens，3 月曾涨价 463%）· Kimi K3
    ¥20/¥100 · 豆包 Pro ¥6–15/¥26–30。
  - **GLM-5.3-Flash 实弹测试 DONE（2026-09-11 晚）：gate FAIL，引擎结论
    变更为"不适用于全管线"。** 用户提供了智谱 key，连续三次 no-plant
    E2E gate（`results/mako/ZhipuAI_glm-5.3-flash/.../glm53flash_noplant_gate/`，
    三次运行已自动归档）：
    1. 尝试 1：DE forward 3×APITimeout @300s 默认超时 → 提高超时修复；
    2. 尝试 2：ME forward 返回 **None**（GLM 未发起 function call，
       langchain 解析器对无工具调用返回 None）→ 节点裸崩 → 修复见下；
    3. 尝试 3（最远，2h）：DE✓（722s，契约一次过）→ ME 三轮渐进知识✓
       （期间 None-retry 实战命中并救回一次）→ PD✓ → solver
       **OPTIMAL-but-gap 17.6%**（obj 1,144,989 < z* 1,389,581，比最优
       还便宜 = 约束缺失，Spurious Optimal 模式）→ 诊断环：ME comply
       被执行反驳推翻 → 探 PD → **PD backward 3×APITimeout @900s 崩溃**。
    **判定**：(i) 无种植故障前向建模 gap 17.6%（DashScope
    deepseek-v4-flash 历史 smoke 直接命中 z*）；(ii) always-on thinking
    （API 1210 不允许关闭，仅 low/high/max）使单次调用 12–18 分钟，PD
    backward 大 prompt 在 900s 内无法完成非流式返回；(iii) 成本本身极低
    （~¥0.5–1/次），但墙钟与可靠性不可行。**建议主引擎回到
    DashScope / deepseek-flash（V4.1）**，GLM-5.3-Flash 降级为"已评估、
    不适用于全管线"（跨模型块若仍想用 GLM 需先解决流式/超时工程）。
    **本次留下的 harness 加固（引擎无关，全部保留）**：
    `llm_config`：ZhipuAI GLM → `thinking_effort=low` + 900s 超时；
    新增 `utils/llm_invoke.invoke_structured`：结构化输出返回 None 时
    附加"必须调用工具"提醒重试一次，仍 None 则抛清晰错误——8 个调用点
    （DE/ME/PD forward+backward、rank、diagnosis）全部接入。
    快照机制实弹验证**延后**（GLM 跑不到 backward 修复环；离线测试
    10/10 已覆盖机械正确性，live 验证随 deepseek gate 一并做）。
  - **官方 DeepSeek deepseek-flash：GATE PASS + 快照实弹验证 DONE
    （2026-09-11 晚，全部 ~¥0.15）。** 用户提供新官方 key（旧 key 401
    已换；余额仅 ¥0.49 也够了——V4.1 Flash + 缓存命中价极便宜）。
    1. **No-plant E2E gate PASS**：41.5s / 80,232 tok / retries=0 /
       **obj 1,389,581.0815 vs z\* …0841，gap 0.0%，一次命中 Practical
       Optimal**（`results/mako/DeepSeek_deepseek-flash/.../
       ds41_noplant_gate/`）。单次调用 7–8s（thinking 禁用生效，
       `deepseek-flash` 无 "v4" token 的匹配坑已修）。
    2. **快照机制 live 验证 PASS**：冻结一次（`--inject
       me_force_zero_sea --snapshot_dir results/snapshots/
       me_force_zero_sea_s1`，surface=INFEASIBLE，forward 71,096 tok），
       三臂恢复（causal/reverse/random seed=1）全部从同一黑板分支出：
       `resumed_from` + `snapshot_forward_tokens` 落盘于各 manifest；
       loop tokens = total − snapshot 基线（93–96k/臂）；顺序对照如期
       （causal ω=[ME,PD,DE] tip ME；reverse 固定 [PD,ME,DE]；random
       seed=1 恰为 [ME,PD,DE]）。**同板公平承诺现已是逐字节事实。**
    3. 单抽样观察（非论文数据）：三臂 K=3 内均未达 verified/Practical
       Optimal（obj 1.34M/1.41M/1.41M），与 DashScope 早期抽样同类，
       需要的是复格与调参，非机械问题。
    **引擎定案：官方 DeepSeek / deepseek-flash（V4.1 Flash）**。余额
    剩 ¥0.34——扩 n 网格前需充值（按今日实测：一臂 ~¥0.05，整个
    n=5×3 序网格估 <¥2）。
  - **扩样战役 DONE（2026-09-11 深夜；用户充值 ¥20 后）**：快照模式
    29 运行（5 冻结 + 15 Exp-A 臂 + 10 Exp-B 臂）共 **¥1.70 / ~20 分钟**
    （余额 ¥18.37 余）。**预注册 Exp-A 预期在 n=5 下全部满足**：first-
    probe causal 5/5 > random 3/5 > reverse 0/5；verified 0.20 > 0 /
    0.20 = 0.20 / 0；SSR 2/5 最高；成本同量级。**关键机制证据**：random
    唯一 verified 命中恰为洗牌成 causal 同序（[ME,PD,DE]）的那格，
    其 3 个 first-probe 命中全部是 ME 首位前缀——随机臂收益完全来自
    偶然对齐因果序。Exp-B internal：SB = adversarial（verified 平
    0.20）但 adversarial 花 **2.03× token**，first-probe 0.60 vs 1.00，
    sequential 全零。总结（含逐格细节与保留意见）：
    `results/exp_ab_ds41_snapshot/summary.md`。**下一步**：稿件 measured
    表换本战役数字（声明快照模式与 loop-token 口径）→ Phase-4 多 plant
    网格（同法冻结/恢复）→ Exp-B external（Debate/Reflexion）→ Exp-C。
  - **n=20 扩样 DONE（2026-09-11 深夜，用户要求加样）**：序贯披露
    （n=5 窥视 → 预期不变 → 扩至 20 可用 seed；冻结期契约失败 3/23
    = s7/s12/s14，以 s21–s23 补齐）。**first-probe 子句强确认**：
    causal 20/20 vs random 10/20 vs reverse 0/20，配对 McNemar
    p<0.0001 / p=0.002；random 的 10 个 first-probe 命中与"洗牌后
    ME 首位"**逐一精确对应**。verified：causal 0.45 / random 0.45 /
    reverse 0.25——vs reverse 点估计严格高但 p=0.125 未显著（n≈40–50
    可分辨）；vs random 平局。**机制修正**：random 的 9 个 verified
    命中中 3 个来自未对齐洗牌——对齐完全决定 first-probe、提高
    verified 到达率，但多轮搜索是第二条路径。Exp-B：SB 0.45 ≥ adv
    0.40（p=1.0）≫ seq 0（p=0.0039），adv 等值 verified 花 1.93×
    token。总花费 ¥7.22（n=20 全战役），余额 ¥12.85。总结（含检验
    与保留意见）：`results/exp_ab_ds41_snapshot/summary.md`。
    **推荐下一步**：Phase-4 多 plant 网格（优先于 n=40 加密单 plant）。
  - **Phase-4 多 plant 归因网格 DONE（2026-09-12 凌晨，快照模式，¥7.42）**：
    `de_swap_demand_supply_source`（a\*=DE）与 `pd_comment_out_balance`
    （a\*=PD）各 n=10 可用 seed × {causal, reverse, random}，同引擎同法
    （官方 deepseek-flash，冻结+`--resume_from` 分臂，random 带
    `--probe_seed=s`）。**结论 = 顺序效应的条件复制**：
    1. **对齐完全决定 first-probe，3 plant × 40 个 random 臂零例外**
       （命中 ⟺ ω₁=a\*）。
    2. "causal 严格>reverse" 依赖 rank 对 a\* 的可见性：ME 植物
       （force-zero 嗅探）20/20 vs 0/20 强确认；PD 植物 rank 正确 tip
       （causal 9/10）但 reverse 骨架对 code 层**构造性对齐**（10/10，
       子句不成立；causal vs random 7:0 **p=0.0156 显著**）；DE 语义
       植物对 rank 完全不可见 → causal=reverse=0/10，**子句被证伪**，
       仅 random 偶然对齐命中（4/10）。
    3. **verified 归因在所有植物均无显著序效应**（K=3 多轮执行反驳 +
       seed 级可修复性主导；me_force 的 p=0.125 应解读为对齐优势的
       边缘残留）。主张应修订为"承诺对齐在排名可见故障层时获胜"。
    4. **新前向方差类（DE 植物独有）**：12/22 冻结尝试中换源参数未进
       入生成模型 → obj 逐位=z\*（Practical Optimal，无失败黑板）→
       计为前向方差换 seed；55% 的"植物未表达率"本身是语义数据故障
       的实测性质。pdbal 修复后 2/17 DE 契约拒收（常规类）。
    - **战役间 harness 修复（零数据窗口，已披露）**：pdbal 旧植物的
      行级正则匹配不到真实 PD 生成的多行 `addConstr(...)`（name= 在
      续行）→ s1-s5 全灭（¥0.06）；改为语句级括号配对匹配
      （`injection/plants.py::_constr_statement_spans`）+ 回归测试后
      重跑。deswap/me_force 植物函数未动。
    - 1 臂（deswap s16 random 首跑）LengthFinishReasonError → 同板
      重跑成功（n=20 战役"s15 补尾"先例）。
    - 编排 `scripts/run_p4_grid.py`（可续跑/按 seed 配对/牌价上界守卫
      ≈实际×2-3）；汇总 `scripts/summarize_p4_grid.py`（对 n=20 战役
      复算逐项校验一致）。总结：`results/p4_grid/summary.md`（必读，
      含三植物表、配对 McNemar、对齐审计、保留意见）。
  - **稿件 measured 表已刷新（同夜）**：`tab:exp_ablation` 换 n=20
    快照模式数字（20/10/0、9/9/5、loop-token 口径）；新增
    `tab:exp_multiplant`（Phase-4 两植物 3×n 表）+ 条件复制 prose；
    `tab:exp_baselines` 换 n=20（SB 0.45 ≥ adv 0.40 ≫ seq 0，adv 花
    1.9× token）；Setup 注入协议改为三层植物网格 + 快照同板公平性；
    provenance 段声明 hardened 实现 + no-plant gate + 快照/loop-token
    口径 + 序贯披露；失效模式节新增"Rank-invisible semantic faults"
    家族。编译 0 错误。Exp-C 与 Debate 仍未跑。
  - **实例族扩展 DONE（2026-09-12 凌晨，me_force × 2 新实例，快照模式，¥5.14）**：
    `fam_H4_Omega10`（T=4，|Ω|=10，seed7，z*=1,433,725.34）与
    `fam_H6_Omega5`（T=6，|Ω|=5，seed13，z*=2,633,849.94），各 n=10 可用
    seed × 3 序。**预注册 Exp-A 预期全部满足，且 smoke n=20 未分辨的
    verified 子句在族上分辨到显著**：first-probe causal 三实例全满贯
    （40/40）、reverse 全零；池化 causal vs reverse 20:0（p=2e-06）、
    vs random 11:0（p=0.001）；**verified 池化 causal 0.60 vs reverse
    0.20（8:0，p=0.0078）、vs random 0.25（8:1，p=0.039）**——smoke 的
    causal=random 平局（9/20=9/20）不复现。causal 臂最省 token、random
    臂最贵（未对齐多轮搜索）。对齐审计继续零例外。总结（含逐实例表与
    保留意见）：`results/instance_family/summary.md`；机器可读 =
    `results/p4_grid/summary.json` 的 `instances` 节。
    - **两个新 schema 漂移类（战役间修复，均先于或后于数据窗口）**：
      (i) 嵌套层级漂移（objective_function/constraints 被提到顶层）——
      gate 期发现，before-validator 提升修复，先于战役任何格落地；
      (ii) 重复约束名（非负性按变量族拆成同名多条）——战役中 4 次冻结
      死亡按前向方差记账（H4_Ω10 3 次、H6_Ω5 1 次），战役后落
      `ModelComponents.dedup_constraint_names` 去重强制器 + 回归测试；
      套件 **94 passed**。
    - 编排/汇总脚本已实例化：`run_p4_grid.py --prob_name/--state_root/
      --prefix`（快照目录 `{prob}__{plant}_s{seed}` 实例限定），
      `summarize_p4_grid.py` 增 `instances` 块。
  - **Exp-B external DONE（2026-09-12 凌晨，Debate/Reflexion × me_force
    n=20，快照同板，用户明示解冻）**：实现两模式（debate = 三透镜辩手
    单轮投票 `{suspected_agent, argument}` + 机械多数票 + status prior
    平票决胜，不采信自报置信度；reflexion = 单判官初始归因 + 回合记忆
    批判修订），与既有修复路径共享；测试 106/106。**上线前 subagent
    彩排 PASS**（用户要求的零成本门：4 子代理在真实黑板回放真实
    prompt，3/3 辩手不同透镜独立投中 a\*、反思从伪造 PD 初始归因正确
    修订到 ME、零 schema 漂移；`results/subagent_rehearsal/
    debate_reflexion/`，gitignored）。**结果**：debate/reflexion 的
    first-probe 与 adversarial **逐格全同**（12/20，0:0 不一致对）——
    多叫 LLM 不改善命名；SB 对两者 first-probe 8:0（p=0.0078）。
    verified：reflexion 点估最高 11/20 (0.55) vs SB 9/20 (0.45)
    （0:2，ns，诚实保留）；debate 最低 7/20。成本决定性：debate 3.7×、
    reflexion 2.8× SB token。**修订触发子句未触发**（无外部协议在相近
    成本下支配 SB）。1 格退化（reflexion_s2，PD backward
    LengthFinishReason 基建类）按补尾先例重跑成功。总结：
    `results/exp_b_external/summary.md`；编排
    `scripts/run_ebext_grid.py`。稿件 tab:exp_baselines 已换五方法全表
    + 外部 prose，编译 0 错误。
  - **Next concrete work (ordered) — Debate unfrozen and DONE:**
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
    4. ~~No-plant E2E gate after recharge~~ **done 2026-09-11**（官方
       deepseek-flash，gap 0.0%，Practical Optimal 一次命中）。
    5. ~~Phase-4 多 plant 网格 + 稿件 measured 表刷新~~ **done
       2026-09-12**（见上；总结 `results/p4_grid/summary.md`）。
    6. ~~实例族扩展~~ **done 2026-09-12**（见上；verified 子句已显著，
       n≈40 加密的必要性下降）。可选后续：(a) DE/PD 植物的实例族复制；
       (b) 更大网络（节点/弧维度）实例；(c) me_force n→40 加密若仍想
       压缩 smoke 上的置信区间。
    7. 已注册的设计修订（未实施）：DE 语义故障的 provenance 证据
       （rank 不可见类，见稿件失效模式节），战役间窗口才可改。
    8. Exp-B external (Debate/Reflexion) + Exp-C — 仍冻结，用户明示后
       才开。

### Next-agent prompt (copy-paste)

Paste the block below into a new chat to continue.

```markdown
# 任务：Exp-C（诊断动力学）设计 + 离线先行，或 DE/PD 实例族复制

## 现状（2026-09-12 凌晨，全部已提交推送至 main HEAD）

- 引擎：官方 DeepSeek / deepseek-flash（余额以 GET
  api.deepseek.com/user/balance 实查为准）。
- **实验主体已齐**：Exp-A n=20（smoke）+ 实例族 n=10×2（verified 子句
  池化显著 p=0.0078/0.039）+ Phase-4 三植物条件复制 + Exp-B 五方法
  同板 n=20 全表（SB/adv/seq/debate/reflexion：SB first-probe 8:0
  支配全部对手；reflexion verified 0.55 点估高于 SB 0.45 但 ns 且
  2.8× 成本）。各总结：results/exp_ab_ds41_snapshot/、
  results/instance_family/、results/p4_grid/、results/exp_b_external/。
- 稿件 measured 表全数落地（tab:exp_ablation / tab:exp_multiplant /
  tab:exp_instances / tab:exp_baselines），编译 0 错误；结论与摘要仍是
  占位符。schema 漂移防御 4 类已落地（106 tests）。
- 编排工具链：run_p4_grid.py（实例化）/ run_ebext_grid.py（外部位）/
  summarize_p4_grid.py + exp_b_external/summary.json。

## 本任务（按优先级）

1. **Exp-C 设计 + 离线先行**：attr vs K 曲线与 deflect-overturn 率可先
   从既有 200+ 臂的 events.jsonl/refutation_log 离线重分析；需要补格
   （如 K 扫描）再实跑，先 2 seed 试运行。
2. 或 **DE/PD 植物实例族复制**（p4 结论的实例交叉）。

## 硬约束

- Debate/Reflexion 已解冻（用户 2026-09-12 明示）；如需新协议变体仍
  先 subagent 彩排再花钱。
- 主指标 verified_attribution_hit；战役内设计不可变；序贯扩样披露。
- 引擎/运行方式不与旧战役混池；manifest 溯源。
- 预算：先查余额，战役硬帽与用户确认；每格先 2 seed 试运行。

## 验收

- [ ] 所选方案产出 summary（含检验）入 results/ 对应目录
- [ ] 稿件对应表/prose 刷新（Exp-C 现为 mock 占位）
- [ ] HANDOVER 更新 + 未超预算 + 全部提交推送
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
- **`debate` / `reflexion` modes implemented (2026-09-12)** in
  `diagnosis_agent.py` (`_debate_diagnosis`, `_reflexion_diagnosis`);
  Exp-III-era note "not yet implemented" is obsolete.

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

### Phase 4 — Attribution grid — RUN DONE (2026-09-12; offline half 2026-08-29)

Primary metric already fixed in manuscript.

- **Analysis:** `scripts/analyze_attribution_grid.py` auto-discovers runs
  (reads `run_manifest.json` or legacy `experiment_result.json`), and emits
  true-layer × attributed-layer confusion, surface × true matrices, and
  per-method rates. Unlike mako (which *guessed* origin from the surface),
  the true layer comes from the injected plant. Debugged on 85 historical
  runs (`results/attribution_grid/`). The Phase-4 campaign itself is
  orchestrated/aggregated by `scripts/run_p4_grid.py` +
  `scripts/summarize_p4_grid.py` (snapshot mode, paired McNemar,
  random-alignment audit; results: `results/p4_grid/summary.md`).
- **Plants:** layer grid closed — `de_swap_demand_supply_source` (DE→ME
  boundary; wrong catalog mapping under the value-free contract; statement-
  level matching for multi-line `addConstr` fixed 2026-09-12) and
  `pd_comment_out_balance` (PD→solver boundary; relaxed DEP, correct ME).
  `fault_injector` is now a layer-aware factory (`make_fault_injector_node`);
  global one-shot `fault_injected` flag unchanged. Tests:
  `tests/test_injection.py`.
- **Result (n=10/plant × 3 orders, snapshot mode):** conditional
  replication — alignment (ω₁=a\*) fully determines first-probe (zero
  exceptions over 3 plants × 40 random arms); "causal > reverse" holds
  only where the hybrid rank can see a\* (ME smell yes; PD surface yes but
  reverse is structurally aligned for code-layer faults; DE semantic swap
  invisible → falsified there); verified attribution shows no significant
  order effect in any plant. New forward-variance class: DE-plant silent
  draws (12/22 freezes solved to exact z\*, no failure blackboard).

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
| Next experiment priority | **(a) me_force verified 加密 n≈40/臂 或 (b) 实例族扩展** | Phase-4 网格已完成（条件复制结论）；verified 0.45 vs 0.25 仍未分辨（p=0.125）；两案均快照模式先试 2 seed；DE 语义 provenance 修订属独立战役 |
| 测试阶段 LLM 引擎 | **官方 DeepSeek / deepseek-flash（V4.1）——已验证** | 2026-09-11 gate PASS（41s / gap 0.0%）+ 快照三臂 live 验证 PASS；n=20 + Phase-4 + 实例族三战役实战 ~¥19.8；余额 ¥9.55（2026-09-12，用户已再充值）；GLM 已判不适用 |
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
| Snapshot freeze/resume (cost grids) | `src/fsm_stackelberg/graph/snapshot.py`, `tests/test_snapshot_resume.py`; CLI `--snapshot_dir` / `--resume_from` |
| Evidence ranking + ω align | `src/fsm_stackelberg/game/ranking.py` + `prompts/templates/diagnosis_agent/rank.md` |
| Fault plants (Exp-I; DE/ME/PD layers) | `src/fsm_stackelberg/injection/` + `agents/fault_injector.py` (layer-aware boundaries) |
| Phase-4 attribution-grid analyzer | `scripts/analyze_attribution_grid.py` |
| Phase-4 campaign orchestrator + summarizer | `scripts/run_p4_grid.py`（实例化：--prob_name/--state_root/--prefix）, `scripts/summarize_p4_grid.py`, `scripts/run_ebext_grid.py`；总结 `results/p4_grid/`、`results/instance_family/`、`results/exp_b_external/` |
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
