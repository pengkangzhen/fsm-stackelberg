# Spec: Evidence-informed Stackelberg diagnosis

> Status: **design locked for implementation** (2026-07-24).  
> Do **not** burn Exp-B external (Debate) API until this ships and smoke shows non-zero
> `verified_attribution_hit`.  
> Companion: [`HANDOVER.md`](./HANDOVER.md), manuscript `§Experiments`.

---

## 0. Goal (one sentence)

Leader uses failure evidence to form a **layer ranking** (not calibrated LLM
probabilities), **commits** probing order \(\omega\), then probes with
**executed refutation**; report **verified** attribution that is **≥**
strong single-judge accuracy while remaining auditable.

---

## 1. What stays / what changes

| Piece | Keep | Change |
|---|---|---|
| FSM nodes, blackboard, \(K\), inject plants | yes | — |
| Commit-once \(\sigma=(\omega,\nu)\); no mid-episode re-accusation drift | yes | — |
| \(\nu =\) executed refutation (re-solve) | yes | clarify verified-attr accounting |
| \(\omega\) from `Prior(status)` + causal rotate only | **no** (legacy) | evidence-informed rank → commit |
| `attribution_hit` = last-comply \(\hat a\) | secondary only | primary = **verified** |
| Adversarial / sequential baselines | yes | fair: same blackboard |

Legacy status-prior path remains behind a flag for A/B smoke only
(`--omega_source status_prior`), default = `evidence_rank`.

---

## 2. Pipeline (per failure episode)

```text
failure blackboard (status, stack, agent I/O history)
        │
        ▼
┌───────────────────────┐
│  A. Layer ranking     │  rank: List[agent]  (no P(·) from LLM)
│     rank_layers(...)  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  B. Align + commit ω  │  probe_order ∈ {causal, reverse, random}
│     commit_omega(...) │  writes inspection_policy (immutable)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  C. Probe loop ≤ K    │  next uncleared layer in ω
│     backward + re-solve│  ν: executed refutation
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  D. Verified attr     │  â_verified from refutation log
│     finalize payoffs  │  + first_probe_hit, SSR, tokens
└───────────────────────┘
```

**Forbidden:** each diagnosis round calling LLM to freely re-pick
`error_agent` outside committed \(\omega\) (“随便改指控”).

---

## 3. Module A — Layer ranking

### 3.1 API

```python
# src/fsm_stackelberg/game/ranking.py  (new)

Layer = Literal["data_engineer", "model_expert", "python_developer"]

@dataclass
class RankResult:
    rank: list[Layer]           # best-first
    method: str                 # "llm_rank" | "heuristic" | "hybrid"
    rationale: str              # short text for logs (not a probability)
    raw: dict                   # debug only; never treat as calibrated P

def rank_layers(state: dict, *, method: str = "llm_rank") -> RankResult:
    ...
```

### 3.2 Methods (implement in this order)

**A1. `llm_rank` (default for parity with adversarial)**  
- One LLM call, same evidence budget as adversarial (stack + histories).  
- Output schema: **ordered list only**  
  `{ "rank": ["model_expert", "python_developer", "data_engineer"],
     "rationale": "..." }`  
- **Reject** any `confidence` / `probability` fields in the contract; if the
  model emits them, ignore.  
- Must be a permutation of the three layers; repair invalid outputs by
  appending missing layers in causal order.

**A2. `heuristic` (cheap smoke / ablation)**  
- Score features: status prior, traceback mentions of files/agents,
  empty/missing stage outputs, injection-agnostic keywords.  
- Sort by score; ties broken by causal order.  
- No LLM tokens.

**A3. `hybrid` (optional later)**  
- Start from heuristic; allow LLM to **reorder** only (still no probs).

### 3.3 Fairness vs adversarial

| | Adversarial | Evidence-informed SB |
|---|---|---|
| Evidence | blackboard | **same** |
| LLM use at commit | accuse top-1 | output **full rank** |
| After first pick | re-accuse freely each round | **locked** to committed \(\omega\) |

Same information set at \(t=0\); difference is commitment + sequential
refutation, not “SB saw less.”

---

## 4. Module B — Align ranking with ablation mode + commit

### 4.1 Align

Given `rank = [r0, r1, r2]` and `probe_order`:

| `probe_order` | Committed \(\omega\) |
|---|---|
| `causal` | Prefer causal skeleton `DE→ME→PD`, **rotated** so `r0` is first, remainder follows causal cycle among leftover layers *(same rotate idea as today, but seed = rank[0] not status prior)* |
| `reverse` | Reverse of the **causal-aligned** \(\omega\) above (or reverse causal skeleton seeded by `r0` — pick one and freeze in code+tests) |
| `random` | Random permutation of three layers (`probe_seed`); **independent of rank** for pure order ablation *or* shuffle of `rank` with seed — **freeze: shuffle of `{DE,ME,PD}` with `probe_seed`, ignore rank** so Exp-A still isolates order |

**Exp-A freeze (recommended):**  
- `causal`: rank-informed rotate on causal cycle.  
- `reverse`: fixed `PD→ME→DE` (no rank; pure anti-causal).  
- `random`: seeded shuffle of three layers (no rank).  

Then Exp-A still tests commitment order; rank helps **causal** compete with
adversarial on accuracy, without giving reverse/random the same tip.

**Exp-B freeze:** causal + evidence rank vs adversarial / sequential /
Debate / Reflexion (one baselines table; stage internal before external).

### 4.2 Commit payload (`inspection_policy`)

```python
inspection_policy = {
    "omega": [...],                 # immutable list
    "nu": "executed_refutation",
    "probe_order": "causal|reverse|random",
    "omega_source": "evidence_rank",  # vs "status_prior"
    "rank": [...],                  # RankResult.rank at commit time
    "rank_method": "llm_rank",
    "rank_rationale": "...",
    "seed": omega[0],
    "probe_seed": ...,
    "committed_at_round": state["current_round"],
}
probe_queue = list(omega)
cleared_layers = []
```

Commit **once** per failure episode (`if not inspection_policy`).

---

## 5. Module C — Probe + executed refutation

Unchanged control flow from current `_stackelberg_diagnosis`:

1. Probe `next_unclear_layer(probe_queue, cleared_layers)`.  
2. Inspectee `*_backward_step` → comply / deflect.  
3. Re-solve = \(\nu\).  
4. On re-entry, clear last probed layer; continue.

### 5.1 Refutation log (new state field)

Append to `state["refutation_log"]` after each probe settlement:

```python
{
  "layer": "model_expert",
  "action": "comply" | "deflect",
  "error_resolved": bool,          # inspectee claim / flag
  "re_solve_strict_success": bool, # Practical Optimal after this probe path
  "symptom_cleared": bool,         # optional finer signal
  "overturn": bool,                # deflect/comply contradicted by re-solve
  "retry_index": int,
}
```

---

## 6. Module D — Verified attribution (metrics)

### 6.1 Definitions

| Symbol | Definition |
|---|---|
| `first_probe_hit` | \(\mathbf{1}\{\omega_1 = a^\star\}\) (unchanged) |
| `attribution_hit` | last-comply \(\hat a = a^\star\) (**secondary**, legacy) |
| `verified_attribution_hit` | \(\mathbf{1}\{\hat a_{\mathrm{ver}} = a^\star\}\) (**primary** for Exp-B+) |

**Proposed \(\hat a_{\mathrm{ver}}\) rule (freeze in tests):**

1. If some probe has `action=comply` and subsequent re-solve reaches
   **Practical Optimal**, and that layer is \(a^\star\) → verified hit
   (guilty complied and repair worked).  
2. Else if plant layer was probed and `plant_layer_complied` with
   symptom improvement uniquely attributable — same as (1).  
3. Else if only deflects / failed repairs → \(\hat a_{\mathrm{ver}} =\)
   `None` (no verified attribution), even if last-comply is set.  
4. Never set \(\hat a_{\mathrm{ver}}\) from LLM confidence alone.

Refine (1)–(3) in unit tests with fixtures; do not ship ambiguous last-comply
as verified.

### 6.2 Payoff / JSON

Extend `game/payoff.py` / `episode_payoff`:

```text
verified_attributed_layer
verified_attribution_hit
attribution_hit          # legacy last-comply
first_probe_hit / kill_hit
refutation_log_summary
```

Summarizers (`summarize_exp_i_pilot.py`, `summarize_exp_ii.py`) grow a
`verified_attribution_hit` column; Exp-B primary = verified.

---

## 7. Code touch list

| Path | Change |
|---|---|
| `src/fsm_stackelberg/game/ranking.py` | **new** `rank_layers` |
| `src/fsm_stackelberg/prompts/` | rank-only diagnosis prompt |
| `src/fsm_stackelberg/agents/diagnosis_agent.py` | commit path calls rank + align; keep status_prior behind flag |
| `src/fsm_stackelberg/graph/state.py` | `refutation_log`, optional `rank_result` |
| `src/fsm_stackelberg/game/payoff.py` | verified attribution |
| `src/fsm_stackelberg/main.py` | `--omega_source evidence_rank\|status_prior`, `--rank_method` |
| `tests/test_ranking.py`, `tests/test_verified_attr.py` | **new** |
| `scripts/summarize_exp_*.py` | verified column |
| CLI smoke | 1× causal evidence_rank vs 1× adversarial, same plant |

Do **not** modify Debate/Reflexion until redesigned Exp-A and Exp-B-internal pass.

---

## 8. CLI (proposed)

```bash
uv run python -m fsm_stackelberg.main \
  --diagnosis_mode stackelberg \
  --probe_order causal \
  --omega_source evidence_rank \
  --rank_method llm_rank \
  --inject me_force_zero_sea \
  --true_root_cause model_expert \
  --max_retries 3 \
  --knowledge progressive \
  --provider Qwen --model qwen3.7-plus \
  --dataset prob_tslp_ecr_demand --prob_name smoke_H4_Omega5
```

Adversarial baseline unchanged (fair compare).

---

## 9. Implementation order (no big Exp burn)

1. Unit tests for align rules + verified-attr fixtures (no API).  
2. `ranking.py` + prompt; wire commit in `_stackelberg_diagnosis`.  
3. Refutation log hooks on backward/re-solve path.  
4. Payoff + summarizer fields.  
5. **Smoke (budget ≤ ¥1):** 1 causal evidence_rank + 1 adversarial.  
   Gate: `verified_attribution_hit` or `first_probe_hit` informative;
   no crash; commit payload present.  
6. Only then cost-gated re-Exp-A / Exp-B-internal (`TOKEN_BUDGET`, `SKIP_EXISTING`).

---

## 10. Success / fail gates (before Exp-B external / Exp-C)

| Gate | Pass | Fail → |
|---|---|---|
| Smoke | commit+rank logged; episode finishes | fix code |
| Redesigned Exp-A | causal > reverse on first-probe / verified | revise align or Prop claim |
| Redesigned Exp-B internal | verified-attr(SB) ≥ verified-attr(adversarial) | revise rank prompt / verified rule; **do not** run Debate |
| Cost | tokens per verified attr not absurd vs adversarial | report cost trade |

---

## 11. Explicit non-goals

- Using LLM-reported probabilities as calibrated posteriors.  
- Mid-episode free re-accusation (destroys commitment).  
- Claiming SSR supremacy (sequential may repair blindly).  
- Expanding \(n\) or Exp-B external on status-prior \(\omega\).

---

## 12. HANDOVER pointer

Next agent: implement this spec starting at §9 step 1; update
`HANDOVER.md` TL;DR when smoke passes.
