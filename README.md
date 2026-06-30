# fsm-stackelberg

**A state-machine Stackelberg game for diagnosis-repair in agentic natural-language optimization (ECR).**

`fsm-stackelberg` is the research codebase for a paper that re-architects the **error diagnosis & repair** stage of an LLM multi-agent optimization pipeline. The agent workflow (Data Engineer → Model Expert → Python Developer → Solver Executor) is orchestrated as a **LangGraph state machine (FSM)**; when the solver fails, repair is driven by a **Stackelberg leader-follower game** whose *commitment order* mirrors the pipeline's causal dependency order (data ⊳ model ⊳ code), rather than by a single-judge LLM accusation. The FSM is the stage-game apparatus on which the Stackelberg game is defined.

- **Status**: scaffold forked from `mako`; workflow imports verified; the Stackelberg redesign is **not yet implemented** (diagnosis still equals mako's `_adversarial_diagnosis`).
- **Start here**: [`HANDOVER.md`](./HANDOVER.md).

## Quick start

```bash
uv sync
cp /path/to/mako/.env .env          # API keys + Gurobi license env vars
uv run python -m fsm_stackelberg.main \
  --algorithm mako --dataset prob_ecr_shipper_consignee \
  --prob_name instances/high_demand_5-3_5 \
  --provider DeepSeek --model deepseek-chat \
  --diagnosis_mode adversarial --knowledge enable --max_retries 3
```

## Provenance

Forked from [`mako`](../mako) @ `fa2cbc8` (2026-06-30). Independent project — changes here must not be pushed back to `mako`.
