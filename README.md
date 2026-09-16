# fsm-stackelberg

**A state-machine Stackelberg game for diagnosis-repair in agentic natural-language optimization**,
validated on **demand-uncertainty sea–land ECR** (two-stage SP / DEP from [`tslp-ecr-demand`](../tslp-ecr-demand)).

The agent workflow (Data Engineer → Model Expert → Python Developer → Solver Executor) is a
**LangGraph FSM**. On solver failure, repair is driven by a **Stackelberg inspection game** whose
commitment order follows data ⊳ model ⊳ code — not a single-judge LLM accusation.

- **Application model:** TSLP [D1]–[D2] (exogenous \(\xi\), random \(\eta^\omega\)). Mako-era
  shipper–consignee instances have been **removed**.
- **Start here:** [`HANDOVER.md`](./HANDOVER.md). Dataset notes: [`dataset/prob_tslp_ecr_demand/README.md`](./dataset/prob_tslp_ecr_demand/README.md).

## Quick start

```bash
uv sync
cp /path/to/mako/.env .env          # API keys (DEEPSEEK_API_KEY 主引擎; QWEN_*/ZHIPUAI_* 可选) + Gurobi license env vars

# (Re)export the smoke TSLP instance if needed
uv run python -m generator.cli \
  --output dataset/prob_tslp_ecr_demand/instances/smoke_H4_Omega5 \
  --T 4 --scenarios 5 --seed 42

uv run python -m fsm_stackelberg.main \
  --algorithm mako --dataset prob_tslp_ecr_demand \
  --prob_name smoke_H4_Omega5 \
  --provider DeepSeek --model deepseek-flash \
  --diagnosis_mode stackelberg --knowledge progressive --max_retries 3
```

Engine discipline: the validated main engine is official DeepSeek
`deepseek-flash` (thinking disabled, temperature 0). Switching engine =
starting a new campaign freeze; never pool cells across engines.

`--knowledge progressive` (default; `enable` is an alias) is **catalog-first
on-demand** injection — not dump-all, not RAG. Use `--knowledge disable` only
for ablations.

## Provenance

Forked from [`mako`](../mako) @ `fa2cbc8` (2026-06-30) for the agent workflow; application model
from [`tslp-ecr-demand`](../tslp-ecr-demand). Independent project — do not push changes back to mako.
