# Dataset: `prob_tslp_ecr_demand`

Application problem for **fsm-stackelberg**: demand-uncertainty sea–land empty-container
repositioning as a **two-stage stochastic LP** ([D1]–[D2] DEP), ported from
[`tslp-ecr-demand`](../../tslp-ecr-demand) + [`ecr-shared-data`](../../ecr-shared-data).

This replaces the mako-era deterministic shipper–consignee MCNF instances
(`prob_ecr_shipper_consignee`), which have been removed from this repo.

## Layout

```
prob_tslp_ecr_demand/
├── description.txt          # Slim NL task (business + stages + objective categories)
└── instances/
    └── smoke_H4_Omega5/     # default smoke window (H=T=4, |Ω|=5, seed=42)
        ├── sample.json
        ├── optimal.json     # expert DEP objective (PuLP/CBC)
        └── metadata.json
```

Formulation details and JSON field maps live in progressive knowledge modules
under `src/fsm_stackelberg/knowledge/domains/tslp/` (e.g. `tslp-data-access`,
`tslp-stage1-sea`, `tslp-stage2-inland`) — not in `description.txt`.

## Regenerate / export

```bash
uv sync
uv run python -m generator.cli \
  --output dataset/prob_tslp_ecr_demand/instances/smoke_H4_Omega5 \
  --T 4 --scenarios 5 --seed 42
```

Ground truth uses the same DEP as `tslp-ecr-demand` (`solve_dep`). Agentic runs
should match `optimal.json` → `objective` under the strict-success gap rule.

## Run fsm-stackelberg

```bash
uv run python -m fsm_stackelberg.main \
  --algorithm mako \
  --dataset prob_tslp_ecr_demand \
  --prob_name smoke_H4_Omega5 \
  --provider DeepSeek --model deepseek-chat \
  --diagnosis_mode stackelberg \
  --knowledge progressive --max_retries 3
```

Knowledge is **progressive** (catalog → ModelExpert requests → full text), not
a one-shot dump and not RAG. Pass `--knowledge disable` only for ablations.
