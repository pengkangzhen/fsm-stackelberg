# Dataset: `prob_tslp_ecr_demand`

Application problem for **fsm-stackelberg**: demand-uncertainty sea–land empty-container
repositioning as a **two-stage stochastic LP** ([D1]–[D2] DEP), ported from
[`tslp-ecr-demand`](../../tslp-ecr-demand) + [`ecr-shared-data`](../../ecr-shared-data).

This replaces the mako-era deterministic shipper–consignee MCNF instances
(`prob_ecr_shipper_consignee`), which have been removed from this repo.

## Layout

```
prob_tslp_ecr_demand/
├── description.txt              # Slim NL task (business + stages + objective categories)
├── base/                        # problem-level shared config (generation defaults)
│   ├── costs_base.json          #   base cost scalars + mode unit rates
│   └── candidate_nodes.json     #   candidate hub/spoke/dry_port pool
├── config/
│   └── scenarios.yaml           # batch generation recipes
└── instances/
    └── smoke_H4_Omega5/         # default smoke window (H=T=4, |Ω|=5, seed=42)
        ├── nodes.csv            # node_name, type(hub/spoke/dry_port)
        ├── arcs.csv             # from, to, mode, distance_km, transit_time, capacity, unit_cost
        ├── lambda_hl.csv        # sea↔dryport linkage
        ├── vessel_calls.csv     # first-stage班轮挂靠 V
        ├── supply_demand.csv    # deterministic xi / eta_bar / E (outer-joined)
        ├── scenarios.csv        # stochastic demand realisation η(ω)
        ├── scenario_probability.csv
        ├── inventory.csv        # I0, U_cap, c_hold, c_lease (per node)
        ├── transport_modes.csv  # mode + derived base unit cost
        ├── first_stage.json     # B_in/B_out/D_ext_eff, c_sea_in/out, c_spill
        ├── metadata.json        # authoritative sets + _meta
        ├── sample.json          # consolidated fallback (structurally identical)
        └── optimal.json         # expert DEP objective (PuLP/CBC)
```

The loader (`_load_multi_source`) reads the per-instance CSV/JSON files and
reconstructs a dict identical to `sample.json`; the latter is kept only as a
fallback.  Two-stage-specific files (vs. the single-stage MAKO layout) are
`scenarios.csv`, `scenario_probability.csv`, `vessel_calls.csv`, and
`first_stage.json`.

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
