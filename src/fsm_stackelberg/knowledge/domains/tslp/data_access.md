---
name: tslp-data-access
description: sample.json layout — sets, topology, first_stage, supply_demand, inventory keys mapped to model symbols
---

# TSLP data access (`sample.json`)

The instance JSON is already structured for the DEP. **Do not** invent laden→empty
conversion or street-turn logic. Read tables as lists of records and build
dicts keyed by the fields below.

## Top-level blocks

| Block | Role |
|-------|------|
| `sets` | Index sets: `hubs`, `spokes`, `dry_ports`, `seas`, `nodes`, `periods`, `modes`, `scenarios` |
| `topology` | `lambda_hl` (sea–dry hinterland links), `arcs` (allowed inland moves) |
| `first_stage` | Stage-1 sea parameters for hubs |
| `supply_demand` | \(\xi\), \(E\), \(\eta\), \(\pi\), and nominal \(\bar\eta\) |
| `inventory` | \(I_0\), \(U\), holding/lease/spill costs |
| `Q_bar` | Reserved / unused in the smoke export (may be empty) |
| `_meta` | Instance tags (formulation, \(T\), \(n\) scenarios); not decision data |

## `topology.arcs` (inland)

Each arc record:

| Field | Symbol / use |
|-------|----------------|
| `from`, `to` | Arc endpoints \(i,j\) |
| `mode` | Mode \(\kappa\) (`road` / `rail`) |
| `transit_time` | \(\tau_{ij\kappa}\) (periods) |
| `capacity` | Arc cap \(F_{ij\kappa}\) |
| `unit_cost` | Land unit cost \(c_{ij\kappa}^{\mathrm{repo}}\) (repositioning) |
| `distance_km` | Diagnostic only; do not invent extra costs from it |

Use only listed arcs; do not add shipper/consignee legs.

## `first_stage` (hubs / [D1])

| Field | Symbol |
|-------|--------|
| `vessel_calls` | \(V_{i,t}\) — records `{hub, period, calls}` |
| `B_in`, `B_out` | \(B_i^{\mathrm{in}}, B_i^{\mathrm{out}}\) — `{hub, value}` |
| `D_ext_eff` | \(D_{i,t}^{\mathrm{ext,eff}}\) — `{hub, period, value}` |
| `c_sea_in`, `c_sea_out` | Scalars \(c^{\mathrm{sea,in}}, c^{\mathrm{sea,out}}\) |

## `supply_demand`

| Field | Symbol | Record keys |
|-------|--------|-------------|
| `xi` | \(\xi_{n,t}\) | `{node, period, value}` |
| `E` | \(E_{n,t}\) (spoke top-up; 0 elsewhere) | `{node, period, value}` |
| `eta` | \(\eta_{n,t}^\omega\) | `{scenario, node, period, value}` |
| `eta_bar` | Nominal mean (reference only) | `{node, period, value}` |
| `scenario_probability` | \(\pi_\omega\) | `{scenario, probability}` |

The DEP **must** use `eta` and `scenario_probability`, not only `eta_bar`.

## `inventory`

| Field | Symbol | Record keys |
|-------|--------|-------------|
| `I0` | \(I_{n,0}\) | `{node, value}` |
| `U_cap` | \(U_n\) | `{node, value}` |
| `c_hold` | \(c_n^{\mathrm{hold}}\) | `{node, value}` |
| `c_lease` | \(c_n^{\mathrm{lease}}\) | `{node, value}` |
| `c_spill` | Scalar \(c^{\mathrm{spill}}\) (large) | number |

## Modeling checklist tied to data

1. Stage-1 variables only on `sets.hubs` × `sets.periods`.
2. Stage-2 inland flows only on `topology.arcs` × scenarios × periods (respect \(\tau\)).
3. Inventory / lease / spill on `sets.nodes` (or the relevant subset per balance).
4. No shipper/consignee indices appear in this dataset.
