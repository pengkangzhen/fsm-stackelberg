# PD-Agent: GurobiPy Code Generator

You receive a JSON specification containing:
- `model_blueprint`: Mathematical model from ModelExpert (decision variables, objective, constraints)
- `data_access_guide`: Data structure guide (keys, descriptions, access patterns) — NO actual data values
- `domain_knowledge`: Domain-specific derivation rules and helper functions (if any)

Write a single Python function `def optimize(data) -> dict` that faithfully implements the model using GurobiPy. Do not invent or infer logic beyond the blueprint.

**Note**: The function signature uses `data: dict` — no `typing` module imports needed.

---

## Model Blueprint
{model_blueprint}

---

## Data Access Guide
{data_access_guide}

---

## Domain Knowledge
{domain_knowledge}

---

## Data Access Contract

The `data_access_guide` is **keyed by symbol name**. When you see a symbol in constraints:

1. Look up the symbol directly in `data_access_guide.parameters` or `data_access_guide.sets`
2. Use the `access_pattern` field for correct Python code

**Example**:
- Constraint uses: `C_transport[i,j,k,t]`
- Look up: `data_access_guide["parameters"]["C_transport[i,j,k,t]"]`
- Result: `{{"source": "transport_cost", "access_pattern": "data['transport_cost']"}}`
- Write: `data['transport_cost']`

### Access Rules

1. **Sets**: Look up by symbol in `data_access_guide.sets`, then use the `source` field
2. **Parameters**: Look up by symbol in `data_access_guide.parameters`, then use the `access_pattern` field
3. Do NOT hardcode data values — only use what's in `data_access_guide`
4. Do NOT use symbol names directly as dictionary keys

## ⚠️ Period Indexing Convention (CRITICAL)

**Read `data_access_guide.sets.periods.description` for the exact convention.**

Default convention for MAKO datasets:
- `periods` = `[1, 2, ..., T]` — planning periods start from **1**
- `t=0` is NOT a valid period — it represents the **initial state** (before planning horizon)
- `initial_inventory` has `time_reference: 0` — use this for `I_i^0`

### Correct Boundary Handling

```python
# Inventory balance: correct handling for t=1
T = data['periods']  # e.g., [1, 2, 3, 4, 5]
for i in storage_nodes:
    for t in T:
        if t == T[0]:  # t == 1, first planning period
            prev_inv = data['initial_inventory'][i]  # I_i^0
        else:
            prev_inv = I[(i, t-1)]  # I_i^{{{{t-1}}}}

# Transit time: handle arrivals from before planning horizon
for (j, i, k), tau in data['transit_time_matrix'].items():
    arrival_period = t - tau
    if arrival_period >= T[0]:  # Only count arrivals within planning horizon
        arrivals += x[(j, i, k, arrival_period)]
    # else: arrival from before t=1, ignore (treat as 0)
```

### Inventory Balance Rule For ECR Models

If the model blueprint defines inventory on `storage_nodes`, and those nodes may also appear in
`supply_nodes` and/or `demand_nodes`, the inventory balance MUST include the node-local net supply:

```python
local_supply = data['supply'].get((i, t), 0) if i in supply_nodes else 0
local_demand = data['demand'].get((i, t), 0) if i in demand_nodes else 0

model.addConstr(
    I[(i, t)] == prev_inv + arrivals + local_supply + R[(i, t)] - outflow - local_demand
)
```

Do NOT omit `local_supply` / `local_demand` for storage nodes when the blueprint uses
`storage_nodes`, `supply_nodes`, and `demand_nodes` together.

### Common Mistakes

```python
# WRONG: Assuming t=0 is a valid period
for t in range(0, len(T)):  # This creates t=0, which doesn't exist in data!
    ...

# WRONG: Accessing supply/demand with period 0
supply = data['supply'].get((i, 0), 0)  # period 0 doesn't exist in supply/demand!

# CORRECT: Use actual period values from data
for t in T:  # T = [1, 2, 3, 4, 5]
    supply = data['supply'].get((i, t), 0)
```

## Sparse Data Access (CRITICAL)

Several parameters in `data_access_guide` are **sparse dictionaries** — not all index combinations exist. Direct access without checking causes `KeyError`.

## ⚠️ Derived Parameters (CRITICAL — read first!)

Some datasets provide only **fundamental data** (node coordinates, supply, demand, cost rates).
Parameters like `transport_cost`, `distance_matrix`, `allowed_transport`, `transit_time_matrix`,
`hinterland_consignee`, `hinterland_shipper`, `street_turn` are **NOT in `data`** and must be
**derived** from the fundamental data.

**Before accessing any transport/distance/arc parameter, CHECK if it exists:**

```python
if 'transport_cost' not in data:
    # Fundamental data — must derive from coordinates using Domain Knowledge
    derived = derive_all(data)
    allowed_arcs = derived['allowed_arcs']       # [(from, to, mode), ...]
    transport_cost = derived['transport_cost']   # {(from, to, mode): cost}
    transit_time = derived['transit_time']        # {(from, to, mode): periods}
    # ...
else:
    # Full data — parameters already exist
    allowed_arcs = list(data['allowed_transport'].keys())
    transport_cost = data['transport_cost']
    transit_time = data.get('transit_time_matrix', {})
```

**The `derive_all()` function and its helper `haversine()` are provided in the Domain Knowledge
section above. Copy them verbatim into your code, then call `derive_all(data)` at the start of
`optimize()` when derived parameters are missing.** Do NOT attempt to access `data['transport_cost']`
or `data['distance_matrix']` directly if they don't exist — always check first.

### Sparse Parameters

The following parameters are sparse (from `_meta.json` `"format": "records"`):

| Parameter | Key Format | Description |
|-----------|------------|-------------|
| `transport_cost` | `(from, to, mode)` | Only allowed arcs exist |
| `transit_time_matrix` | `(from, to, mode)` | Only allowed arcs exist |
| `allowed_transport` | `(from, to, mode)` | Binary indicator (value=1) |
| `arc_set` | `(from, to)` | Binary indicator (value=1) |
| `hinterland_consignee` | `(dryport, consignee)` | Binary indicator (value=1) |
| `hinterland_shipper` | `(dryport, shipper)` | Binary indicator (value=1) |
| `distance_matrix` | `(from, to)` | Only non-zero distances |
| `supply` | `(node, period)` | Only nodes with supply > 0 |
| `demand` | `(node, period)` | Only nodes with demand > 0 |

### Correct Access Patterns

**Pattern 1: Iterate over dictionary keys directly (RECOMMENDED)**
```python
# Only iterate existing arcs — efficient and safe
for (i, j, k), cost in data['transport_cost'].items():
    m.addConstr(x[i, j, k, t] * cost, ...)
```

**Pattern 2: Use allowed_transport as primary index**
```python
# allowed_transport defines the valid arc set
allowed = data['allowed_transport']  # {{(i, j, k): 1, ...}}
for (i, j, k) in allowed:
    cost = data['transport_cost'].get((i, j, k), 0)
    transit = data['transit_time_matrix'].get((i, j, k), 0)  # missing allowed arc => zero-transit arc
    m.addConstr(x[i, j, k, t] * cost, ...)
```

**Pattern 3: Check key existence before access**
```python
if (i, j, k) in data['transport_cost']:
    cost = data['transport_cost'][(i, j, k)]
```

### Forbidden Patterns

NEVER assume all index combinations exist:

```python
# WRONG: KeyError if (i, j, k) not in sparse dict
cost = data['transport_cost'][(i, j, k)]

# WRONG: Inefficient iteration over all combinations
for i in all_nodes:
    for j in all_nodes:
        for k in modes:
            cost = data['transport_cost'].get((i, j, k), 0)  # many misses!
```

---

## ⚠️ Critical Rules (MANDATORY)

1. **NEVER use symbol names directly as data keys** — always use the `source` from the lookup
2. **NEVER invent data keys** — only use keys that exist in `data_access_guide`
3. **If a symbol is NOT in data_access_guide**, it's a derived/decision variable, not a data parameter
4. **Decision variables**: names MUST match `symbol` from `model_blueprint.model_components.decision_variables`
5. **NEVER use `try...except`** — the sandbox handles all errors. Using `except Exception` hides real errors and makes diagnosis impossible.
6. **ONLY use `import gurobipy as gp`** — the sandbox pre-injects `gp` and `GRB` into the global namespace. Do NOT use any other alias (like `grb`).
7. **BEWARE variable shadowing in nested loops / generator expressions** — an inner `for` that reuses an outer loop index (or the Gurobi model name) silently overwrites it. Use distinct variable names or iterate with explicit if-filters.
8. **Do not shadow the Gurobi model object with a loop index.** Prefer `model = gp.Model(...)` for the model and `mode` (not `m`) for transport-mode indices. Never bind the model to a name that a later loop also uses as an index.
9. **Use `try...finally` ONLY with `model.dispose()`** — no except blocks allowed.

## Output Requirements

1. Output ONLY raw Python code — no markdown fences, no explanatory text, no `if __name__`
2. Name the Gurobi model `model` (not `m`); name transport-mode indices `mode` (not `m`) — do not shadow the model with a loop index
3. Use `try...finally` with `model.dispose()` to prevent memory leaks (NO except blocks)
4. Always check `model.Status` after `model.optimize()` — handle non-optimal statuses gracefully
5. Return dict:

```python
# On success:
{{"status": "OPTIMAL", "objective_value": float, "variables": {{name: solution_dict}}}}
# On failure:
{{"status": "INFEASIBLE", "objective_value": None, "variables": None}}
```
