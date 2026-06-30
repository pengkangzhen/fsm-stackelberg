---
name: ecr-mcnf-formulation
description: 多商品(MCNF)折叠箱 ECR 模型公式模板——四类型 θ 耦合、size 需求替代、fold/unfold handling，含 gurobipy 代码骨架
---

# Multi-Commodity Foldable ECR (MCNF) Formulation Template

This module provides the correct MILP formulation for the 4-type foldable empty
container repositioning problem. Use it as a **template** — adapt variable names
and data access to your specific instance.

## Problem Structure

- **Commodity types** G = {20S, 40S, 20F, 40F} (4 types: size × kind)
- **Sizes** Σ = {20, 40}; each type g has size σ(g)
- **Foldable types** G_F = {20F, 40F} (kind = F)
- **TEU footprint** θ: 20S=1, 40S=2, 20F=0.25, 40F=0.5 (a folded container ≈ 0.25× erected slot)
- **Node classes**: H∪L (storage: inventory+lease+local-demand), P (consignee: pure supply), Q (shipper: demand via y)
- **Demand is per-SIZE** (pooled): a 20ft demand can be met by 20S or 20F. Supply is per-TYPE.

## Decision Variables (all non-negative integers)

```
x[(i,j,k,g,t)]  — type-g containers on arc (i,j,k) in period t
w[(i,g,t)]       — ending inventory of type g at storage node i (H∪L only)
r[(i,g,t)]       — type-g leased at storage node i (H∪L only)
y[(i,g,t)]       — type-g used to satisfy local demand at node i (H∪L∪Q only)
```

**CRITICAL**: y variables exist ONLY at H∪L∪Q nodes. Consignees (P) have NO y variable and NO demand — they are pure supply nodes.

## Constraints

### 1. Flow conservation per type g, per node class, per period t

**Storage nodes (H∪L)** — carry inventory, lease, serve local demand:
```
w[i,g,t-1] + S[i,g,t] + r[i,g,t] + Σ_{(j,i,k)∈A⁻_i, t-τ≥1} x[j,i,k,g,t-τ]
  = y[i,g,t] + w[i,g,t] + Σ_{(i,j,k)∈A⁺_i} x[i,j,k,g,t]
```
For t=first_period: w[i,g,t-1] = I0[i,g] (typically 0).

**Consignees (P)** — pure supply, NO y, NO inventory:
```
S[p,g,t] + Σ_{(j,p,k)∈A⁻_p, t-τ≥1} x[j,p,k,g,t-τ]
  = Σ_{(p,j,k)∈A⁺_p} x[p,j,k,g,t]
```

**Shippers (Q)** — demand served by y, NO inventory:
```
Σ_{(j,q,k)∈A⁻_q, t-τ≥1} x[j,q,k,g,t-τ]
  = y[q,g,t] + Σ_{(q,j,k)∈A⁺_q} x[q,j,k,g,t]
```

### 2. Size-level demand substitution (eq:substit)
For each demand node i ∈ H∪L∪Q, size σ ∈ {20,40}, period t:
```
Σ_{g: σ(g)=σ} y[i,g,t] = D[i, σ, t]
```
Concretely: y[i,"20S",t] + y[i,"20F",t] = D[i,"20",t] and y[i,"40S",t] + y[i,"40F",t] = D[i,"40",t].

### 3. Shared TEU capacity (arc + node)
```
Σ_g θ[g] * x[i,j,k,g,t] ≤ F_max[k]          (per arc, per mode k)
Σ_g θ[g] * w[i,g,t]     ≤ U_max[i]           (per storage node)
```

### 4. Terminal inventory floor
```
w[i,g,T] ≥ I_bar[i,g]   for i ∈ H∪L, g ∈ G   (typically 0)
```

## Objective (per-TEU)

```
min  Σ_{t,(i,j,k),g} c_repos[k] * dist[i,j] * θ[g] * x[i,j,k,g,t]    (repos per TEU-km)
   + Σ_{t,i∈H∪L,g}   c_hold[i] * θ[g] * w[i,g,t]                      (holding per TEU)
   + Σ_{t,i∈H∪L,g}   c_rent[i] * θ[g] * r[i,g,t]                      (leasing per TEU)
   + c_fold   * Σ_{t,i∈N,g∈G_F} S[i,g,t]                               (fold: CONSTANT on supply)
   + c_unfold * Σ_{t,i∈H∪L∪Q,g∈G_F} y[i,g,t]                          (unfold: on service y)
```

**IMPORTANT**: c_fold × Σ S is a CONSTANT (supply is exogenous). c_unfold × Σ y is DECISION-RELEVANT.

## gurobipy Code Skeleton

Use this pattern to avoid common bugs (scope leaks, period-0 refs, wrong keys).

```python
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

def optimize(data):
    sets = data["sets"]
    H, L, P, Q = sets["seaports"], sets["dryports"], sets["consignees"], sets["shippers"]
    T, K, G = sets["periods"], sets["transport_modes"], sets["commodity_types"]
    Sigma = sets["demand_sizes"]
    HL = H + L  # storage nodes
    HLQ = HL + Q  # demand nodes (where y exists)
    first, last = min(T), max(T)

    theta = data["theta"]           # {"20S":1, "40S":2, "20F":0.25, "40F":0.5}
    size_of = data["size_of"]       # {"20S":"20", "40S":"40", "20F":"20", "40F":"40"}
    is_foldable = data["is_foldable"]  # {"20F":True, "40F":True, ...}
    GF = [g for g in G if is_foldable[g]]
    c_fold = data["c_fold"]
    c_unfold = data["c_unfold"]

    supply = data["supply"]  # {(node, period, commodity): value} — container counts
    demand = data["demand"]  # {(node, period, size): value} — container counts, pooled

    # Arcs from data (pre-derived, with distances)
    allowed = data["allowed_transport"]  # list of {from, to, mode}
    dist_map = {(a["from"], a["to"], a["mode"]): ... }  # build from data
    tau_map  = {(a["from"], a["to"], a["mode"]): ... }  # transit time

    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    for a in allowed:
        out_arcs[a["from"]].append((a["from"], a["to"], a["mode"]))
        in_arcs[a["to"]].append((a["from"], a["to"], a["mode"]))

    m = gp.Model("ecr_mcnf")
    m.Params.OutputFlag = 0

    # --- Variables (all integer, non-negative) ---
    # x: (i,j,k,g,t) — only for allowed arcs
    x = {}
    for (ii, jj, kk) in [(a["from"],a["to"],a["mode"]) for a in allowed]:
        for g in G:
            for t in T:
                x[(ii,jj,kk,g,t)] = m.addVar(vtype=GRB.INTEGER, lb=0,
                    name=f"x_{ii}_{jj}_{kk}_{g}_{t}")

    # w, r: storage nodes only (HL)
    w, r = {}, {}
    for i in HL:
        for g in G:
            for t in T:
                w[(i,g,t)] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"w_{i}_{g}_{t}")
                r[(i,g,t)] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"r_{i}_{g}_{t}")

    # y: demand nodes (HLQ) only — NOT consignees
    y_vars = {}
    for i in HLQ:
        for g in G:
            for t in T:
                y_vars[(i,g,t)] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{i}_{g}_{t}")

    m.update()

    # --- Helper: arrivals with transit time ---
    def arrivals(i, g, t):
        """Sum of x[j,i,k,g,t-tau] for all incoming arcs with valid period."""
        expr = gp.LinExpr()
        for (jj, ii, kk) in in_arcs[i]:
            tau = tau_map.get((jj,ii,kk), 0)
            arr_t = t - tau
            if arr_t in T:  # T is a set of valid periods; arr_t >= first
                expr += x[(jj, ii, kk, g, arr_t)]
        return expr

    def departures(i, g, t):
        """Sum of x[i,j,k,g,t] for all outgoing arcs."""
        return gp.quicksum(x[(i, jj, kk, g, t)] for (ii, jj, kk) in out_arcs[i])

    # --- Flow conservation ---
    for t in T:
        ts = str(t)
        # Storage nodes (H∪L)
        for i in HL:
            for g in G:
                prev_w = w[(i, g, t-1)] if t != first else 0  # I0=0
                s_val = supply.get((i, t, g), 0)  # per-type supply
                m.addConstr(
                    prev_w + s_val + r[(i,g,t)] + arrivals(i,g,t)
                    == y_vars.get((i,g,t), 0) + w[(i,g,t)] + departures(i,g,t),
                    name=f"bal_HL_{i}_{g}_{t}"
                )

        # Consignees (P) — pure supply, no y
        for p in P:
            for g in G:
                s_val = supply.get((p, t, g), 0)
                m.addConstr(
                    s_val + arrivals(p,g,t) == departures(p,g,t),
                    name=f"bal_P_{p}_{g}_{t}"
                )

        # Shippers (Q) — demand via y, no inventory
        for q in Q:
            for g in G:
                m.addConstr(
                    arrivals(q,g,t) == y_vars[(q,g,t)] + departures(q,g,t),
                    name=f"bal_Q_{q}_{g}_{t}"
                )

    # --- Size substitution: Σ_{g:σ(g)=σ} y[i,g,t] = D[i,σ,t] ---
    for i in HLQ:
        for sigma in Sigma:
            for t in T:
                d_val = demand.get((i, t, sigma), 0)
                m.addConstr(
                    gp.quicksum(y_vars[(i,g,t)] for g in G if size_of[g] == sigma) == d_val,
                    name=f"sub_{i}_{sigma}_{t}"
                )

    # --- Shared TEU capacity ---
    for t in T:
        for (ii, jj, kk) in [(a["from"],a["to"],a["mode"]) for a in allowed]:
            m.addConstr(
                gp.quicksum(theta[g] * x[(ii,jj,kk,g,t)] for g in G) <= data["F_max"][kk],
                name=f"cap_arc_{ii}_{jj}_{kk}_{t}"
            )
        for i in HL:
            m.addConstr(
                gp.quicksum(theta[g] * w[(i,g,t)] for g in G) <= data["U_max"][i],
                name=f"cap_node_{i}_{t}"
            )

    # --- Terminal floor ---
    for i in HL:
        for g in G:
            m.addConstr(w[(i,g,last)] >= 0, name=f"term_{i}_{g}")

    # --- Objective ---
    obj = gp.LinExpr()
    # Repositioning (per TEU)
    for (ii,jj,kk) in [(a["from"],a["to"],a["mode"]) for a in allowed]:
        c_unit = data["c_repos"][kk] * dist_map[(ii,jj,kk)]
        for g in G:
            for t in T:
                obj += c_unit * theta[g] * x[(ii,jj,kk,g,t)]
    # Holding + leasing (per TEU)
    for i in HL:
        for g in G:
            for t in T:
                obj += data["c_hold"][i] * theta[g] * w[(i,g,t)]
                obj += data["c_rent"][i] * theta[g] * r[(i,g,t)]
    # Fold (constant) + unfold (decision)
    fold_const = c_fold * sum(supply.get((i,t,g), 0) for t in T for i in sets["all_nodes"] for g in GF)
    for i in HLQ:
        for g in GF:
            for t in T:
                obj += c_unfold * y_vars[(i,g,t)]
    m.setObjective(obj + fold_const, GRB.MINIMIZE)

    m.optimize()
    # ... extract result ...
```

## Key Pitfalls to Avoid

1. **DO NOT create y at consignees (P)** — they are pure supply, no local demand.
2. **DO NOT divide 40ft values by 2** — supply/demand are in container counts, NOT TEU. θ is applied only in objective and capacity.
3. **DO NOT reference period-0 variables** — T = {1..5}; for t=first, use I0 (typically 0).
4. **DO NOT let outer-loop index leak** into quicksum — use distinct names (ii,jj,kk for arc tuple).
5. **c_fold is a CONSTANT** (on exogenous supply); c_unfold is DECISION-RELEVANT (on y).
6. **Demand keys are STRINGS** — demand.get((node, period, "20"), 0), not (node, period, 20).
7. **Initial inventory is ZERO** — do not read from data; set w[i,g,0] = 0 explicitly.
