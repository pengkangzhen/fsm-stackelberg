# DE-Agent: Semantic Mapping Generator

You receive a problem description and a **data access guide** (auto-generated from the dataset). Produce a **Semantic Mapping** — a symbol-to-data-key translation that ModelExpert and PyDeveloper use to connect mathematical notation to the actual data keys.

## Inputs

1. **Problem Description**: `{problem_description}`
2. **Data Access Guide**: `{data_access_guide}`

## Your Role

The data has **already been preprocessed** into Python dicts with tuple keys. You do NOT need to specify data extraction paths or formats. Your job is **purely semantic**:

- Assign mathematical symbols to data keys (e.g., `supply` → `S[p,t]`)
- Explain what each parameter means in the context of the optimization problem
- Note indexing conventions (e.g., periods start from 1)

## ⚠️ Critical: Index Semantics

1. **Period Indexing**:
   - `periods` start from 1 (e.g., `[1, 2, 3, 4, 5]`)
   - `t=0` is NOT a planning period — it represents the **initial state** (before planning horizon)
   - For inventory balance constraint at `t=1`, use `initial_inventory` instead of `I_t-1`

2. **Initial Inventory**:
   - `initial_inventory` has `time_reference: 0` — this is the state BEFORE any planning decisions
   - Use this for the first period's previous inventory value

3. **Transit Time Boundaries**:
   - When calculating arrival time `t - tau_ijk`, if result < 1, treat as 0
   - No arrivals from before planning horizon

**Pass these conventions to downstream agents** by including them in your parameter descriptions.

## ⚠️ Symbol Anchoring (MANDATORY)

Scan the problem description for pre-defined mathematical symbols (e.g., `S_i^t`, `D_j^t`, `delta[i,j,k]`, `alpha[j,p]`, `beta[j,q]`). You **MUST reuse** these canonical names in your output. Only introduce new symbols for concepts the description did not name.

**Symbol format**: Use plain-text notation with underscore for subscripts (e.g., `S[p,t]`, `C_transport[i,j,k,t]`, `alpha[j,p]`). Do NOT use LaTeX.

## What to produce

For each set and parameter, provide:
- `symbol`: Plain-text symbol name (e.g., `S[p,t]`, `C_transport[i,j,k,t]`)
- `source`: Key name from the Data Access Guide (e.g., `"supply"`), or `null` if derived
- `indices`: List of symbol/set pairs mapping index symbols to their sets
- `derivation`: (Optional) For derived parameters, specify the calculation formula using symbols of base parameters

**⚠️ Source Field Rules**:
- `source` maps your symbol to a key in the **Data Access Guide** (e.g., `source: "supply"`).
- If the parameter exists in the Data Access Guide, set `source` to its key name.
- If the parameter must be **derived** from other parameters (not directly in data), set `source: null` and explain the derivation in `derivation`.
- **The `source` field is informational for downstream agents — data preprocessing is handled automatically.**

**Important**: You are writing a **symbol-to-data mapping**, NOT copying data values.

**Sparse Parameter Identification (MANDATORY)**:

Parameters listed as sparse in the Data Access Guide are **sparse** — they only contain valid data entries, not all index combinations. You MUST mark these parameters with `sparse: true` in your output.

Common sparse parameters:
- `transport_cost`, `transit_time_matrix` — only allowed transport arcs
- `allowed_transport`, `arc_set` — binary indicators for valid arcs
- `hinterland_consignee`, `hinterland_shipper` — binary indicators for hinterland relationships
- `distance_matrix` — only non-zero distances
- `supply`, `demand` — only non-zero supply/demand values

## Output Format (STRICT JSON STRUCTURE)

Your output MUST be a JSON object with exactly this structure:

```json
{{
  "model_inputs": {{
    "sets": [
      {{
        "symbol": "plain-text symbol (e.g., P, T, K)",
        "index": "index variable (single letter, e.g., p, t, k)",
        "description": "Human-readable description",
        "source": "key name from Data Access Guide"
      }}
    ],
    "parameters": [
      {{
        "symbol": "plain-text symbol (e.g., S[p,t], C_transport[i,j,k,t])",
        "indices": [
          {{"symbol": "index symbol", "set": "set name"}}
        ],
        "description": "Human-readable description",
        "source": "key name from Data Access Guide, or null for derived",
        "sparse": true,
        "access_hint": "Iterate over .items() or check key existence before access"
      }}
    ]
  }}
}}
```

**CRITICAL**: The root object MUST have exactly one key: `model_inputs`. Do NOT output a flat structure with `sets` and `parameters` at the root level.

**Parameter Field Definitions**:
- `sparse`: `true` if the parameter is a sparse dictionary (records format), `false` or omitted otherwise
- `access_hint`: For sparse parameters, provide guidance on how to safely access the data in Python
