Diagnose the failure using a two-dimensional framework: identify the observable solver outcome, then trace the error back to its origin agent.

## Observable Status

**Gurobi Status**: `{gurobi_status}`

(You are in the diagnosis loop, so the solver either failed or produced a suboptimal result.)

| gurobi_status | Status | Description |
|---------------|--------|-------------|
| `""` | CRASH | Code never reached the solver |
| `OPTIMAL` | WRONG_OBJ | Solver found a solution but objective gap exceeds threshold |
| `INFEASIBLE` | INFEASIBLE | Model has no feasible solution |
| `UNBOUNDED` | UNBOUNDED | Model is unbounded |
| `ERROR` | SOLVER_ERROR | Gurobi internal error |

---

## Agent Outputs

### Python Code (PythonDeveloper)
```python
{python_code}
```

### Model Definition (ModelExpert)
```json
{model_expert_output}
```

### Data Access Guide (DataEngineer)
```json
{data_access_guide}
```

---

## Error Location
```json
{error_location}
```

## Stack Trace
```
{stack_trace}
```

## Error Key Analysis
```json
{error_key_analysis}
```

---

## Origin Tracing Checklist

Identify the observable status above, then follow the corresponding checklist. Check evidence in order; stop when you find a definitive cause.

### CRASH (solver not reached)

1. **[DataEngineer]** Does the traceback show a data access or parsing error? Does `data_access_guide` contain all expected sets/parameters with correct keys? → **data_engineer**
2. **[ModelExpert]** Does the code reference variables, indices, or parameters that do NOT exist in `model_expert_output`? (e.g., code uses `x[i,j]` but model defines `x[i,j,t]`) → **model_expert**
3. **[PythonDeveloper]** Is the code itself wrong? (syntax, wrong Gurobi API, wrong loop structure, wrong dictionary key format) → **python_developer**

### WRONG_OBJ (OPTIMAL but gap too large)

1. **[ModelExpert]** Inspect `model_expert_output`:
   - Are all expected variable families present? (transport, leasing/rental, inventory/storage, etc.)
   - Is the objective complete? (all cost terms: transport, holding, leasing, etc.)
   - Are all expected constraint families present? (flow conservation, demand, supply, capacity, etc.)
   - Missing component → **model_expert**
2. **[PythonDeveloper]** Compare `model_expert_output` against the Python code line by line:
   - Is every decision variable in the model created in code?
   - Is every constraint in the model implemented in code?
   - Are coefficients and indices correct?
   - Divergence → **python_developer**
3. **[DataEngineer]** Check parameter values in `data_access_guide`:
   - Are cost/demand/supply/capacity values reasonable and non-zero?
   - Wrong values → **data_engineer**

### INFEASIBLE

1. **[ModelExpert]** Check if the formulation is missing necessary constraints:
   - Flow conservation? Demand satisfaction? Supply limits? Capacity bounds? Inventory balance?
   - Missing constraint family → **model_expert**
2. **[PythonDeveloper]** Check if constraints are correctly translated to Gurobi:
   - Wrong sense (<= vs >=), wrong summation range, missing terms
   - Implementation diverges from model → **python_developer**
3. **[DataEngineer]** Check if data values create logical infeasibility:
   - Demand exceeds total supply, capacity = 0, contradictory parameters
   - Conflicting data → **data_engineer**

### UNBOUNDED

1. **[ModelExpert]** Are variable bounds and capacity constraints specified in the model? → **model_expert**
2. **[PythonDeveloper]** Are bounds correctly applied in the Gurobi code? → **python_developer**

---

## Decision Rules

| Evidence | Agent |
|----------|-------|
| Code references variable/index NOT in model_expert_output | model_expert |
| model_expert_output missing expected variable family or constraint family | model_expert |
| model_expert_output missing objective cost term | model_expert |
| Model is complete but code implementation diverges | python_developer |
| Syntax error or wrong Gurobi API usage | python_developer |
| Data access guide has wrong/missing keys or values | data_engineer |
| LaTeX symbol used as dictionary key in code | model_expert (model used symbol instead of source key) |

---

## Candidate Agents
{candidate_agents}

---

## Output

Return a JSON object:
- `suspected_agent`: one of {candidate_agents}
- `confidence`: 0–1
- `reason`: 1–2 sentences referencing specific evidence from the checklist
- `evidence`: the exact code line, model component, or data field that proves the diagnosis