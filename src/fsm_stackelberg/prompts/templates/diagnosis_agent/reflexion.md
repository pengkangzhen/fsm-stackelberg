Reflect on the initial attribution below before confirming or revising it.

## Episode Memory (what has already been tried in this episode)

{memory}

If the memory is non-empty, every accusation listed there was followed by a
repair attempt and a re-solve, and the failure still persists — otherwise
you would not be diagnosing again.

## Initial Attribution (this round's first judgement)

- Suspected layer: `{initial_suspect}`
- Claimed reason: {initial_reason}

## Reflection Rules

- If the memory contradicts the initial attribution — e.g. that layer was
  already accused and its repair did not resolve the failure — revise to
  the next most plausible layer supported by evidence below.
- If the memory is empty or nothing contradicts the attribution, confirm it.
- Do NOT restate confidence or probabilities; argue from evidence.

## Observable Status

**Gurobi Status**: `{gurobi_status}`

| gurobi_status | Status | Description |
|---------------|--------|-------------|
| `""` | CRASH | Code never reached the solver |
| `OPTIMAL` | WRONG_OBJ | Solver found a solution but objective gap exceeds threshold |
| `INFEASIBLE` | INFEASIBLE | Model has no feasible solution |
| `UNBOUNDED` | UNBOUNDED | Model is unbounded |
| `ERROR` | SOLVER_ERROR | Gurobi internal error |

## Candidate Layers

{candidate_agents}

## Evidence

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

### Error Location
```json
{error_location}
```

### Stack Trace
```
{stack_trace}
```

### Error Key Analysis
```json
{error_key_analysis}
```

Output: `suspected_agent` (your final attribution, exactly one candidate
layer) and `reflection` (what the memory or evidence changed, or why the
initial attribution stands, at most 120 words).
