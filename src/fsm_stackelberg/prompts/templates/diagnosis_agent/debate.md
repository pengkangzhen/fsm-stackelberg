Debate the root cause of this pipeline failure under your assigned analytical lens, then cast your vote.

## Your Lens (Debater {lens_id} of 3)

{lens}

## Rules

- Name exactly one layer as the most likely root cause.
- Ground your argument in specific evidence below; cite what you rely on.
- The traceback / solver surface marks WHERE the pipeline failed, not
  necessarily WHO caused it — a downstream symptom from an upstream defect
  is common in this pipeline (DataEngineer -> ModelExpert -> PythonDeveloper).
- Do NOT report confidence, scores, or probabilities. A vote plus an
  argument is all the panel needs.

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

Vote now: output `suspected_agent` (exactly one candidate layer) and
`argument` (your strongest evidence-grounded case, at most 120 words).
