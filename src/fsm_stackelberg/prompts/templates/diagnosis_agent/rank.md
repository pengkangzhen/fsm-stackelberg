Rank the three pipeline layers by how likely each is the **root-cause** of the current failure. Same evidence budget as single-judge diagnosis; output is a **full ordering**, not a single accusation and **not** calibrated probabilities.

## Observable Status

**Gurobi Status**: `{gurobi_status}`

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

Use the same layered checklist as diagnosis: CRASH → data/model/code; WRONG_OBJ / INFEASIBLE / UNBOUNDED → prefer formulation gaps before code translation before data values.

**Hard rules (do not violate):**

1. **Symptom locus ≠ root cause.** A Python `Traceback`, `NameError`, `KeyError`, or crash line in `python_code` is often a *downstream symptom* of a bad model/data contract. Do **not** tip `python_developer` solely because the stack names a Python file.
2. **Force-zero / empty-sea smell → prefer `model_expert`.** If ModelExpert constraints or Python code force sea repositioning to zero (`y_in[·]=0` and/or `y_out[·]=0`, names like `Force_Zero`, `Injected_Force_Zero`, `force_zero_yin` / `force_zero_yout`), tip **`model_expert` first** — even under CRASH or OPTIMAL-with-gap. That is a formulation fault, not a coding typo.
3. **OPTIMAL / WRONG_OBJ** (solver ran, large objective gap): prefer formulation (`model_expert`) before translation (`python_developer`) before data values, unless clear evidence the wrong objective/constraints exist only in code and not in the model JSON.
4. Rank by **causal origin**, not by who last touched the failing line.

---

## Layers to rank
{candidate_agents}

---

## Output

Return a JSON object with **only**:
- `rank`: ordered list of all three agents, **most likely root cause first**. Must be a permutation of `data_engineer`, `model_expert`, `python_developer`.
- `rationale`: 1–3 sentences citing concrete evidence (not a probability).

Do **not** include `confidence`, `probability`, or any numeric posterior fields. If you would assign confidences, omit them and keep only the order.
