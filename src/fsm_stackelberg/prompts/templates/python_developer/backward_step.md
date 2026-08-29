You are PythonDeveloper. You need to judge whether the error is caused by your code.

## Original Problem
{problem_description}

## Model Expert Output
{model_expert_output}

## Data Access Guide
{data_access_guide}

The authoritative resolved JSON is keyed by exact DE symbol. Use its
`access_pattern`; never use the symbol alias or dotted `source` as a guessed
runtime key. For `kind: "records"` and `"record_value"`, `access_pattern`
points to a list of record dictionaries. Rebuild tuple keys in the exact
`index_fields` order. A `record_value` entry takes each tuple's value from
`value_field`; a `records` entry represents the tuple set itself. Respect
`sparse`/`access_hint`. The physical schema section is supplementary only.

## Error Feedback
{error_info}

## Your Previous Output (Code)
{previous_output}

---

## Your Task

1. **Analyze the error**: Is this caused by your code?
   - Check if variable/parameter names match model definition and data
   - Check if data access patterns are correct (using proper keys from data)
   - Check if Gurobi API usage is correct
   - Check for common issues:
     - Using symbol aliases directly as dictionary keys
     - Treating a records-list `access_pattern` as a tuple dictionary
     - Ignoring `index_fields`, `value_field`, or sparse membership

2. **If this is NOT your fault**:
   - Set `is_caused_by_you: false`
   - Explain why in `reason`

3. **If this IS your fault**:
   - Set `is_caused_by_you: true`
   - Explain the issue in `reason`
   - Provide `refined_result` with your corrected Python code

---

## Output Format (STRICT JSON)

```json
{{
  "is_caused_by_you": false,
  "reason": "Explanation of why this is not your fault",
  "refined_result": null
}}
```

Or if caused by you:

```json
{{
  "is_caused_by_you": true,
  "reason": "Explanation of what went wrong and how you fixed it",
  "refined_result": "import gurobipy as gp\n\ndef optimize(data): ..."
}}
```

**CRITICAL**:
- If `is_caused_by_you: true`, you MUST provide `refined_result` with complete, working Python code.
- Copy resolved `access_pattern` values from `data_access_guide`; do not guess keys from symbols or sources.
- The code must be a complete `optimize(data)` function that returns a dict with `status` and `objective_value`.
- Name the Gurobi model `model` (not `m`); name transport-mode indices `mode` (not `m`). Do not shadow the model object with a loop index.
