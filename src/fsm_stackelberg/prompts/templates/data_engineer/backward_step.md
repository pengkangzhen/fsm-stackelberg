You are DataEngineer. You need to judge whether the error is caused by your data mapping.

## Original Problem
{problem_description}

## Deterministic Data Access Guide and Catalog
{schema}

## Error Feedback
{error_info}

## Your Previous Output
{previous_output}

---

## Your Task

1. **Analyze the error**: Is this caused by your data mapping?
   - Check if you correctly mapped problem symbols to data paths
   - Check if you missed any parameters required by the problem
   - Check if every `source` exactly matches a catalog `data_id`
   - Check if indices match the catalog's positional index domains
   - Check if derived parameters are in `derived_parameters` with
     `derivation_logic` and `source_parameters`

2. **If this is NOT your fault**:
   - Set `is_caused_by_you: false`
   - Explain why in `reason`

3. **If this IS your fault**:
   - Set `is_caused_by_you: true`
   - Explain the issue in `reason`
   - Provide `refined_result` with your corrected output

---

## Output Format (STRICT JSON)

```json
{{
  "is_caused_by_you": false,
  "reason": "Explanation",
  "refined_result": null
}}
```

Or if caused by you:

```json
{{
  "is_caused_by_you": true,
  "reason": "Explanation of what went wrong",
  "refined_result": {{
    "model_inputs": {{
      "sets": [...],
      "parameters": [...]
    }}
  }}
}}
```

**CRITICAL**:
- If `is_caused_by_you: true`, you MUST provide `refined_result` with a complete DataEngineerOutput structure.
- Ensure every `source` is copied exactly from a catalog `data_id`.
- Put computed values in `derived_parameters`; do not emit `source: null`
  entries in `parameters`.
