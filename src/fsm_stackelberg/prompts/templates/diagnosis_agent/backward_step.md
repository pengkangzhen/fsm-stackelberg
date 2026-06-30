You are {agent_name}. You need to judge whether the error is caused by your output.

## Original Problem
{problem_description}

## Error Feedback
{error_info}

## Your Previous Output
{previous_output}

---

## Your Task

Analyze whether the error is caused by your output. If it is, provide a refined result.

Output ONLY a JSON object with:
- `is_caused_by_you`: true if the error is your fault, false otherwise
- `reason`: Brief explanation
- `refined_result`: If caused by you, provide your corrected output (optional)