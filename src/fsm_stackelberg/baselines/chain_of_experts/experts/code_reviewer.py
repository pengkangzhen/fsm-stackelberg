"""CodeReviewer expert — reviews generated code for correctness."""

from .base_expert import BaseExpert


class CodeReviewer(BaseExpert):
    ROLE_DESCRIPTION = (
        "You are a code reviewer specializing in operations research and "
        "optimization code. You review Python code that uses Gurobi or similar "
        "solvers, checking for correctness, efficiency, and best practices."
    )

    FORWARD_TASK = """Now the origin problem is as follow:
{problem_description}

And the code being reviewed is as follow:
{code_example}

And the comments from other experts are as follow:
{comments_text}

Review the code for:
1. Correctness of the mathematical model implementation
2. Proper use of solver API (Gurobi)
3. Constraint completeness
4. Variable bounds and types
5. Potential runtime issues
Give your review comments:"""

    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined review.
The original problem is as follow:
{problem_description}
The feedback is as follow:
{feedback}
Your previous review:
{previous_answer}
The output format is a JSON structure followed by refined review:
{{
"is_caused_by_you": false,
"reason": "leave empty string if the problem is not caused by you",
"refined_result": "Your refined review..."
}}
"""

    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Code Reviewer",
            description="Reviews optimization code for correctness, efficiency, and best practices.",
            model=model,
            provider=provider,
        )

    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        output = self._predict(
            self.forward_prompt,
            problem_description=problem["description"],
            code_example=problem["code_example"],
            comments_text=comments_text,
        )
        self.previous_answer = output
        return output

    def backward(self, feedback_pool):
        if not hasattr(self, "problem"):
            raise NotImplementedError("Please call forward first!")
        output = self._predict(
            self.backward_prompt,
            problem_description=self.problem["description"],
            previous_answer=self.previous_answer,
            feedback=feedback_pool.get_current_comment_text(),
        )
        return output
