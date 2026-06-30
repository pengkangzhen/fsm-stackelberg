"""TerminologyInterpreter expert — interprets domain-specific terminology."""

from .base_expert import BaseExpert


class TerminologyInterpreter(BaseExpert):
    ROLE_DESCRIPTION = (
        "You are a terminology interpreter in the field of Operations Research "
        "and Optimization. You are good at interpreting domain-specific terms "
        "and converting them into mathematical descriptions."
    )

    FORWARD_TASK = """Now the origin problem is as follow:
{problem_description}

Let's analyse the problem step by step, and then give your interpretation of the terminology used in this problem.
And the comments from other experts are as follow:
{comments_text}

Your interpretation:"""

    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined result.
The original problem is as follow:
{problem_description}
The feedback is as follow:
{feedback}
Your previous interpretation:
{previous_answer}
The output format is a JSON structure followed by refined interpretation:
{{
"is_caused_by_you": false,
"reason": "leave empty string if the problem is not caused by you",
"refined_result": "Your refined interpretation..."
}}
"""

    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Terminology Interpreter",
            description="Proficient in interpreting domain-specific terminology in optimization problems.",
            model=model,
            provider=provider,
        )

    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        output = self._predict(
            self.forward_prompt,
            problem_description=problem["description"],
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
