"""ParameterExtractor expert — extracts parameters from problem description."""

from .base_expert import BaseExpert


class ParameterExtractor(BaseExpert):
    ROLE_DESCRIPTION = (
        "You are a parameter extractor in the field of Operations Research "
        "and Optimization. Your expertise lies in extracting key parameters "
        "and data from problem descriptions."
    )

    FORWARD_TASK = """Now the origin problem is as follow:
{problem_description}

Let's analyse the problem step by step, and extract all the key parameters, sets, and data needed for optimization modeling.
And the comments from other experts are as follow:
{comments_text}

List all extracted parameters in a structured format:"""

    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined result.
The original problem is as follow:
{problem_description}
The feedback is as follow:
{feedback}
Your previous extraction:
{previous_answer}
The output format is a JSON structure followed by refined extraction:
{{
"is_caused_by_you": false,
"reason": "leave empty string if the problem is not caused by you",
"refined_result": "Your refined extraction..."
}}
"""

    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Parameter Extractor",
            description="Skilled in extracting key parameters and data from optimization problem descriptions.",
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
