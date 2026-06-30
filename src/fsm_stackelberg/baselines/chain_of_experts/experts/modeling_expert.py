"""ModelingExpert — formulates MIP models from problem description."""

from .base_expert import BaseExpert


class ModelingExpert(BaseExpert):
    ROLE_DESCRIPTION = (
        "You are a modeling expert specialized in the field of Operations Research "
        "and Optimization. Your expertise lies in Mixed-Integer Programming (MIP) "
        "models, and you possess an in-depth understanding of various modeling "
        "techniques within the realm of operations research."
    )

    FORWARD_TASK = """Now the origin problem is as follow:
{problem_description}

And the comments from other experts are as follow:
{comments_text}

Give your MIP model of this problem. Additionally, please note that your model needs to be a solvable linear programming model or a mixed-integer programming model. This also means that the expressions of the constraint conditions can only be equal to, greater than or equal to, or less than or equal to (> or < are not allowed to appear and should be replaced to be \\geq or \\leq).

Your output format should be a JSON like this:
{{
"VARIABLES": "A mathematical description about variables",
"CONSTRAINS": "A mathematical description about constrains",
"OBJECTIVE": "A mathematical description about objective"
}}
"""

    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined result.
The original problem is as follow:
{problem_description}
The feedback is as follow:
{feedback}
The modeling you give previously is as follow:
{previous_answer}
The output format is a JSON structure followed by refined code:
{{
"is_caused_by_you": false,
"reason": "leave empty string if the problem is not caused by you",
"refined_result": "Your refined result"
}}
"""

    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Modeling Expert",
            description="Proficient in constructing mathematical optimization models based on the extracted information.",
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
        # Enforce MIP convention: strict inequality → weak
        output = output.replace(" < ", " \\leq ").replace(" > ", " \\geq ")
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
