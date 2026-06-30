"""ModelingKnowledgeSupplementExpert — supplements modeling knowledge."""

from .base_expert import BaseExpert


class ModelingKnowledgeSupplementExpert(BaseExpert):
    ROLE_DESCRIPTION = (
        "You are a modeling knowledge supplement expert in the field of "
        "Operations Research and Optimization. You provide additional modeling "
        "knowledge, techniques, and insights that complement the work of other experts."
    )

    FORWARD_TASK = """Now the origin problem is as follow:
{problem_description}

And the comments from other experts are as follow:
{comments_text}

Based on the problem and the comments from other experts, provide additional modeling knowledge, tips, or techniques that could help improve the formulation. Focus on:
1. Common pitfalls in this type of problem
2. Reformulation techniques that could improve solver performance
3. Symmetry breaking or valid inequalities if applicable
Your supplement:"""

    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined result.
The original problem is as follow:
{problem_description}
The feedback is as follow:
{feedback}
Your previous supplement:
{previous_answer}
The output format is a JSON structure followed by refined supplement:
{{
"is_caused_by_you": false,
"reason": "leave empty string if the problem is not caused by you",
"refined_result": "Your refined supplement..."
}}
"""

    def __init__(self, model=None, provider=None):
        super().__init__(
            name="Modeling Knowledge Supplement Expert",
            description="Provides additional modeling knowledge and techniques for optimization.",
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
