"""Comment pool with visibility matrix for Chain-of-Experts."""

import numpy as np

from .comment import Comment


class CommentPool:
    """Manages comments from all experts with a visibility matrix.

    The visibility matrix controls which expert can see which other expert's
    comments. In the default (fully visible) setting, every expert sees all
    previous comments.
    """

    def __init__(self, all_experts: list, visible_matrix: np.ndarray):
        self.all_experts = all_experts
        self.visible_matrix = visible_matrix  # shape: (n_experts, n_experts)
        self.comments: list[Comment] = []

    def add_comment(self, comment: Comment):
        self.comments.append(comment)

    def pop_comment(self) -> Comment:
        return self.comments.pop()

    def get_current_comment_text(self) -> str:
        """Return concatenated text of all comments."""
        return "\n\n".join(str(c) for c in self.comments)

    def get_comments_for_expert(self, expert) -> str:
        """Return comments visible to a specific expert."""
        expert_idx = self.all_experts.index(expert)
        visible_parts = []
        for comment in self.comments:
            commenter_idx = self.all_experts.index(comment.expert)
            if self.visible_matrix[expert_idx][commenter_idx]:
                visible_parts.append(str(comment))
        return "\n\n".join(visible_parts)
