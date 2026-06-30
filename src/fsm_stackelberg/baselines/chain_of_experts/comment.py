"""Comment data class for Chain-of-Experts."""


class Comment:
    """A comment produced by an expert during the CoE collaboration loop."""

    def __init__(self, expert, comment_text: str):
        self.expert = expert
        self.comment_text = comment_text

    def __str__(self):
        return f"[{self.expert.name}]: {self.comment_text}"
