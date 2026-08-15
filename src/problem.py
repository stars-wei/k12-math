"""Small, transport-neutral data contracts for problem classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    """A normalized text question plus the expression the local solver can execute."""

    question_text: str
    expression_sympy: str
    expression_latex: str | None = None
    item_question_text: str | None = None
    reference_expressions: tuple[str, ...] = ()

    @property
    def task_text(self) -> str:
        """Return the current item's request, falling back to the full question."""
        return self.item_question_text or self.question_text


@dataclass(frozen=True)
class Candidate:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class Classification:
    """A graph-validated decision returned by the classification layer."""

    task: Candidate
    strategy: Candidate
    reason: str
    confidence: float | None
