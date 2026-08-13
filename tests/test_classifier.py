"""Tests for deterministic routing between supported quadratic tasks."""

from __future__ import annotations

import unittest

from classifier import match_task_from_question
from problem import Candidate, Problem


TASKS = [
    Candidate("quadratic-function-axis", "求一元二次函数图像的对称轴", ""),
    Candidate("quadratic-function-extremum", "求一元二次函数最值", ""),
]


class TaskRoutingTests(unittest.TestCase):
    def test_axis_question(self) -> None:
        problem = Problem("求函数 y=x²-4x+3 图像的对称轴", "x**2-4*x+3")
        self.assertEqual(match_task_from_question(problem, TASKS).id, "quadratic-function-axis")

    def test_extremum_question(self) -> None:
        problem = Problem("求函数 y=x²-4x+3 的最小值", "x**2-4*x+3")
        self.assertEqual(match_task_from_question(problem, TASKS).id, "quadratic-function-extremum")

    def test_ambiguous_question_uses_model_fallback(self) -> None:
        problem = Problem("研究函数 y=x²-4x+3", "x**2-4*x+3")
        self.assertIsNone(match_task_from_question(problem, TASKS))


if __name__ == "__main__":
    unittest.main()
