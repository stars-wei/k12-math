"""Tests for detecting every requested task in a compound question."""

from __future__ import annotations

import unittest

from multi_solver import (
    ProblemItemOutcome,
    TaskIntent,
    TaskOutcome,
    detect_task_intents,
    match_requested_strategy,
    render_all_results,
    render_problem_items,
)
from problem import Problem


COMPOUND_QUESTION = """已知一元二次函数 y=-1/2*x^2+4*x+2。
(1) 指出它的图像可以由函数 y=-1/2*x^2 的图像经过怎样的变换得到；
(2) 指出它的图像的对称轴，试述函数值的变化趋势及最大值或最小值。"""


class MultiTaskTests(unittest.TestCase):
    def test_detects_all_four_tasks_in_order(self) -> None:
        intents = detect_task_intents(COMPOUND_QUESTION)
        self.assertEqual(
            [intent.id for intent in intents],
            [
                "quadratic-function-transformation",
                "quadratic-function-axis",
                "quadratic-function-monotonicity",
                "quadratic-function-extremum",
            ],
        )

    def test_unsupported_task_message(self) -> None:
        problem = Problem(COMPOUND_QUESTION, "-x**2/2+4*x+2")
        outcome = TaskOutcome(
            TaskIntent("quadratic-function-transformation", "判断图像变换", "图像变换"),
            "not_registered",
        )
        page = render_all_results(problem, [outcome])
        self.assertIn("该题型未入库", page)

    def test_explicit_completing_square_strategy_is_honored(self) -> None:
        from problem import Candidate

        candidates = [
            Candidate("axis-formula", "公式法", ""),
            Candidate("axis-by-completing-square", "配方法求对称轴", ""),
        ]
        selected = match_requested_strategy("用配方法求对称轴", candidates)
        self.assertEqual(selected.id, "axis-by-completing-square")

    def test_multiple_items_are_grouped(self) -> None:
        intent = TaskIntent("quadratic-function-axis", "求对称轴", "对称轴")
        item1 = Problem("用配方法求对称轴", "x**2", None)
        item2 = Problem("用配方法求对称轴", "-x**2", None)
        page = render_problem_items(
            item1.question_text,
            [
                ProblemItemOutcome("（1）", item1, [TaskOutcome(intent, "not_registered")]),
                ProblemItemOutcome("（2）", item2, [TaskOutcome(intent, "not_registered")]),
            ],
        )
        self.assertIn("识别出 2 个待求函数", page)
        self.assertIn("（1）", page)
        self.assertIn("（2）", page)


if __name__ == "__main__":
    unittest.main()
