"""Tests for detecting every requested task in a compound question."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from multi_solver import (
    ProblemItemOutcome,
    TaskIntent,
    TaskOutcome,
    detect_task_intents,
    match_requested_strategy,
    render_all_results,
    render_problem_items,
)
from problem import Candidate, Problem
from solve import Answer, Solution


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
        self.assertIn("知识图谱中尚没有该题型的可执行策略", page)

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

    def test_vertex_is_detected(self) -> None:
        intents = detect_task_intents("求函数的顶点坐标和对称轴")
        self.assertEqual(
            [intent.id for intent in intents],
            ["quadratic-function-vertex", "quadratic-function-axis"],
        )

    @patch("multi_solver.load_task_candidates")
    @patch("multi_solver.select_strategy")
    @patch("multi_solver.solve_task")
    def test_axis_reuses_vertex_solution(
        self,
        solve_task_mock,
        select_strategy_mock,
        load_tasks_mock,
    ) -> None:
        question = "求函数图像的顶点和对称轴"
        problem = Problem(question, "x**2 - 4*x + 3")
        load_tasks_mock.return_value = [
            Candidate("quadratic-function-vertex", "求顶点", ""),
            Candidate("quadratic-function-axis", "求对称轴", ""),
        ]
        select_strategy_mock.return_value = Candidate("vertex-by-completing-square", "配方法", "")
        solve_task_mock.return_value = Solution(
            "quadratic-function-vertex",
            "求顶点",
            "配方法",
            None,
            [],
            Answer("求顶点", "顶点为 (2, -1)。", r"\left(2,\,-1\right)"),
            {"axis": 2, "vertex_value": -1},
        )
        from multi_solver import solve_all_tasks

        outcomes = solve_all_tasks(problem, "password", "url", client=None)  # type: ignore[arg-type]
        self.assertEqual([outcome.status for outcome in outcomes], ["solved", "reused"])
        self.assertEqual(solve_task_mock.call_count, 1)
        self.assertIn("复用", render_all_results(problem, outcomes))


if __name__ == "__main__":
    unittest.main()
