"""Regression tests for preserving confirmed OCR text."""

from __future__ import annotations

import unittest

from web import build_problem, normalize_item_label


class ConfirmedQuestionTests(unittest.TestCase):
    def test_expression_extraction_cannot_truncate_question(self) -> None:
        question = "已知函数 y=x²。（1）求对称轴；（2）求最值。"
        extracted = {
            "label": "（1）",
            "target_expression": "x**2",
            "reference_expressions": ["-x**2"],
            "question_text": "求对称轴",
        }
        problem = build_problem(question, extracted)
        self.assertEqual(problem.question_text, question)
        self.assertEqual(problem.task_text, "求对称轴")
        self.assertEqual(problem.reference_expressions, ("-x**2",))

    def test_item_labels_are_locally_normalized(self) -> None:
        self.assertEqual(normalize_item_label("ги1гй", 1, 2), "（1）")
        self.assertEqual(normalize_item_label("unexpected", 2, 2), "（2）")
        self.assertEqual(normalize_item_label("(1)", 1, 1), "")


if __name__ == "__main__":
    unittest.main()
