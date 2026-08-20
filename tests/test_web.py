"""Regression tests for preserving confirmed OCR text."""

from __future__ import annotations

import unittest
from pathlib import Path

from web import build_problem, multipart_form, normalize_item_label


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


class StudioContractTests(unittest.TestCase):
    def test_multipart_form_returns_named_fields_and_image(self) -> None:
        boundary = "----k12-math-test"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="intent"\r\n\r\n'
            "solve\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            "PaddlePaddle/PaddleOCR-VL-1.5\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; filename="question.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + b"PNG-DATA\r\n" + f"--{boundary}--\r\n".encode("utf-8")

        fields, image, image_type, image_filename = multipart_form(
            f"multipart/form-data; boundary={boundary}",
            body,
        )

        self.assertEqual(fields["intent"], "solve")
        self.assertEqual(fields["model"], "PaddlePaddle/PaddleOCR-VL-1.5")
        self.assertEqual(image, b"PNG-DATA")
        self.assertEqual(image_type, "image/png")
        self.assertEqual(image_filename, "question.png")

    def test_templates_keep_one_web_interface_and_one_cli_report(self) -> None:
        template_dir = Path(__file__).resolve().parents[1] / "src" / "templates"
        self.assertEqual(
            {path.name for path in template_dir.glob("*.html")},
            {"studio.html", "result.html"},
        )

    def test_studio_exposes_three_modes_and_four_grading_states(self) -> None:
        studio = (
            Path(__file__).resolve().parents[1] / "src" / "templates" / "studio.html"
        ).read_text(encoding="utf-8")

        for value, label in (("auto", "自动"), ("grade", "批改"), ("solve", "求解")):
            self.assertIn(f'value="{value}"', studio)
            self.assertIn(f"<span>{label}</span>", studio)
        self.assertIn('status === "passed_with_note"', studio)
        self.assertIn('status === "unverified"', studio)
        self.assertIn('verdict === "NEEDS_REVIEW"', studio)
        self.assertIn(".grade-step.is-passed .method", studio)
        self.assertIn(".grade-step.is-invalid .method", studio)
        self.assertIn(".grade-step.is-unverified .grade-comment", studio)
        self.assertIn(".grade-step.is-invalid .grade-comment", studio)
        self.assertNotIn(".grade-step.is-unverified { border-left", studio)
        self.assertNotIn(".grade-step.is-invalid { border-left", studio)
        self.assertIn('class="result-block reference-solution"', studio)


if __name__ == "__main__":
    unittest.main()
