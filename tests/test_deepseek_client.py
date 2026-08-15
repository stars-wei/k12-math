"""Tests for validating structured DeepSeek extraction results."""

from __future__ import annotations

import unittest

from deepseek_client import DeepSeekClient
from errors import UpstreamServiceError


class ExtractedProblemValidationTests(unittest.TestCase):
    def test_valid_target_and_reference_are_preserved(self) -> None:
        result = {
            "items": [
                {
                    "label": "（1）",
                    "target_expression": "-x**2/2 + 4*x + 2",
                    "reference_expressions": ["-x**2/2"],
                    "question_text": "指出图像经过怎样的变换得到",
                }
            ]
        }

        self.assertIs(DeepSeekClient._validate_extracted_problems(result), result)

    def test_missing_target_field_is_rejected(self) -> None:
        result = {
            "items": [
                {
                    "label": "（1）",
                    "reference_expressions": ["-x**2/2"],
                    "question_text": "指出图像经过怎样的变换得到",
                }
            ]
        }

        with self.assertRaises(UpstreamServiceError):
            DeepSeekClient._validate_extracted_problems(result)

    def test_unsafe_reference_expression_is_rejected(self) -> None:
        result = {
            "items": [
                {
                    "label": "（1）",
                    "target_expression": "-x**2/2 + 4*x + 2",
                    "reference_expressions": ["__import__('os')"],
                    "question_text": "指出图像经过怎样的变换得到",
                }
            ]
        }

        with self.assertRaises(UpstreamServiceError):
            DeepSeekClient._validate_extracted_problems(result)


if __name__ == "__main__":
    unittest.main()
