"""Tests for validating structured DeepSeek extraction results."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

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


class OcrNormalizationContractTests(unittest.TestCase):
    def test_continuity_schema_separates_omission_ambiguity_and_logical_break(self) -> None:
        client = object.__new__(DeepSeekClient)
        with patch.object(DeepSeekClient, "_tool_call", return_value={}) as tool_call:
            client.normalize_ocr_math_steps("sample")

        parameters = tool_call.call_args.args[2]
        step_properties = parameters["properties"]["steps"]["items"]["properties"]
        self.assertEqual(
            step_properties["continuity_status"]["enum"],
            ["complete", "acceptable_omission", "ambiguous", "logical_break"],
        )
        self.assertEqual(
            step_properties["mathematical_validity"]["enum"],
            ["valid", "invalid", "unknown"],
        )
        self.assertEqual(
            step_properties["ocr_agreement"]["enum"],
            ["agree", "disagree", "uncertain", "not_checked"],
        )
        self.assertIn(
            "每个数学表达式必须分别使用 KaTeX 行内定界符",
            step_properties["secondary_ocr_evidence"]["description"],
        )
        self.assertNotIn("has_discontinuity", step_properties)
        self.assertIn("KaTeX", step_properties["diagnostic_message"]["description"])
        self.assertIn("\\(", step_properties["omitted_reasoning"]["description"])
        system_prompt = tool_call.call_args.kwargs["system"]
        self.assertIn("acceptable_omission", system_prompt)
        self.assertIn(r"前式不含 \(a\)", system_prompt)
        self.assertIn("KaTeX 输出规范", system_prompt)
        self.assertIn(
            r"\(\frac{\frac{a}{2}}{1+\frac{1}{4}}=\frac{2}{5}\)",
            system_prompt,
        )
        self.assertIn("不得使用 / 表示分数", system_prompt)
        self.assertIn("不要输出未包裹的 ASCII 算式", system_prompt)
        self.assertIn("双 OCR 证据规则", system_prompt)

    def test_dual_ocr_disagreement_is_forced_to_unknown(self) -> None:
        client = object.__new__(DeepSeekClient)
        model_result = {
            "question_stem": "sample",
            "steps": [
                {
                    "step_number": 1,
                    "ocr_agreement": "disagree",
                    "continuity_status": "logical_break",
                    "mathematical_validity": "invalid",
                }
            ],
            "overall_summary": "sample",
        }
        with patch.object(DeepSeekClient, "_tool_call", return_value=model_result):
            result = client.normalize_ocr_math_steps("primary", "secondary")

        self.assertEqual(result["steps"][0]["continuity_status"], "ambiguous")
        self.assertEqual(result["steps"][0]["mathematical_validity"], "unknown")

    def test_dual_ocr_not_checked_is_downgraded_to_uncertain(self) -> None:
        client = object.__new__(DeepSeekClient)
        model_result = {
            "question_stem": "sample",
            "steps": [
                {
                    "step_number": 1,
                    "ocr_agreement": "not_checked",
                    "continuity_status": "complete",
                    "mathematical_validity": "valid",
                    "verification_message": "",
                }
            ],
            "overall_summary": "sample",
        }
        with patch.object(DeepSeekClient, "_tool_call", return_value=model_result):
            result = client.normalize_ocr_math_steps("primary", "secondary")

        step = result["steps"][0]
        self.assertEqual(step["ocr_agreement"], "uncertain")
        self.assertEqual(step["continuity_status"], "ambiguous")
        self.assertEqual(step["mathematical_validity"], "unknown")


class DeepSeekTraceTests(unittest.TestCase):
    def test_tool_call_reports_request_and_response_without_api_key(self) -> None:
        events = []
        client = DeepSeekClient(
            api_key="secret-test-key",
            model="deepseek-test",
            trace_callback=lambda stage, payload, duration: events.append(
                (stage, payload, duration)
            ),
        )
        response = MagicMock()
        response.__enter__.return_value = response
        upstream_payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "sample_tool",
                                    "arguments": '{"answer": 1}',
                                }
                            }
                        ]
                    }
                }
            ]
        }

        with patch("deepseek_client.urlopen", return_value=response), patch(
            "deepseek_client.json.load", return_value=upstream_payload
        ):
            result = client._tool_call(
                "sample_tool",
                "sample description",
                {"type": "object"},
                "sample system",
                "sample user",
            )

        self.assertEqual(result, {"answer": 1})
        self.assertEqual([event[0] for event in events], ["deepseek_request", "deepseek_response"])
        self.assertEqual(events[0][1]["system"], "sample system")
        self.assertNotIn("api_key", events[0][1])
        self.assertNotIn("Authorization", events[0][1])
        self.assertIsInstance(events[1][2], int)


class SafeJsonDecodeTests(unittest.TestCase):
    def test_double_newline_becomes_real_line_breaks(self) -> None:
        raw = r'{"question_stem":"第一问\n\n(2)第二问"}'

        result = DeepSeekClient._safe_decode_json(raw)

        self.assertEqual(result["question_stem"], "第一问\n\n(2)第二问")

    def test_latex_command_starting_with_n_keeps_backslash(self) -> None:
        raw = r'{"text":"\\(x\neq 0\\)"}'

        result = DeepSeekClient._safe_decode_json(raw)

        self.assertEqual(result["text"], r"\(x\neq 0\)")

if __name__ == "__main__":
    unittest.main()
