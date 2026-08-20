"""Unit tests for the Grader module."""

import sys
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sympy as sp

from grader import (
    ErrorCategory,
    GradingReport,
    OverallVerdict,
    StudentStep,
    grade_solution,
    parse_raw_steps_fallback,
)
from problem import Problem


class TestGrader(unittest.TestCase):
    def test_grade_solution_fully_correct(self):
        """Test when all steps and conclusions are mathematically sound."""
        problem = Problem(
            question_text="求函数 y=-1/2x^2+4x+2 的最大值。",
            expression_sympy="-x**2/2 + 4*x + 2",
        )
        steps = [
            StudentStep(
                step_index=1,
                raw_text="y = -1/2(x^2 - 8x) + 2",
                expression_sympy="-1/2*(x**2 - 8*x) + 2",
                step_intent="factor_coefficient",
            ),
            StudentStep(
                step_index=2,
                raw_text="y = -1/2(x - 4)^2 + 10",
                expression_sympy="-1/2*(x - 4)**2 + 10",
                step_intent="completing_square",
            ),
            StudentStep(
                step_index=3,
                raw_text="因为 a = -1/2 < 0 开口向下，所以当 x = 4 时取得最大值 10",
                marker="therefore",
                claimed_axis="4",
                claimed_extremum_kind="max",
                claimed_extremum_value="10",
                step_intent="determine_extremum",
            ),
        ]

        report = grade_solution(problem, steps)

        self.assertEqual(report.overall_verdict, OverallVerdict.CORRECT)
        self.assertIsNone(report.first_error_step_index)
        self.assertEqual(report.valid_steps_count, 3)
        self.assertEqual(report.total_steps, 3)
        self.assertTrue(all(s.is_valid for s in report.steps_evaluation))
        self.assertIn("恭喜", report.summary_feedback)

    def test_grade_solution_calculation_error_in_completing_square(self):
        """Test when student makes a calculation mistake on the constant term."""
        problem = Problem(
            question_text="用配方法求函数 y=x^2-6x+5 的最小值。",
            expression_sympy="x**2 - 6*x + 5",
        )
        steps = [
            StudentStep(
                step_index=1,
                raw_text="y = (x - 3)^2 + 2",
                expression_sympy="(x - 3)**2 + 2",
                step_intent="completing_square",
            ),
            StudentStep(
                step_index=2,
                raw_text="所以当 x = 3 时，最小值为 2",
                marker="therefore",
                claimed_axis="3",
                claimed_extremum_kind="min",
                claimed_extremum_value="2",
                step_intent="determine_extremum",
            ),
        ]

        report = grade_solution(problem, steps)

        self.assertEqual(report.overall_verdict, OverallVerdict.INCORRECT)
        self.assertEqual(report.first_error_step_index, 1)
        self.assertFalse(report.steps_evaluation[0].is_valid)
        self.assertEqual(report.steps_evaluation[0].error_category, ErrorCategory.CALCULATION_ERROR.value)
        self.assertIn("不恒等", report.steps_evaluation[0].error_detail)

    def test_grade_solution_concept_error_on_extremum_kind(self):
        """Test when completing square is correct, but extremum type is confused (min instead of max)."""
        problem = Problem(
            question_text="求函数 y=-2x^2+12x-10 的最值。",
            expression_sympy="-2*x**2 + 12*x - 10",
        )
        steps = [
            StudentStep(
                step_index=1,
                raw_text="y = -2(x^2 - 6x) - 10",
                expression_sympy="-2*(x**2 - 6*x) - 10",
                step_intent="factor_coefficient",
            ),
            StudentStep(
                step_index=2,
                raw_text="y = -2(x - 3)^2 + 8",
                expression_sympy="-2*(x - 3)**2 + 8",
                step_intent="completing_square",
            ),
            StudentStep(
                step_index=3,
                raw_text="所以当 x = 3 时，函数取得最小值 8",
                marker="therefore",
                claimed_axis="3",
                claimed_extremum_kind="min",
                claimed_extremum_value="8",
                step_intent="determine_extremum",
            ),
        ]

        report = grade_solution(problem, steps)

        self.assertEqual(report.overall_verdict, OverallVerdict.PARTIALLY_CORRECT)
        self.assertEqual(report.first_error_step_index, 3)
        self.assertTrue(report.steps_evaluation[0].is_valid)
        self.assertTrue(report.steps_evaluation[1].is_valid)
        self.assertFalse(report.steps_evaluation[2].is_valid)
        self.assertEqual(report.steps_evaluation[2].error_category, ErrorCategory.CONCEPT_ERROR.value)
        self.assertIn("开口向下", report.steps_evaluation[2].error_detail)

    def test_grade_solution_sign_error_on_axis(self):
        """Test when axis sign is flipped (e.g. x = -3 instead of x = 3)."""
        problem = Problem(
            question_text="求函数 y=x^2-6x+5 的对称轴。",
            expression_sympy="x**2 - 6*x + 5",
        )
        steps = [
            StudentStep(
                step_index=1,
                raw_text="y = (x - 3)^2 - 4",
                expression_sympy="(x - 3)**2 - 4",
                step_intent="completing_square",
            ),
            StudentStep(
                step_index=2,
                raw_text="所以对称轴为 x = -3",
                marker="therefore",
                claimed_axis="-3",
                step_intent="determine_axis",
            ),
        ]

        report = grade_solution(problem, steps)

        self.assertEqual(report.overall_verdict, OverallVerdict.PARTIALLY_CORRECT)
        self.assertEqual(report.first_error_step_index, 2)
        self.assertTrue(report.steps_evaluation[0].is_valid)
        self.assertFalse(report.steps_evaluation[1].is_valid)
        self.assertEqual(report.steps_evaluation[1].error_category, ErrorCategory.SIGN_ERROR.value)

    def test_parse_raw_steps_fallback(self):
        """Test parsing raw Chinese steps into structured StudentSteps."""
        raw = """
        ∵ y = -1/2(x^2 - 8x) + 2
        y = -1/2(x - 4)^2 + 10
        ∴ 对称轴为 x = 4，最大值 10
        """
        steps = parse_raw_steps_fallback(raw)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].marker, "because")
        self.assertEqual(steps[2].marker, "therefore")
        self.assertEqual(steps[2].claimed_axis, "4")
        self.assertEqual(steps[2].claimed_extremum_kind, "max")
        self.assertEqual(steps[2].claimed_extremum_value, "10")


    def test_grade_normalized_steps_odd_function_correct(self):
        """Test grading for odd function analytic expression with perfect steps."""
        from grader import grade_normalized_steps

        stem = r"12. 已知函数 \( f(x) = \frac{ax + b}{1 + x^2} \) 是定义在 \( (-1, 1) \) 上的奇函数，且 \( f\left(\frac{1}{2}\right) = \frac{2}{5} \)，求函数 \( f(x) \) 的解析式。"
        steps = [
            {"step_number": 1, "marker": "because", "step_text": r"\( f(x) \) 是定义在 \( (-1, 1) \) 上的奇函数，∴ \( f(0) = 0 \)", "has_discontinuity": False},
            {"step_number": 2, "marker": "none", "step_text": r"即 \( \frac{b}{1 + 0^2} = 0 \)，∴ \( b = 0 \)", "has_discontinuity": False},
            {"step_number": 3, "marker": "none", "step_text": r"又 \( f\left(\frac{1}{2}\right) = \frac{\frac{1}{2}a}{1 + \frac{1}{4}} = \frac{2}{5} \)", "has_discontinuity": False},
            {"step_number": 4, "marker": "therefore", "step_text": r"\( a = 1 \)", "has_discontinuity": False},
            {"step_number": 5, "marker": "therefore", "step_text": r"\( f(x) = \frac{x}{1 + x^2} \)", "has_discontinuity": False},
        ]
        result = grade_normalized_steps(stem, steps)
        self.assertEqual(result["overall_verdict"], "CORRECT")
        self.assertIsNone(result["first_error_step_index"])
        self.assertEqual(result["total_score"], 10)
        self.assertEqual(result["max_total_score"], 10)
        self.assertIn("恭喜", result["summary_feedback"])

    def test_grade_normalized_steps_with_discontinuity(self):
        """A confirmed logical break is deducted without poisoning independent later steps."""
        from grader import grade_normalized_steps

        stem = r"12. 已知函数 \( f(x) = \frac{ax + b}{1 + x^2} \) 是定义在 \( (-1, 1) \) 上的奇函数，且 \( f\left(\frac{1}{2}\right) = \frac{2}{5} \)，求函数 \( f(x) \) 的解析式。"
        steps = [
            {"step_number": 1, "marker": "because", "step_text": r"\( f(x) \) 是定义在 \( (-1, 1) \) 上的奇函数，∴ \( f(0) = 0 \)", "has_discontinuity": False},
            {"step_number": 2, "marker": "none", "step_text": r"即 \( \frac{b}{1 + 0^2} = 0 \)，∴ \( b = 0 \)", "has_discontinuity": False},
            {"step_number": 3, "marker": "none", "step_text": r"又 \( f\left(\frac{1}{2}\right) = \frac{1}{1 + \frac{1}{4}} = \frac{2}{5} \)", "has_discontinuity": True, "pedagogical_warning": "缺少含 a 的未知数"},
            {"step_number": 4, "marker": "therefore", "step_text": r"\( a = 1 \)", "has_discontinuity": False},
            {"step_number": 5, "marker": "therefore", "step_text": r"\( f(x) = \frac{x}{1 + x^2} \)", "has_discontinuity": False},
        ]
        result = grade_normalized_steps(stem, steps)
        self.assertEqual(result["overall_verdict"], "PARTIALLY_CORRECT")
        self.assertEqual(result["first_error_step_index"], 3)
        self.assertEqual(result["steps_evaluation"][2]["step_score"], 0)
        self.assertEqual(result["steps_evaluation"][2]["evaluation_status"], "failed")
        self.assertEqual(result["steps_evaluation"][2]["error_category"], ErrorCategory.LOGIC_DISCONTINUITY.value)
        self.assertTrue(result["steps_evaluation"][3]["is_valid"])
        self.assertTrue(result["steps_evaluation"][4]["is_valid"])

    def test_acceptable_algebraic_omission_gets_full_credit(self):
        """Routine simplification omitted between an equation and a=1 is not an error."""
        from grader import grade_normalized_steps

        stem = r"12. 已知函数 \( f(x) = \frac{ax + b}{1 + x^2} \) 是定义在 \( (-1, 1) \) 上的奇函数，且 \( f\left(\frac{1}{2}\right) = \frac{2}{5} \)，求函数 \( f(x) \) 的解析式。"
        steps = [
            {"step_number": 1, "marker": "because", "step_text": r"\( f(x) \) 是奇函数，∴ \( f(0) = 0 \)", "continuity_status": "complete", "mathematical_validity": "valid"},
            {"step_number": 2, "marker": "none", "step_text": r"\( b = 0 \)", "continuity_status": "complete", "mathematical_validity": "valid"},
            {"step_number": 3, "marker": "none", "step_text": r"\( f\left(\frac{1}{2}\right) = \frac{\frac{1}{2}a}{1 + \frac{1}{4}} = \frac{2}{5} \)", "continuity_status": "complete", "mathematical_validity": "valid"},
            {
                "step_number": 4,
                "marker": "therefore",
                "step_text": r"\( a = 1 \)",
                "continuity_status": "acceptable_omission",
                "mathematical_validity": "valid",
                "omitted_reasoning": r"\( \frac{2a}{5}=\frac{2}{5} \)",
                "diagnostic_message": r"由 \(\frac{a/2}{1+1/4}=\frac{2}{5}\) 能够正确推出 \(a=1\)，属于常规约分跳步。",
            },
            {"step_number": 5, "marker": "therefore", "step_text": r"\( f(x) = \frac{x}{1 + x^2} \)", "continuity_status": "complete", "mathematical_validity": "valid"},
        ]

        result = grade_normalized_steps(stem, steps)

        self.assertEqual(result["overall_verdict"], "CORRECT")
        self.assertEqual(result["total_score"], result["max_total_score"])
        self.assertTrue(result["steps_evaluation"][3]["is_valid"])
        self.assertEqual(result["steps_evaluation"][3]["evaluation_status"], "passed_with_note")
        feedback = result["steps_evaluation"][3]["feedback"]
        self.assertIn(r"\(\frac{a/2}{1+1/4}=\frac{2}{5}\)", feedback)
        self.assertIn(r"\(a=1\)", feedback)
        self.assertIn(r"\( \frac{2a}{5}=\frac{2}{5} \)", feedback)

    def test_full_score_with_advisory_step_is_overall_correct(self):
        """A full-score advisory step cannot downgrade the overall verdict."""
        from grader import grade_normalized_steps

        stem = r"已知函数 \(f(x)=\frac{ax+b}{1+x^2}\) 是定义在 \((-1,1)\) 上的奇函数，且 \(f\left(\frac12\right)=\frac25\)，求函数 \(f(x)\) 的解析式。"
        steps = [
            {
                "step_number": 1,
                "marker": "because",
                "step_text": r"\(f(x)\) 是定义在 \((-1,1)\) 上的奇函数。",
                "continuity_status": "complete",
                "mathematical_validity": "valid",
            },
            {
                "step_number": 2,
                "marker": "therefore",
                "step_text": r"\(f(0)=0\)，即 \(\frac{b}{1+0^2}=0\)，∴ \(b=0\)。",
                "continuity_status": "complete",
                "mathematical_validity": "valid",
            },
            {
                "step_number": 3,
                "marker": "therefore",
                "step_text": r"\(f\left(\frac12\right)=\frac{\frac12a}{1+\frac14}=\frac25\)，∴ \(a=1\)。",
                "continuity_status": "acceptable_omission",
                "mathematical_validity": "valid",
                "omitted_reasoning": r"\(\frac{2a}{5}=\frac25\)，故 \(a=1\)。",
                "diagnostic_message": "省略了常规约分和方程求解过程。",
            },
            {
                "step_number": 4,
                "marker": "therefore",
                "step_text": r"\(f(x)=\frac{x}{1+x^2}\)。",
                "continuity_status": "complete",
                "mathematical_validity": "valid",
            },
        ]

        result = grade_normalized_steps(stem, steps)

        self.assertEqual(result["total_score"], 8)
        self.assertEqual(result["max_total_score"], 8)
        self.assertEqual(result["overall_verdict"], "CORRECT")
        self.assertIsNone(result["first_error_step_index"])
        self.assertEqual(
            result["steps_evaluation"][2]["evaluation_status"],
            "passed_with_note",
        )

    def test_ambiguous_ocr_step_requires_review_without_direct_deduction(self):
        """Unreadable OCR produces a review state instead of an automatic error."""
        from grader import grade_normalized_steps

        stem = "求函数解析式。"
        steps = [
            {"step_number": 1, "marker": "none", "step_text": "OCR 残缺算式", "continuity_status": "ambiguous", "mathematical_validity": "unknown", "diagnostic_message": "分母无法辨认"},
            {"step_number": 2, "marker": "therefore", "step_text": "结论完整", "continuity_status": "complete", "mathematical_validity": "valid"},
        ]

        result = grade_normalized_steps(stem, steps)

        self.assertEqual(result["overall_verdict"], "NEEDS_REVIEW")
        self.assertFalse(result["score_final"])
        self.assertIsNone(result["steps_evaluation"][0]["is_valid"])
        self.assertIsNone(result["steps_evaluation"][0]["step_score"])
        self.assertEqual(result["steps_evaluation"][0]["evaluation_status"], "unverified")
        self.assertTrue(result["steps_evaluation"][1]["is_valid"])

    def test_dual_ocr_disagreement_overrides_invalid_without_deduction(self):
        """Conflicting OCR evidence is reviewed even if the model also says invalid."""
        from grader import grade_normalized_steps

        steps = [
            {
                "step_number": 1,
                "marker": "therefore",
                "step_text": r"\(f(x)=\frac{1-t}{1+x}\)",
                "continuity_status": "logical_break",
                "mathematical_validity": "invalid",
                "ocr_agreement": "disagree",
                "secondary_ocr_evidence": r"\(f(x)=\frac{1-x}{1+x}\)",
                "verification_message": "变量 t 与 x 的识别不一致。",
                "ocr_fix_suggestion": r"f(x)=\frac{1-x}{1+x}",
            }
        ]

        result = grade_normalized_steps("求函数解析式。", steps)
        evaluation = result["steps_evaluation"][0]

        self.assertEqual(result["overall_verdict"], "NEEDS_REVIEW")
        self.assertIsNone(evaluation["step_score"])
        self.assertEqual(evaluation["evaluation_status"], "unverified")
        self.assertEqual(evaluation["continuity_status"], "ambiguous")
        self.assertEqual(evaluation["ocr_agreement"], "disagree")
        self.assertIn("变量 t 与 x", evaluation["feedback"])

    def test_agreed_student_error_is_still_deducted(self):
        """Agreement between OCR models allows a confirmed math error to be scored."""
        from grader import grade_normalized_steps

        steps = [
            {
                "step_number": 1,
                "marker": "therefore",
                "step_text": r"\(2[ax+b]+17\)",
                "continuity_status": "logical_break",
                "mathematical_validity": "invalid",
                "ocr_agreement": "agree",
                "secondary_ocr_evidence": r"\(2[ax+b]+17\)",
                "verification_message": "两套 OCR 对该表达式识别一致。",
            }
        ]

        result = grade_normalized_steps("右端为 2x+17。", steps)
        evaluation = result["steps_evaluation"][0]

        self.assertEqual(evaluation["step_score"], 0)
        self.assertEqual(evaluation["evaluation_status"], "failed")

    def test_bare_secondary_ocr_formulas_receive_katex_delimiters(self):
        """Bare formulas from OCR evidence are made renderable before display."""
        from grader import grade_normalized_steps

        steps = [
            {
                "step_number": 1,
                "step_text": r"\(f(x)=\frac{1-t}{1+x}\)",
                "continuity_status": "ambiguous",
                "mathematical_validity": "unknown",
                "ocr_agreement": "disagree",
                "secondary_ocr_evidence": (
                    r"f(t)=\frac{1-t}{1+t}；f(x)=\frac{1-x}{1+x}"
                ),
                "verification_message": "两套 OCR 识别不一致。",
            }
        ]

        result = grade_normalized_steps("求函数解析式。", steps)
        feedback = result["steps_evaluation"][0]["feedback"]

        self.assertIn(r"\(f(t)=\frac{1-t}{1+t}\)", feedback)
        self.assertIn(r"\(f(x)=\frac{1-x}{1+x}\)", feedback)
        self.assertIn(r"\)；\(", feedback)

    def test_delimited_secondary_ocr_formula_is_not_wrapped_twice(self):
        """Evidence already formatted for KaTeX remains unchanged."""
        from grader import grade_normalized_steps

        evidence = r"\(f(x)=\frac{1-x}{1+x}\)"
        steps = [
            {
                "step_number": 1,
                "step_text": evidence,
                "continuity_status": "ambiguous",
                "mathematical_validity": "unknown",
                "ocr_agreement": "uncertain",
                "secondary_ocr_evidence": evidence,
            }
        ]

        result = grade_normalized_steps("求函数解析式。", steps)
        feedback = result["steps_evaluation"][0]["feedback"]

        self.assertEqual(feedback.count(r"\("), 1)
        self.assertEqual(feedback.count(r"\)"), 1)

    def test_plain_secondary_ocr_text_is_not_treated_as_math(self):
        """Ordinary Chinese evidence remains ordinary explanatory text."""
        from grader import grade_normalized_steps

        steps = [
            {
                "step_number": 1,
                "step_text": "字迹不清",
                "continuity_status": "ambiguous",
                "mathematical_validity": "unknown",
                "ocr_agreement": "uncertain",
                "secondary_ocr_evidence": "该处字迹无法辨认",
            }
        ]

        result = grade_normalized_steps("求函数解析式。", steps)
        feedback = result["steps_evaluation"][0]["feedback"]

        self.assertIn("复核 OCR 识别为：该处字迹无法辨认", feedback)
        self.assertNotIn(r"\(", feedback)


if __name__ == "__main__":
    unittest.main()

