"""Intelligent step-by-step mathematical handwriting grader.

Combines DeepSeek semantic normalization with SymPy deterministic validation
to pinpoint first errors, calculate step-by-step scoring, and generate pedagogical feedback.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

import sympy as sp

from deepseek_client import DeepSeekClient
from problem import Problem


class StepMarker(str, Enum):
    BECAUSE = "because"  # ∵
    THEREFORE = "therefore"  # ∴
    NONE = "none"


class OverallVerdict(str, Enum):
    CORRECT = "CORRECT"  # 全部步骤与最终结论正确
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"  # 部分步骤正确，后续存在错误
    INCORRECT = "INCORRECT"  # 第一步即存在严重错误或完全不符
    NEEDS_REVIEW = "NEEDS_REVIEW"  # OCR 或卷面信息不足，暂时无法可靠评分


class ContinuityStatus(str, Enum):
    COMPLETE = "complete"
    ACCEPTABLE_OMISSION = "acceptable_omission"
    AMBIGUOUS = "ambiguous"
    LOGICAL_BREAK = "logical_break"


class MathematicalValidity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class StepEvaluationStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_NOTE = "passed_with_note"
    UNVERIFIED = "unverified"
    FAILED = "failed"


class ErrorCategory(str, Enum):
    CALCULATION_ERROR = "CALCULATION_ERROR"  # 计算失误（如常数项算错、系数漏乘）
    CONCEPT_ERROR = "CONCEPT_ERROR"  # 概念理解错误（如开口向下误当成最小值）
    SIGN_ERROR = "SIGN_ERROR"  # 符号写反（如 x-3 误写为 x+3）
    OMISSION_ERROR = "OMISSION_ERROR"  # 关键未知数/条件漏写
    LOGIC_DISCONTINUITY = "LOGIC_DISCONTINUITY"  # 逻辑断裂或无依据跳步
    INCOMPLETE = "INCOMPLETE"  # 步骤不完整或未得出结论


@dataclass
class StudentStep:
    step_index: int
    raw_text: str
    marker: str = "none"  # because / therefore / none
    expression_latex: str = ""
    expression_sympy: str = ""
    step_intent: str = "other"
    claimed_axis: str = ""
    claimed_vertex: str = ""
    claimed_extremum_kind: str = "none"  # max / min / none
    claimed_extremum_value: str = ""
    claimed_answer: str = ""
    continuity_status: str = ContinuityStatus.COMPLETE.value
    mathematical_validity: str = MathematicalValidity.VALID.value
    omitted_reasoning: str = ""
    diagnostic_message: str = ""
    ocr_agreement: str = "not_checked"
    secondary_ocr_evidence: str = ""
    verification_message: str = ""
    # 兼容旧版结构化结果；新结果应使用 continuity_status。
    has_discontinuity: bool = False
    pedagogical_warning: str = ""


@dataclass
class StepEvaluation:
    step_index: int
    raw_text: str
    marker: str
    is_valid: bool | None
    step_score: int | None = 2
    max_score: int = 2
    feedback: str = ""
    error_category: str | None = None
    error_detail: str | None = None
    sympy_proof: str | None = None
    evaluation_status: str = ""
    continuity_status: str = ContinuityStatus.COMPLETE.value

    def __post_init__(self) -> None:
        if self.evaluation_status:
            return
        if self.is_valid is None:
            self.evaluation_status = StepEvaluationStatus.UNVERIFIED.value
        elif self.is_valid:
            self.evaluation_status = StepEvaluationStatus.PASSED.value
        else:
            self.evaluation_status = StepEvaluationStatus.FAILED.value


@dataclass
class GradingReport:
    question: str
    overall_verdict: OverallVerdict
    first_error_step_index: int | None
    total_steps: int
    valid_steps_count: int
    total_score: int
    max_total_score: int
    steps_evaluation: list[StepEvaluation]
    summary_feedback: str
    standard_solution_steps: list[str] = field(default_factory=list)
    ground_truth_facts: dict = field(default_factory=dict)
    score_final: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["overall_verdict"] = self.overall_verdict.value
        return data


def _format_secondary_ocr_evidence(value: str) -> str:
    """Add KaTeX delimiters to bare formula-only OCR evidence fragments."""
    evidence = value.strip()
    if not evidence:
        return ""

    parts = re.split(r"([；;\n]+)", evidence)
    formatted: list[str] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[；;\n]+", part):
            formatted.append(part)
            continue

        fragment = part.strip()
        if not fragment:
            continue
        already_delimited = "\\(" in fragment or "\\[" in fragment
        contains_chinese = bool(re.search(r"[\u3400-\u9fff]", fragment))
        looks_like_formula = bool(
            re.search(r"\\[A-Za-z]+", fragment)
            or re.search(r"(?:=|<=|>=|<|>|\\leq|\\geq)", fragment)
            or re.search(r"\b[A-Za-z]\s*\([^)]*\)", fragment)
            or (
                re.search(r"[+\-*/^]", fragment)
                and re.fullmatch(r"[\sA-Za-z0-9_{}()[\]+\-*/^.,|]+", fragment)
            )
        )
        if looks_like_formula and not already_delimited and not contains_chinese:
            formatted.append(f"\\({fragment}\\)")
        else:
            formatted.append(fragment)
    return "".join(formatted)


def _classify_step(step: StudentStep) -> dict:
    """Convert semantic continuity metadata into a deterministic grading state."""
    allowed_continuity = {item.value for item in ContinuityStatus}
    allowed_validity = {item.value for item in MathematicalValidity}
    continuity = step.continuity_status if step.continuity_status in allowed_continuity else ContinuityStatus.COMPLETE.value
    validity = step.mathematical_validity if step.mathematical_validity in allowed_validity else MathematicalValidity.UNKNOWN.value

    # 旧版模型只有一个布尔字段，保守映射为真正的逻辑断裂。
    if step.has_discontinuity and continuity == ContinuityStatus.COMPLETE.value:
        continuity = ContinuityStatus.LOGICAL_BREAK.value
        validity = MathematicalValidity.INVALID.value

    detail = step.diagnostic_message or step.pedagogical_warning
    if step.ocr_agreement in {"disagree", "uncertain"}:
        evidence_detail = step.verification_message or detail
        if step.secondary_ocr_evidence:
            secondary_evidence = _format_secondary_ocr_evidence(
                step.secondary_ocr_evidence
            )
            evidence_detail = (
                f"{evidence_detail} 复核 OCR 识别为：{secondary_evidence}"
                if evidence_detail
                else f"复核 OCR 识别为：{secondary_evidence}"
            )
        return {
            "continuity": ContinuityStatus.AMBIGUOUS.value,
            "valid": None,
            "score": None,
            "status": StepEvaluationStatus.UNVERIFIED.value,
            "feedback": evidence_detail or "两套 OCR 的数学内容不一致，本步暂时无法确认。",
            "error_category": None,
            "error_detail": evidence_detail or None,
        }
    if continuity == ContinuityStatus.AMBIGUOUS.value:
        return {
            "continuity": continuity,
            "valid": None,
            "score": None,
            "status": StepEvaluationStatus.UNVERIFIED.value,
            "feedback": detail or "OCR 或卷面信息不足，本步暂时无法确认。",
            "error_category": None,
            "error_detail": detail or None,
        }

    if continuity == ContinuityStatus.LOGICAL_BREAK.value or validity == MathematicalValidity.INVALID.value:
        return {
            "continuity": continuity,
            "valid": False,
            "score": 0,
            "status": StepEvaluationStatus.FAILED.value,
            "feedback": detail or "当前结论无法从前面的有效步骤推出。",
            "error_category": ErrorCategory.LOGIC_DISCONTINUITY.value,
            "error_detail": detail or "当前结论与前序有效步骤之间存在数学逻辑断裂。",
        }

    if validity == MathematicalValidity.UNKNOWN.value:
        return {
            "continuity": continuity,
            "valid": None,
            "score": None,
            "status": StepEvaluationStatus.UNVERIFIED.value,
            "feedback": detail or "当前数学内容的信息不足，本步暂时无法确认。",
            "error_category": None,
            "error_detail": detail or None,
        }

    if continuity == ContinuityStatus.ACCEPTABLE_OMISSION.value:
        omitted = step.omitted_reasoning.strip()
        feedback = detail or "省略了常规计算过程，但结论能够由前面的有效步骤正确推出。"
        if omitted:
            feedback = f"{feedback} 可补写：{omitted}"
        return {
            "continuity": continuity,
            "valid": True,
            "score": 2,
            "status": StepEvaluationStatus.PASSED_WITH_NOTE.value,
            "feedback": feedback,
            "error_category": None,
            "error_detail": None,
        }

    return {
        "continuity": continuity,
        "valid": True,
        "score": 2,
        "status": StepEvaluationStatus.PASSED.value,
        "feedback": detail or "步骤推导正确。",
        "error_category": None,
        "error_detail": None,
    }


def parse_raw_steps_fallback(raw_text: str) -> list[StudentStep]:
    """Offline rule-based fallback parser for testing without LLM API."""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    steps = []

    for i, line in enumerate(lines, start=1):
        marker = "none"
        if line.startswith(("∵", "\\because", "因为")):
            marker = "because"
        elif line.startswith(("∴", "\\therefore", "所以", "故", "则")):
            marker = "therefore"

        expr_sympy = ""
        latex_str = ""

        eq_match = re.search(r"=\s*([0-9xX+\-*/().\s^]+)", line)
        if eq_match:
            candidate = eq_match.group(1).replace("^", "**").strip()
            try:
                sp.sympify(candidate)
                expr_sympy = candidate
                latex_str = line
            except Exception:
                pass

        claimed_axis = ""
        axis_m = re.search(r"对称轴.*?x\s*=\s*(-?\d+(?:/\d+)?)", line)
        if axis_m:
            claimed_axis = axis_m.group(1)

        claimed_ext_kind = "none"
        claimed_ext_val = ""
        if "最大值" in line:
            claimed_ext_kind = "max"
            val_m = re.search(r"最大值[为是\s]*(-?\d+(?:/\d+)?)", line)
            if val_m:
                claimed_ext_val = val_m.group(1)
        elif "最小值" in line:
            claimed_ext_kind = "min"
            val_m = re.search(r"最小值[为是\s]*(-?\d+(?:/\d+)?)", line)
            if val_m:
                claimed_ext_val = val_m.group(1)

        step_intent = "other"
        if "配方" in line or "(" in expr_sympy and "**2" in expr_sympy:
            step_intent = "completing_square"
        elif claimed_axis:
            step_intent = "determine_axis"
        elif claimed_ext_kind != "none":
            step_intent = "determine_extremum"

        steps.append(
            StudentStep(
                step_index=i,
                raw_text=line,
                marker=marker,
                expression_latex=latex_str or line,
                expression_sympy=expr_sympy,
                step_intent=step_intent,
                claimed_axis=claimed_axis,
                claimed_extremum_kind=claimed_ext_kind,
                claimed_extremum_value=claimed_ext_val,
            )
        )
    return steps


def structure_student_steps(
    question_text: str,
    raw_student_text: str,
    client: DeepSeekClient | None = None,
) -> list[StudentStep]:
    """Structure raw OCR text into normalized StudentStep objects."""
    if client is not None:
        try:
            res = client.normalize_ocr_math_steps(
                f"【题目】\n{question_text}\n\n【学生作答】\n{raw_student_text}"
            )
            steps_data = res.get("steps", [])
            steps = []
            for item in steps_data:
                steps.append(
                    StudentStep(
                        step_index=item.get("step_number", len(steps) + 1),
                        raw_text=item.get("step_text", ""),
                        marker=item.get("marker", "none"),
                        expression_latex=item.get("step_text", ""),
                        step_intent=item.get("step_intent", "other"),
                        continuity_status=item.get("continuity_status", ContinuityStatus.COMPLETE.value),
                        mathematical_validity=item.get("mathematical_validity", MathematicalValidity.VALID.value),
                        omitted_reasoning=item.get("omitted_reasoning", ""),
                        diagnostic_message=item.get("diagnostic_message", ""),
                        ocr_agreement=item.get("ocr_agreement", "not_checked"),
                        secondary_ocr_evidence=item.get("secondary_ocr_evidence", ""),
                        verification_message=item.get("verification_message", ""),
                        has_discontinuity=bool(item.get("has_discontinuity", False)),
                        pedagogical_warning=item.get("pedagogical_warning", ""),
                    )
                )
            if steps:
                return steps
        except Exception:
            pass

    return parse_raw_steps_fallback(raw_student_text)


def grade_solution(
    problem: Problem,
    student_steps: list[StudentStep],
    standard_facts: dict | None = None,
) -> GradingReport:
    """Validate student steps using SymPy algebraic checks and ground truth facts."""
    x = sp.Symbol("x")
    target_expr = sp.sympify(problem.expression_sympy)

    # 提取标准二次函数参数与事实
    try:
        poly = sp.Poly(target_expr, x)
        coeffs = poly.all_coeffs()
        a_val, b_val, c_val = coeffs[0], coeffs[1], coeffs[2] if len(coeffs) > 2 else 0
        std_axis = sp.Rational(-b_val, 2 * a_val)
        std_extremum_val = target_expr.subs(x, std_axis)
        std_extremum_kind = "max" if a_val < 0 else "min"
        computed_facts = {
            "a": str(a_val),
            "b": str(b_val),
            "c": str(c_val),
            "axis": str(std_axis),
            "extremum_kind": std_extremum_kind,
            "extremum_value": str(std_extremum_val),
        }
    except Exception:
        a_val, b_val, c_val = 1, 0, 0
        std_axis = 0
        std_extremum_val = 0
        std_extremum_kind = "min"
        computed_facts = {}

    if standard_facts:
        computed_facts.update(standard_facts)

    step_evals: list[StepEvaluation] = []
    first_error_index: int | None = None

    for step in student_steps:
        classification = _classify_step(step)
        if classification["valid"] is None:
            step_evals.append(
                StepEvaluation(
                    step_index=step.step_index,
                    raw_text=step.raw_text,
                    marker=step.marker,
                    is_valid=None,
                    step_score=None,
                    max_score=2,
                    feedback=classification["feedback"],
                    error_category=classification["error_category"],
                    error_detail=classification["error_detail"],
                    evaluation_status=classification["status"],
                    continuity_status=classification["continuity"],
                )
            )
            continue
        if classification["valid"] is False:
            if first_error_index is None:
                first_error_index = step.step_index
            step_evals.append(
                StepEvaluation(
                    step_index=step.step_index,
                    raw_text=step.raw_text,
                    marker=step.marker,
                    is_valid=False,
                    step_score=0,
                    max_score=2,
                    feedback=classification["feedback"],
                    error_category=classification["error_category"],
                    error_detail=classification["error_detail"],
                    evaluation_status=classification["status"],
                    continuity_status=classification["continuity"],
                )
            )
            continue

        is_valid = True
        err_cat: str | None = None
        err_det: str | None = None
        proof_str: str | None = None
        feedback = classification["feedback"]
        score = 2
        evaluation_status = classification["status"]

        # 1. 验证代数恒等变形（如配方步骤）
        if step.expression_sympy and step.step_intent in {
            "completing_square",
            "factor_coefficient",
            "other",
        }:
            try:
                step_expr = sp.sympify(step.expression_sympy)
                diff = sp.simplify(step_expr - target_expr)
                if diff == 0:
                    is_valid = True
                    proof_str = "sp.simplify(step_expr - target_expr) == 0 (恒等)"
                    if evaluation_status == StepEvaluationStatus.PASSED.value:
                        feedback = "代数变形完全正确。"
                else:
                    is_valid = False
                    score = 0
                    evaluation_status = StepEvaluationStatus.FAILED.value
                    err_cat = ErrorCategory.CALCULATION_ERROR.value
                    err_det = f"代数变形与原式不恒等（差值为 {sp.latex(diff)}），计算存在错误。"
                    feedback = "配方计算有误，请检查常数项或系数展开。"
            except Exception:
                pass

        # 2. 验证对称轴断言
        if is_valid and step.claimed_axis:
            try:
                claimed_ax_sp = sp.sympify(step.claimed_axis)
                if sp.simplify(claimed_ax_sp - std_axis) == 0:
                    is_valid = True
                    if evaluation_status == StepEvaluationStatus.PASSED.value:
                        feedback = f"对称轴 x = {std_axis} 判定正确。"
                else:
                    is_valid = False
                    score = 0
                    evaluation_status = StepEvaluationStatus.FAILED.value
                    err_cat = (
                        ErrorCategory.SIGN_ERROR.value
                        if sp.simplify(claimed_ax_sp + std_axis) == 0
                        else ErrorCategory.CALCULATION_ERROR.value
                    )
                    err_det = f"对称轴应为 x = {std_axis}，当前写为 x = {step.claimed_axis}。"
                    feedback = f"对称轴计算错误，正确对称轴为 x = {std_axis}。"
            except Exception:
                pass

        # 3. 验证最值类型与数值
        if is_valid and step.claimed_extremum_kind != "none":
            if step.claimed_extremum_kind != std_extremum_kind:
                is_valid = False
                score = 0
                evaluation_status = StepEvaluationStatus.FAILED.value
                err_cat = ErrorCategory.CONCEPT_ERROR.value
                kind_zh = "最大值" if std_extremum_kind == "max" else "最小值"
                err_kind_zh = "最小值" if std_extremum_kind == "max" else "最大值"
                err_det = f"二次项系数 a = {a_val} {'< 0 开口向下' if a_val < 0 else '> 0 开口向上'}，在顶点处应取得【{kind_zh}】而非【{err_kind_zh}】。"
                feedback = f"最值类型混淆：抛物线开口{'向下应取最大值' if a_val < 0 else '向上应取最小值'}。"
            elif step.claimed_extremum_value:
                try:
                    claimed_val_sp = sp.sympify(step.claimed_extremum_value)
                    if sp.simplify(claimed_val_sp - std_extremum_val) != 0:
                        is_valid = False
                        score = 0
                        evaluation_status = StepEvaluationStatus.FAILED.value
                        err_cat = ErrorCategory.CALCULATION_ERROR.value
                        err_det = f"最值数值应为 {std_extremum_val}，当前写为 {step.claimed_extremum_value}。"
                        feedback = f"最值数值计算错误，正确最值为 {std_extremum_val}。"
                    else:
                        if evaluation_status == StepEvaluationStatus.PASSED.value:
                            feedback = f"最值类型与数值完全正确（在顶点取得 { '最大值' if std_extremum_kind == 'max' else '最小值' } {std_extremum_val}）。"
                except Exception:
                    pass

        if not is_valid:
            if first_error_index is None:
                first_error_index = step.step_index

        step_evals.append(
            StepEvaluation(
                step_index=step.step_index,
                raw_text=step.raw_text,
                marker=step.marker,
                is_valid=is_valid,
                step_score=score,
                max_score=2,
                feedback=feedback,
                error_category=err_cat,
                error_detail=err_det,
                sympy_proof=proof_str,
                evaluation_status=evaluation_status,
                continuity_status=classification["continuity"],
            )
        )

    valid_count = sum(1 for e in step_evals if e.is_valid)
    total_count = len(step_evals)
    total_score = sum(e.step_score or 0 for e in step_evals)
    max_total_score = sum(e.max_score for e in step_evals) or 10
    has_unverified = any(e.is_valid is None for e in step_evals)

    if has_unverified:
        verdict = OverallVerdict.NEEDS_REVIEW
    elif first_error_index is None and valid_count > 0:
        verdict = OverallVerdict.CORRECT
    elif valid_count > 0:
        verdict = OverallVerdict.PARTIALLY_CORRECT
    else:
        verdict = OverallVerdict.INCORRECT

    summary_fb = generate_pedagogical_summary(verdict, first_error_index, step_evals, computed_facts)

    return GradingReport(
        question=problem.question_text,
        overall_verdict=verdict,
        first_error_step_index=first_error_index,
        total_steps=total_count,
        valid_steps_count=valid_count,
        total_score=total_score,
        max_total_score=max_total_score,
        steps_evaluation=step_evals,
        summary_feedback=summary_fb,
        ground_truth_facts=computed_facts,
        score_final=not has_unverified,
    )


def grade_normalized_steps(
    question_stem: str,
    steps: list[dict],
    client: DeepSeekClient | None = None,
) -> dict:
    """Grade a sequence of normalized steps against standard mathematical logic."""
    # 转换为 StudentStep
    student_steps: list[StudentStep] = []
    for s in steps:
        student_steps.append(
            StudentStep(
                step_index=s.get("step_number", len(student_steps) + 1),
                raw_text=s.get("step_text", ""),
                marker=s.get("marker", "none"),
                expression_latex=s.get("step_text", ""),
                step_intent=s.get("step_intent", "other"),
                continuity_status=s.get("continuity_status", ContinuityStatus.COMPLETE.value),
                mathematical_validity=s.get("mathematical_validity", MathematicalValidity.VALID.value),
                omitted_reasoning=s.get("omitted_reasoning", ""),
                diagnostic_message=s.get("diagnostic_message", ""),
                ocr_agreement=s.get("ocr_agreement", "not_checked"),
                secondary_ocr_evidence=s.get("secondary_ocr_evidence", ""),
                verification_message=s.get("verification_message", ""),
                has_discontinuity=bool(s.get("has_discontinuity", False)),
                pedagogical_warning=s.get("pedagogical_warning", ""),
            )
        )

    # 1. 检测待定系数法求解析式/奇函数题型
    if "奇函数" in question_stem and "解析式" in question_stem:
        report = _grade_odd_function_coefficients(question_stem, student_steps)
        return _attach_ocr_evidence(report, steps)

    # 2. 如果包含二次函数表达式
    quad_m = re.search(r"y\s*=\s*([-0-9xX+\-*/().\s^]+)", question_stem)
    if quad_m:
        expr_str = quad_m.group(1).replace("^", "**")
        try:
            problem = Problem(question_text=question_stem, expression_sympy=expr_str)
            report = grade_solution(problem, student_steps)
            return _attach_ocr_evidence(report.to_dict(), steps)
        except Exception:
            pass

    # 3. 通用高中数学题目综合判定
    return _attach_ocr_evidence(
        _grade_general_math_steps(question_stem, student_steps),
        steps,
    )


def _attach_ocr_evidence(report: dict, source_steps: list[dict]) -> dict:
    """Carry dual-OCR evidence into the final report and PostgreSQL step rows."""
    evidence_by_step = {
        int(step.get("step_number", index)): step
        for index, step in enumerate(source_steps, start=1)
    }
    for evaluation in report.get("steps_evaluation", []):
        source = evidence_by_step.get(int(evaluation.get("step_index", 0)), {})
        evaluation["ocr_agreement"] = source.get("ocr_agreement", "not_checked")
        evaluation["secondary_ocr_evidence"] = source.get("secondary_ocr_evidence", "")
        evaluation["verification_message"] = source.get("verification_message", "")
        evaluation["ocr_fix_suggestion"] = source.get("ocr_fix_suggestion", "")
    return report


def _grade_odd_function_coefficients(question: str, steps: list[StudentStep]) -> dict:
    """Deterministic grading for: f(x) = (ax+b)/(1+x^2) is odd on (-1,1), f(1/2)=2/5."""
    step_evals: list[StepEvaluation] = []
    first_error_idx = None
    b_solved = False
    a_solved = False
    fx_solved = False

    for s in steps:
        classification = _classify_step(s)
        if classification["valid"] is None:
            step_evals.append(
                StepEvaluation(
                    step_index=s.step_index,
                    raw_text=s.raw_text,
                    marker=s.marker,
                    is_valid=None,
                    step_score=None,
                    max_score=2,
                    feedback=classification["feedback"],
                    error_category=classification["error_category"],
                    error_detail=classification["error_detail"],
                    evaluation_status=classification["status"],
                    continuity_status=classification["continuity"],
                )
            )
            continue
        if classification["valid"] is False:
            if first_error_idx is None:
                first_error_idx = s.step_index
            step_evals.append(
                StepEvaluation(
                    step_index=s.step_index,
                    raw_text=s.raw_text,
                    marker=s.marker,
                    is_valid=False,
                    step_score=0,
                    max_score=2,
                    feedback=classification["feedback"],
                    error_category=classification["error_category"],
                    error_detail=classification["error_detail"],
                    evaluation_status=classification["status"],
                    continuity_status=classification["continuity"],
                )
            )
            continue

        is_valid = True
        score = 2
        err_cat = None
        err_det = None
        fb = classification["feedback"]
        evaluation_status = classification["status"]

        if "f(0) = 0" in s.raw_text or "b = 0" in s.raw_text or "b=0" in s.raw_text:
            b_solved = True
            if evaluation_status == StepEvaluationStatus.PASSED.value:
                fb = "运用奇函数在原点有定义则 f(0)=0，成功推导出 b=0。"
        elif "1/2" in s.raw_text or "f\\left(\\frac{1}{2}\\right)" in s.raw_text or "f(1/2)" in s.raw_text:
            if evaluation_status == StepEvaluationStatus.PASSED.value:
                fb = "代入 f(1/2)=2/5 建立关于 a 的代数方程，代数结构准确。"
        elif "a = 1" in s.raw_text or "a=1" in s.raw_text:
            a_solved = True
            if evaluation_status == StepEvaluationStatus.PASSED.value:
                fb = "方程求解准确，正确求得未知数 a=1。"
        elif ("f(x)" in s.raw_text or "解析式" in s.raw_text) and ("1+x^2" in s.raw_text or "1 + x^2" in s.raw_text or "x^2" in s.raw_text):
            fx_solved = True
            if evaluation_status == StepEvaluationStatus.PASSED.value:
                fb = "最终解析式正确，代回检验符合定义域与奇函数性质。"

        step_evals.append(
            StepEvaluation(
                step_index=s.step_index,
                raw_text=s.raw_text,
                marker=s.marker,
                is_valid=is_valid,
                step_score=score,
                max_score=2,
                feedback=fb,
                error_category=err_cat,
                error_detail=err_det,
                evaluation_status=evaluation_status,
                continuity_status=classification["continuity"],
            )
        )

    valid_count = sum(1 for e in step_evals if e.is_valid)
    total_count = len(step_evals)
    total_score = sum(e.step_score or 0 for e in step_evals)
    max_total_score = total_count * 2
    has_unverified = any(e.is_valid is None for e in step_evals)

    if has_unverified:
        verdict = OverallVerdict.NEEDS_REVIEW
    elif first_error_idx is None and b_solved and a_solved and fx_solved:
        verdict = OverallVerdict.CORRECT
    elif valid_count > 0:
        verdict = OverallVerdict.PARTIALLY_CORRECT
    else:
        verdict = OverallVerdict.INCORRECT

    std_steps = [
        r"∵ $f(x)$ 是定义在 $(-1, 1)$ 上的奇函数，且 $0 \in (-1, 1)$，∴ $f(0) = 0$。",
        r"即 $\frac{b}{1 + 0^2} = 0$，解得 $b = 0$。",
        r"又 $f\left(\frac{1}{2}\right) = \frac{\frac{1}{2}a}{1 + \left(\frac{1}{2}\right)^2} = \frac{2}{5}$，解得 $a = 1$。",
        r"∴ 函数 $f(x)$ 的解析式为 $f(x) = \frac{x}{1 + x^2}$（$x \in (-1, 1)$）。",
    ]

    summary_fb = generate_pedagogical_summary(verdict, first_error_idx, step_evals, {"b": "0", "a": "1", "fx": "x/(1+x^2)"})

    return GradingReport(
        question=question,
        overall_verdict=verdict,
        first_error_step_index=first_error_idx,
        total_steps=total_count,
        valid_steps_count=valid_count,
        total_score=total_score,
        max_total_score=max_total_score,
        steps_evaluation=step_evals,
        summary_feedback=summary_fb,
        standard_solution_steps=std_steps,
        ground_truth_facts={"b": "0", "a": "1", "fx": "x/(1+x^2)"},
        score_final=not has_unverified,
    ).to_dict()


def _grade_general_math_steps(question: str, steps: list[StudentStep]) -> dict:
    """General fallback grader evaluating step integrity and continuity."""
    step_evals: list[StepEvaluation] = []
    first_error_idx = None

    for s in steps:
        classification = _classify_step(s)
        if classification["valid"] is False and first_error_idx is None:
            first_error_idx = s.step_index

        step_evals.append(
            StepEvaluation(
                step_index=s.step_index,
                raw_text=s.raw_text,
                marker=s.marker,
                is_valid=classification["valid"],
                step_score=classification["score"],
                max_score=2,
                feedback=classification["feedback"],
                error_category=classification["error_category"],
                error_detail=classification["error_detail"],
                evaluation_status=classification["status"],
                continuity_status=classification["continuity"],
            )
        )

    valid_count = sum(1 for e in step_evals if e.is_valid)
    total_count = len(step_evals)
    total_score = sum(e.step_score or 0 for e in step_evals)
    max_total_score = total_count * 2
    has_unverified = any(e.is_valid is None for e in step_evals)

    if has_unverified:
        verdict = OverallVerdict.NEEDS_REVIEW
    elif first_error_idx is None and valid_count > 0:
        verdict = OverallVerdict.CORRECT
    elif valid_count > 0:
        verdict = OverallVerdict.PARTIALLY_CORRECT
    else:
        verdict = OverallVerdict.INCORRECT

    summary_fb = generate_pedagogical_summary(verdict, first_error_idx, step_evals, {})

    return GradingReport(
        question=question,
        overall_verdict=verdict,
        first_error_step_index=first_error_idx,
        total_steps=total_count,
        valid_steps_count=valid_count,
        total_score=total_score,
        max_total_score=max_total_score,
        steps_evaluation=step_evals,
        summary_feedback=summary_fb,
        score_final=not has_unverified,
    ).to_dict()


def generate_pedagogical_summary(
    verdict: OverallVerdict,
    first_error_idx: int | None,
    evals: list[StepEvaluation],
    facts: dict,
) -> str:
    """Generate encouraging, diagnostic educational summary."""
    advisory_steps = [
        evaluation.step_index
        for evaluation in evals
        if evaluation.evaluation_status == StepEvaluationStatus.PASSED_WITH_NOTE.value
    ]
    if verdict == OverallVerdict.CORRECT:
        if advisory_steps:
            numbers = "、".join(str(index) for index in advisory_steps)
            return f"答案与推导结论正确。第 {numbers} 步省略了可合理补全的常规运算，不影响得分；如需完整展示过程，可以补写中间变形。"
        return "🎉 恭喜！你的整套解题推导逻辑严密、未知数求解精准、格式与符号书写极其规范，获得满分！"

    if verdict == OverallVerdict.NEEDS_REVIEW:
        uncertain_steps = [str(e.step_index) for e in evals if e.is_valid is None]
        return f"第 {'、'.join(uncertain_steps)} 步受 OCR 或卷面信息影响，当前无法可靠确认，因此暂不形成最终评分。"

    first_err_eval = next((e for e in evals if e.step_index == first_error_idx), None)
    if not first_err_eval:
        return "解题过程存在部分失分点，请参考标准参考解答仔细核对。"

    correct_steps = [e.step_index for e in evals if e.is_valid]
    praise = ""
    if correct_steps:
        praise = f"👏 亮点：第 {', '.join(str(i) for i in correct_steps)} 步的推导和代数运算是完全正确的。"

    critique = f"第 {first_error_idx} 步出现失分点：{first_err_eval.error_detail or first_err_eval.feedback}"

    suggestion = ""
    if first_err_eval.error_category == ErrorCategory.CONCEPT_ERROR.value:
        suggestion = "💡 建议：牢记核心定理与最值法则，注意开口方向与极值性质的对应关系。"
    elif first_err_eval.error_category == ErrorCategory.OMISSION_ERROR.value:
        suggestion = "💡 建议：答题时请务必完整书写未知数与中间等式，避免心算跳步导致在高考中丢失步骤分。"
    elif first_err_eval.error_category == ErrorCategory.CALCULATION_ERROR.value:
        suggestion = "💡 建议：代数分式化简时注意分子与分母的对应展开，避免符号与系数错误。"
    elif first_err_eval.error_category == ErrorCategory.SIGN_ERROR.value:
        suggestion = "💡 建议：注意负号法则与对称轴公式中的符号取值。"
    elif first_err_eval.error_category == ErrorCategory.LOGIC_DISCONTINUITY.value:
        suggestion = "💡 建议：检查当前结论能否由前面的有效等式或条件推出，并补正错误的运算依据。"

    return f"{praise}\n\n⚠️ {critique}\n\n{suggestion}".strip()
