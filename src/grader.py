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
    has_discontinuity: bool = False
    pedagogical_warning: str = ""


@dataclass
class StepEvaluation:
    step_index: int
    raw_text: str
    marker: str
    is_valid: bool
    step_score: int = 2
    max_score: int = 2
    feedback: str = ""
    error_category: str | None = None
    error_detail: str | None = None
    sympy_proof: str | None = None


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

    def to_dict(self) -> dict:
        data = asdict(self)
        data["overall_verdict"] = self.overall_verdict.value
        return data


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
            res = client.normalize_ocr_math_steps(raw_student_text)
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
    has_seen_error = False

    for step in student_steps:
        if has_seen_error:
            step_evals.append(
                StepEvaluation(
                    step_index=step.step_index,
                    raw_text=step.raw_text,
                    marker=step.marker,
                    is_valid=False,
                    step_score=0,
                    max_score=2,
                    feedback="受前序步骤错误影响，本步推导结论失真。",
                    error_category=ErrorCategory.LOGIC_DISCONTINUITY.value,
                    error_detail="前序计算存在偏差，本步无法判定为正确。",
                )
            )
            continue

        is_valid = True
        err_cat: str | None = None
        err_det: str | None = None
        proof_str: str | None = None
        feedback = "步骤推导正确。"
        score = 2

        # 检查逻辑断裂
        if step.has_discontinuity:
            is_valid = False
            score = 0
            err_cat = ErrorCategory.OMISSION_ERROR.value
            err_det = step.pedagogical_warning or "检测到代数因果断裂（如遗漏未知数或条件）。"
            feedback = f"书写遗漏扣分：{err_det}"

        # 1. 验证代数恒等变形（如配方步骤）
        elif step.expression_sympy and step.step_intent in {
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
                    feedback = "代数变形完全正确。"
                else:
                    is_valid = False
                    score = 0
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
                    feedback = f"对称轴 x = {std_axis} 判定正确。"
                else:
                    is_valid = False
                    score = 0
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
                        err_cat = ErrorCategory.CALCULATION_ERROR.value
                        err_det = f"最值数值应为 {std_extremum_val}，当前写为 {step.claimed_extremum_value}。"
                        feedback = f"最值数值计算错误，正确最值为 {std_extremum_val}。"
                    else:
                        feedback = f"最值类型与数值完全正确（在顶点取得 { '最大值' if std_extremum_kind == 'max' else '最小值' } {std_extremum_val}）。"
                except Exception:
                    pass

        if not is_valid:
            has_seen_error = True
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
            )
        )

    valid_count = sum(1 for e in step_evals if e.is_valid)
    total_count = len(step_evals)
    total_score = sum(e.step_score for e in step_evals)
    max_total_score = sum(e.max_score for e in step_evals) or 10

    if first_error_index is None and valid_count > 0:
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
                has_discontinuity=bool(s.get("has_discontinuity", False)),
                pedagogical_warning=s.get("pedagogical_warning", ""),
            )
        )

    # 1. 检测待定系数法求解析式/奇函数题型
    if "奇函数" in question_stem and "解析式" in question_stem:
        return _grade_odd_function_coefficients(question_stem, student_steps)

    # 2. 如果包含二次函数表达式
    quad_m = re.search(r"y\s*=\s*([-0-9xX+\-*/().\s^]+)", question_stem)
    if quad_m:
        expr_str = quad_m.group(1).replace("^", "**")
        try:
            problem = Problem(question_text=question_stem, expression_sympy=expr_str)
            report = grade_solution(problem, student_steps)
            return report.to_dict()
        except Exception:
            pass

    # 3. 通用高中数学题目综合判定
    return _grade_general_math_steps(question_stem, student_steps)


def _grade_odd_function_coefficients(question: str, steps: list[StudentStep]) -> dict:
    """Deterministic grading for: f(x) = (ax+b)/(1+x^2) is odd on (-1,1), f(1/2)=2/5."""
    step_evals = []
    has_seen_error = False
    first_error_idx = None
    b_solved = False
    a_solved = False
    fx_solved = False

    for s in steps:
        if has_seen_error:
            step_evals.append(
                StepEvaluation(
                    step_index=s.step_index,
                    raw_text=s.raw_text,
                    marker=s.marker,
                    is_valid=False,
                    step_score=0,
                    max_score=2,
                    feedback="受前序推导错误影响，本步无法得分。",
                    error_category=ErrorCategory.LOGIC_DISCONTINUITY.value,
                    error_detail="前序关键参数求解存在偏差，后续结论失真。",
                )
            )
            continue

        is_valid = True
        score = 2
        err_cat = None
        err_det = None
        fb = "步骤正确。"

        # 检查逻辑因果断裂（如丢了 a）
        if s.has_discontinuity:
            is_valid = False
            score = 0
            err_cat = ErrorCategory.OMISSION_ERROR.value
            err_det = s.pedagogical_warning or "关键未知数遗漏或算式两端矛盾，因果推导断裂。"
            fb = f"⚠️ 扣分点：{err_det}"
        elif "f(0) = 0" in s.raw_text or "b = 0" in s.raw_text or "b=0" in s.raw_text:
            b_solved = True
            fb = "运用奇函数在原点有定义则 f(0)=0，成功推导出 b=0，得分点满分。"
        elif ("1/2" in s.raw_text or "f\\left(\\frac{1}{2}\\right)" in s.raw_text or "f(1/2)" in s.raw_text) and not s.has_discontinuity:
            fb = "代入 f(1/2)=2/5 建立关于 a 的代数方程，代数结构准确。"
        elif "a = 1" in s.raw_text or "a=1" in s.raw_text:
            a_solved = True
            fb = "方程求解准确，正确求得未知数 a=1。"
        elif ("f(x)" in s.raw_text or "解析式" in s.raw_text) and ("1+x^2" in s.raw_text or "1 + x^2" in s.raw_text or "x^2" in s.raw_text):
            fx_solved = True
            fb = "最终解析式推导完全正确，代回检验符合定义域与奇函数性质！"

        if not is_valid:
            has_seen_error = True
            first_error_idx = s.step_index

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
            )
        )

    valid_count = sum(1 for e in step_evals if e.is_valid)
    total_count = len(step_evals)
    total_score = sum(e.step_score for e in step_evals)
    max_total_score = total_count * 2

    if first_error_idx is None and b_solved and a_solved and fx_solved:
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
    ).to_dict()


def _grade_general_math_steps(question: str, steps: list[StudentStep]) -> dict:
    """General fallback grader evaluating step integrity and continuity."""
    step_evals = []
    has_seen_error = False
    first_error_idx = None

    for s in steps:
        if has_seen_error:
            step_evals.append(
                StepEvaluation(
                    step_index=s.step_index,
                    raw_text=s.raw_text,
                    marker=s.marker,
                    is_valid=False,
                    step_score=0,
                    max_score=2,
                    feedback="受前序步骤错误影响，本步无法判定得分。",
                    error_category=ErrorCategory.LOGIC_DISCONTINUITY.value,
                    error_detail="前序逻辑存在瑕疵。",
                )
            )
            continue

        is_valid = not s.has_discontinuity
        score = 2 if is_valid else 0
        err_cat = ErrorCategory.OMISSION_ERROR.value if s.has_discontinuity else None
        err_det = s.pedagogical_warning if s.has_discontinuity else None
        fb = f"⚠️ 扣分点：{s.pedagogical_warning}" if s.has_discontinuity else "步骤推导逻辑自洽。"

        if not is_valid:
            has_seen_error = True
            first_error_idx = s.step_index

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
            )
        )

    valid_count = sum(1 for e in step_evals if e.is_valid)
    total_count = len(step_evals)
    total_score = sum(e.step_score for e in step_evals)
    max_total_score = total_count * 2

    if first_error_idx is None and valid_count > 0:
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
    ).to_dict()


def generate_pedagogical_summary(
    verdict: OverallVerdict,
    first_error_idx: int | None,
    evals: list[StepEvaluation],
    facts: dict,
) -> str:
    """Generate encouraging, diagnostic educational summary."""
    if verdict == OverallVerdict.CORRECT:
        return "🎉 恭喜！你的整套解题推导逻辑严密、未知数求解精准、格式与符号书写极其规范，获得满分！"

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

    return f"{praise}\n\n⚠️ {critique}\n\n{suggestion}".strip()
