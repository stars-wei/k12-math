"""Minimal DeepSeek JSON client. It never knows Neo4j or SymPy details."""

# from __future__ import annotations

import json
import os
import re
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from errors import ConfigurationError, UpstreamServiceError


# DeepSeek strict function calling is currently exposed through the beta URL.
API_URL = "https://api.deepseek.com/beta/chat/completions"
SAFE_EXPRESSION = re.compile(r"^[0-9xX+\-*/().\s]+$")


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        trace_callback: Callable[[str, dict, int | None], None] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.trace_callback = trace_callback
        if not self.api_key:
            raise ConfigurationError("未设置 DEEPSEEK_API_KEY，无法进行题目解析。")

    def _emit_trace(self, stage: str, payload: dict, duration_ms: int | None = None) -> None:
        """Report model diagnostics without allowing tracing to break grading."""
        if self.trace_callback is None:
            return
        try:
            self.trace_callback(stage, payload, duration_ms)
        except Exception:
            return

    def _tool_call(
        self,
        tool_name: str,
        description: str,
        parameters: dict,
        system: str,
        user: str,
    ) -> dict:
        """Make one strict function call and return its JSON arguments."""
        body = {
            "model": self.model,
            # DeepSeek's thinking mode rejects forced tool_choice. This
            # classifier relies on a strict, dynamically generated enum.
            "thinking": {"type": "disabled"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description,
                        "strict": True,
                        "parameters": parameters,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": tool_name},
            },
        }
        trace_payload = {
            "tool_name": tool_name,
            "model": self.model,
            "description": description,
            "parameters": parameters,
            "system": system,
            "user": user,
        }
        self._emit_trace("deepseek_request", trace_payload)
        request = Request(
            API_URL,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started_at = time.monotonic()
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except HTTPError as error:
            self._emit_trace(
                "deepseek_error",
                {"tool_name": tool_name, "model": self.model, "http_status": error.code},
                round((time.monotonic() - started_at) * 1000),
            )
            raise UpstreamServiceError(
                f"DeepSeek 题目解析服务请求失败（HTTP {error.code}），请稍后重试。"
            ) from error
        except URLError as error:
            self._emit_trace(
                "deepseek_error",
                {"tool_name": tool_name, "model": self.model, "error": str(error.reason)},
                round((time.monotonic() - started_at) * 1000),
            )
            raise UpstreamServiceError("无法连接 DeepSeek 题目解析服务，请检查网络后重试。") from error

        self._emit_trace(
            "deepseek_response",
            {"tool_name": tool_name, "model": self.model, "response": payload},
            round((time.monotonic() - started_at) * 1000),
        )

        try:
            tool_calls = payload["choices"][0]["message"]["tool_calls"]
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                raise KeyError("unexpected tool call count")
            tool_call = tool_calls[0]
            if tool_call["function"]["name"] != tool_name:
                raise KeyError("unexpected function")
            raw_args = tool_call["function"]["arguments"]
            result = self._safe_decode_json(raw_args)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise UpstreamServiceError("DeepSeek 未返回可用的题目解析结果，请稍后重试。") from error
        if not isinstance(result, dict):
            raise UpstreamServiceError("DeepSeek 返回的题目解析格式无效，请稍后重试。")
        return result

    @staticmethod
    def _safe_decode_json(s: str) -> dict:
        """Robust JSON decoder that preserves LaTeX backslashes without JSON control char corruption."""
        out = []
        in_string = False
        escape = False
        i = 0
        n = len(s)
        while i < n:
            char = s[i]
            if in_string:
                if escape:
                    if char == '"':
                        out.append('\\"')
                    elif char == '\\':
                        out.append('\\\\')
                    elif char == '/':
                        out.append('/')
                    elif char == 'n':
                        next_char = s[i + 1] if i + 1 < n else ''
                        if next_char and next_char.isascii() and next_char.isalpha():
                            # Preserve LaTeX commands such as \neq, \not and \nabla.
                            out.append('\\\\n')
                        else:
                            # JSON newlines may be followed by another escape, punctuation,
                            # a subquestion label, or non-ASCII text.
                            out.append('\\n')
                    else:
                        # In LaTeX strings, \f in \frac, \r in \right, \b in \because, \t in \theta etc. must remain LaTeX backslashes!
                        out.append('\\\\' + char)
                    escape = False
                else:
                    if char == '\\':
                        escape = True
                    elif char == '"':
                        in_string = False
                        out.append('"')
                    else:
                        out.append(char)
            else:
                if char == '"':
                    in_string = True
                out.append(char)
            i += 1
        if escape:
            out.append('\\\\')
        return json.loads("".join(out))

    @staticmethod
    def _validate_extracted_problems(result: dict) -> dict:
        """Validate model output before any field reaches the web workflow."""
        items = result.get("items")
        if not isinstance(items, list) or not items:
            raise UpstreamServiceError("DeepSeek 未提取出待求小题，请检查题干后重试。")

        expected_fields = {
            "label",
            "target_expression",
            "reference_expressions",
            "question_text",
        }
        for item in items:
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise UpstreamServiceError("DeepSeek 返回的小题字段无效，请稍后重试。")
            if not isinstance(item["label"], str):
                raise UpstreamServiceError("DeepSeek 返回的小题编号无效，请稍后重试。")

            target = item["target_expression"]
            if (
                not isinstance(target, str)
                or not target.strip()
                or SAFE_EXPRESSION.fullmatch(target) is None
            ):
                raise UpstreamServiceError("DeepSeek 返回的待求函数格式无效，请稍后重试。")

            references = item["reference_expressions"]
            if not isinstance(references, list) or any(
                not isinstance(reference, str)
                or not reference.strip()
                or SAFE_EXPRESSION.fullmatch(reference) is None
                for reference in references
            ):
                raise UpstreamServiceError("DeepSeek 返回的参照函数格式无效，请稍后重试。")

            question_text = item["question_text"]
            if not isinstance(question_text, str) or not question_text.strip():
                raise UpstreamServiceError("DeepSeek 返回的小题要求无效，请稍后重试。")
        return result

    def extract_problems(self, question: str) -> dict:
        """Extract every independently solvable function expression in the question."""
        result = self._tool_call(
            "extract_problem_items",
            "Extract explicitly requested problem items. "
            "For each item, distinguish the target function being asked about "
            "from reference functions mentioned only for comparison or transformation.",
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "小题编号，如（1）；没有编号时为空字符串。",
                                },
                                "target_expression": {
                                    "type": "string",
                                    "description": (
                                        "本小题明确要求研究、计算或变换其结果的待求函数。"
                                        "不包含只作为参照、来源或比较对象出现的函数。"
                                    ),
                                },
                                "reference_expressions": {
                                    "type": "array",
                                    "description": (
                                        "回答本小题所需的参照函数，例如图像变换的原始函数；"
                                        "没有参照函数时返回空数组。"
                                    ),
                                    "items": {"type": "string"},
                                },
                                "question_text": {
                                    "type": "string",
                                    "description": "只包含当前小题要求完成的任务文本。",
                                },
                            },
                            "required": [
                                "label",
                                "target_expression",
                                "reference_expressions",
                                "question_text",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["items"],
                "additionalProperties": False,
            },
            system=(
                "你是数学题结构化解析器。输出必须符合工具的 JSON Schema。"
                "先按小题识别题目明确要求完成的任务。"

                "target_expression 表示本小题真正要求研究、计算或描述其性质的函数。"
                "题目中出现的函数不一定都是待求函数。"

                "reference_expressions 只保存回答当前问题所需的参照函数，"
                "例如“函数 A 的图像由函数 B 的图像怎样变换得到”中，"
                "A 是 target_expression，B 是 reference_expressions。"
                "参照函数不得作为独立 item 输出，除非题目还明确要求研究该函数。"

                "question_text 必须摘录当前小题的原文求解要求，不要改写，"
                "必须保留配方法等指定解法和限制条件，不要混入其他小题的要求。"
                "label 保存题号，如（1）、（2）；没有题号时输出空字符串。"

                "所有函数表达式均输出不含 y= 的 SymPy 表达式。"
                "只能使用变量 x、数字、+、-、*、/、**、括号和小数点；"
                "幂必须使用 **，不要输出 LaTeX。"

                "示例：已知 y=-x²/2+4x+2，指出它的图像可以由 "
                "y=-x²/2 的图像经过怎样的变换得到。"
                "正确 JSON 示例为："
                '{"items":[{"label":"（1）",'
                '"target_expression":"-x**2/2+4*x+2",'
                '"reference_expressions":["-x**2/2"],'
                '"question_text":"指出它的图像可以由参照函数的图像经过怎样的变换得到"}]}。'
                "不得把 -x**2/2 作为另一个待求 item。"
            ),
            user=question,
        )
        return self._validate_extracted_problems(result)

    def extract_problem(self, question: str) -> dict:
        """Backward-compatible single-expression wrapper."""
        result = self.extract_problems(question)
        items = result.get("items")
        if not isinstance(items, list) or not items:
            raise RuntimeError("DeepSeek 未提取出函数表达式")
        return {"expression_sympy": items[0]["target_expression"]}

    def structure_solution_steps(self, question: str, raw_steps_text: str) -> dict:
        """Structure raw OCR text into normalized mathematical steps."""
        return self._tool_call(
            "structure_solution_steps",
            "将学生手写解题文本结构化为规范有序的数学步骤，区分因为/所以，并提取标准代数表达式与关键结论断言",
            {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_index": {"type": "integer"},
                                "raw_text": {"type": "string"},
                                "marker": {
                                    "type": "string",
                                    "enum": ["because", "therefore", "none"],
                                },
                                "expression_latex": {"type": "string"},
                                "expression_sympy": {"type": "string"},
                                "step_intent": {
                                    "type": "string",
                                    "enum": [
                                        "premise",
                                        "factor_coefficient",
                                        "completing_square",
                                        "determine_axis",
                                        "determine_vertex",
                                        "determine_extremum",
                                        "substitute_value",
                                        "solve_parameter",
                                        "conclusion",
                                        "other",
                                    ],
                                },
                                "claimed_axis": {"type": "string"},
                                "claimed_vertex": {"type": "string"},
                                "claimed_extremum_kind": {
                                    "type": "string",
                                    "enum": ["max", "min", "none"],
                                },
                                "claimed_extremum_value": {"type": "string"},
                                "claimed_answer": {"type": "string"},
                            },
                            "required": [
                                "step_index",
                                "raw_text",
                                "marker",
                                "expression_latex",
                                "expression_sympy",
                                "step_intent",
                                "claimed_axis",
                                "claimed_vertex",
                                "claimed_extremum_kind",
                                "claimed_extremum_value",
                                "claimed_answer",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["steps"],
                "additionalProperties": False,
            },
            system=(
                "你是一个专业的中学数学助教与解题分析器。"
                "请将学生手写识别出的解题过程，整理为有序的步骤列表。"
                "【严格规范】：\n"
                "1. 严格区分前提条件与推导结论：若内容为已知条件、性质声明（如‘∵ f(x)是奇函数’），marker 输出 'because'；若为推导出的等式、计算结果或最终答案（如‘∴ b=0’、‘∴ 最值为10’），marker 输出 'therefore'；其他普通说明输出 'none'。\n"
                "2. 提取每一步纯净的 SymPy 表达式放入 expression_sympy（只能使用变量x、a、b、数字与+-*/()**，无表达式时输出空字符串）。\n"
                "3. 提取每一步的 LaTeX 表达式放入 expression_latex。\n"
                "4. 识别断言：若学生在某一步断言了对称轴（如 x=3），填入 claimed_axis（如 '3'）；若断言了顶点，填入 claimed_vertex；若断言了最值类型和数值，填入 claimed_extremum_kind ('max'/'min') 和 claimed_extremum_value；否则填 'none' 或空字符串。\n"
            ),
            user=f"【题目】：{question}\n\n【学生手写/输入解题步骤】：\n{raw_steps_text}",
        )

    def normalize_ocr_math_steps(
        self,
        raw_ocr_text: str,
        secondary_ocr_text: str | None = None,
    ) -> dict:
        """Normalize one OCR result, optionally corroborated by a second model."""
        if secondary_ocr_text:
            user_prompt = (
                "请对以下两套 OCR 文本进行逐步对照、结构化与连续性分类。"
                "忽略空格、标点和等价 LaTeX 写法，只比较数学实质。\n\n"
                f"【主 OCR：PaddleOCR】\n{raw_ocr_text}\n\n"
                f"【复核 OCR：DeepSeek-OCR】\n{secondary_ocr_text}"
            )
        else:
            user_prompt = f"请对以下原始 OCR 文本进行结构化与连续性分类：\n\n{raw_ocr_text}"
        result = self._tool_call(
            "normalize_ocr_math_steps",
            "将原始 OCR 数学文本忠实拆解为步骤，并分类相邻步骤之间的数学连续性与正确性",
            {
                "type": "object",
                "properties": {
                    "question_stem": {
                        "type": "string",
                        "description": "从 OCR 中剥离出的纯净题目题干（规范化 LaTeX 格式）",
                    },
                    "steps": {
                        "type": "array",
                        "description": "步骤链",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_number": {"type": "integer"},
                                "marker": {
                                    "type": "string",
                                    "enum": ["because", "therefore", "none"],
                                    "description": "已知前提输出 because(∵)，推导结论输出 therefore(∴)",
                                },
                                "step_text": {
                                    "type": "string",
                                    "description": "忠实还原卷面文字的步骤描述（严禁私自篡改代数实质内容）；其中数学内容使用 KaTeX 支持的 LaTeX，行内公式用 \\(与\\)包裹",
                                },
                                "math_expression_latex": {
                                    "type": "string",
                                    "description": "本步卷面原样呈现的 LaTeX 算式",
                                },
                                "step_intent": {
                                    "type": "string",
                                    "description": "步骤意图（如：声明奇函数性质、代入求参b、代入求参a、得出解析式）",
                                },
                                "continuity_status": {
                                    "type": "string",
                                    "enum": [
                                        "complete",
                                        "acceptable_omission",
                                        "ambiguous",
                                        "logical_break",
                                    ],
                                    "description": "相对于前面有效步骤的推导连续性分类",
                                },
                                "mathematical_validity": {
                                    "type": "string",
                                    "enum": ["valid", "invalid", "unknown"],
                                    "description": "本步数学内容是否成立；OCR 无法确认时输出 unknown",
                                },
                                "ocr_agreement": {
                                    "type": "string",
                                    "enum": ["agree", "disagree", "uncertain", "not_checked"],
                                    "description": (
                                        "两套 OCR 对本步数学实质的识别是否一致；"
                                        "只提供一套 OCR 时输出 not_checked"
                                    ),
                                },
                                "ocr_comparison_status": {
                                    "type": "string",
                                    "enum": [
                                        "exact_match",
                                        "compatible_omission",
                                        "semantic_conflict",
                                        "unaligned",
                                        "not_checked",
                                    ],
                                    "description": (
                                        "两套 OCR 的数学关系：内容完整一致为 exact_match；"
                                        "一套仅省略中间内容、共同断言均一致为 compatible_omission；"
                                        "同一位置出现相互矛盾的变量、系数、符号、表达式或结论为 semantic_conflict；"
                                        "无法可靠对应为 unaligned；只提供一套 OCR 为 not_checked"
                                    ),
                                },
                                "secondary_ocr_evidence": {
                                    "type": "string",
                                    "description": (
                                        "复核 OCR 中与本步对应的原始文字；没有复核结果时为空字符串。"
                                        "其中每个数学表达式必须分别使用 KaTeX 行内定界符 \\( 与 \\) 包裹，"
                                        "例如 \\(f(t)=\\frac{1-t}{1+t}\\)；"
                                        "\\(f(x)=\\frac{1-x}{1+x}\\)"
                                    ),
                                },
                                "verification_message": {
                                    "type": "string",
                                    "description": "说明两套 OCR 一致、冲突或无法对齐的依据",
                                },
                                "omitted_reasoning": {
                                    "type": "string",
                                    "description": "acceptable_omission 时补全的简短中间推导，其他状态输出空字符串；所有数学内容使用 KaTeX 支持的 LaTeX，行内公式用 \\(与\\)包裹",
                                },
                                "diagnostic_message": {
                                    "type": "string",
                                    "description": "对当前分类的客观说明，不直接决定是否扣分；普通说明使用中文，所有数学表达式、变量和数学符号使用 KaTeX 支持的 LaTeX，行内公式用 \\(与\\)包裹",
                                },
                                "ocr_fix_suggestion": {
                                    "type": "string",
                                    "description": "若系 OCR 连笔漏扫，建议补正的标准 LaTeX 算式（若无则为空字符串）",
                                },
                                "format_fix_note": {
                                    "type": "string",
                                    "description": "纯格式/因果符号修改说明（如将∴修正为∵，无修改则为空）",
                                },
                            },
                            "required": [
                                "step_number",
                                "marker",
                                "step_text",
                                "math_expression_latex",
                                "step_intent",
                                "continuity_status",
                                "mathematical_validity",
                                "ocr_agreement",
                                "ocr_comparison_status",
                                "secondary_ocr_evidence",
                                "verification_message",
                                "omitted_reasoning",
                                "diagnostic_message",
                                "ocr_fix_suggestion",
                                "format_fix_note",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "overall_summary": {
                        "type": "string",
                        "description": "整体规范化与疑点分析总结；所有数学内容使用 KaTeX 支持的 LaTeX，行内公式用 \\(与\\)包裹",
                    },
                },
                "required": ["question_stem", "steps", "overall_summary"],
                "additionalProperties": False,
            },
            system=(
                "你是一个中学数学助教。请忠实地将原始 OCR 数学文本拆解为题目题干与步骤链，"
                "并根据相邻步骤之间的数学推导关系，为每一步选择且仅选择一种连续性状态：\n"
                "1. complete：已经写出理解当前结论所需的必要推导。\n"
                "2. acceptable_omission：省略了约分、通分、移项、合并同类项、代入或简单方程求解等常规运算，"
                "但当前结论能够从前面的有效步骤正确、唯一地推出。\n"
                "3. ambiguous：由于 OCR 漏识别、公式残缺或卷面内容不清，无法确认学生原意或推导是否成立。\n"
                "4. logical_break：即使补充合理的常规代数变形，当前结论仍然无法从前面的有效步骤推出，"
                "或者使用了错误的运算、性质、公式或定理。\n"
                "【双 OCR 证据规则】：如果同时提供主 OCR 与复核 OCR，请逐步比较数学实质。"
                "这里的一致是数学相容，不要求两套文本覆盖范围完全相同。"
                "如果一套 OCR 给出完整的 A=B=C，另一套只识别出 A=C，且共同出现的前提、等式和结论均一致，"
                "应输出 ocr_comparison_status=compatible_omission、ocr_agreement=agree；"
                "不得仅因复核 OCR 省略中间算式而输出 disagree 或 uncertain。"
                "只有两套 OCR 在同一位置都给出了数学断言，并且变量、符号、系数、表达式或结论相互矛盾时，"
                "才输出 semantic_conflict 与 disagree，并使用 ambiguous 与 unknown。"
                "无法可靠对齐对应步骤时输出 unaligned 与 uncertain，并同样使用 ambiguous 与 unknown。"
                "完整一致时输出 exact_match 与 agree；只提供一套 OCR 时两个字段均输出 not_checked。"
                "verification_message 的说明必须与结构化枚举一致，不得一面说明‘数学实质一致’，"
                "一面输出 disagree 或 semantic_conflict。\n"
                "【KaTeX 输出规范】：step_text、omitted_reasoning、diagnostic_message 和 overall_summary 中，"
                "普通说明使用中文；所有数学表达式、变量和数学符号必须使用 KaTeX 支持的 LaTeX。"
                "行内公式统一用 \\( 与 \\) 包裹，独立公式用 \\[ 与 \\] 包裹；"
                "secondary_ocr_evidence 中的每个数学表达式也必须分别使用 \\( 与 \\) 包裹，"
                "多个公式之间的中文标点必须放在定界符外；"
                "所有分式（包括分子或分母中的嵌套分式）都必须使用 \\frac{分子}{分母}，"
                "不得使用 / 表示分数；不要输出未包裹的 ASCII 算式，不要使用 Markdown 代码块。\n"
                "示例一：由 \\(\\frac{\\frac{a}{2}}{1+\\frac{1}{4}}=\\frac{2}{5}\\) 直接得到 \\(a=1\\)，"
                "应分类为 acceptable_omission。\n"
                "示例二：由 \\(\\frac{1}{1+\\frac{1}{4}}=\\frac{2}{5}\\) 直接得到 \\(a=1\\)，"
                "前式不含 \\(a\\)，应分类为 logical_break。"
            ),
            user=user_prompt,
        )
        dual_ocr = bool(secondary_ocr_text)
        for step in result.get("steps", []):
            comparison_status = step.get("ocr_comparison_status")
            if not dual_ocr:
                step["ocr_comparison_status"] = "not_checked"
                step["ocr_agreement"] = "not_checked"
                agreement = "not_checked"
            elif comparison_status in {"exact_match", "compatible_omission"}:
                step["ocr_agreement"] = "agree"
                agreement = "agree"
            elif comparison_status == "semantic_conflict":
                step["ocr_agreement"] = "disagree"
                agreement = "disagree"
            elif comparison_status == "unaligned":
                step["ocr_agreement"] = "uncertain"
                agreement = "uncertain"
            else:
                # Backward-compatible fallback for older or incomplete model
                # responses. A dual-OCR request must never remain not_checked.
                agreement = step.get("ocr_agreement", "not_checked")
                if agreement == "agree":
                    step["ocr_comparison_status"] = "exact_match"
                elif agreement == "disagree":
                    step["ocr_comparison_status"] = "semantic_conflict"
                else:
                    step["ocr_comparison_status"] = "unaligned"
                    step["ocr_agreement"] = "uncertain"
                    agreement = "uncertain"
                    step["verification_message"] = (
                        step.get("verification_message")
                        or "已提供两套 OCR，但模型未完成可靠对齐。"
                    )

            if agreement in {"disagree", "uncertain"}:
                step["continuity_status"] = "ambiguous"
                step["mathematical_validity"] = "unknown"
        return result

    def generate_standard_solution(self, question_stem: str) -> dict:
        """Generate high-school level ground-truth step-by-step mathematical solutions for all subquestions."""
        return self._tool_call(
            "generate_standard_solution",
            "为高中数学题目（包含单问或多问）生成高考满分标准的完整推导解答步骤与最终结果",
            {
                "type": "object",
                "properties": {
                    "question_stem": {
                        "type": "string",
                        "description": "规范化后的题目完整题干（LaTeX 格式）",
                    },
                    "subquestions": {
                        "type": "array",
                        "description": "各小问标准解答列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "题号标签，如（1）、（2）"},
                                "question_content": {"type": "string", "description": "该小问的具体设问"},
                                "core_method": {"type": "string", "description": "所用核心数学思想与方法（如：换元法、待定系数法、配方法、导数法）"},
                                "solution_steps": {
                                    "type": "array",
                                    "description": "详细规范的推导步骤链（LaTeX 格式）",
                                    "items": {"type": "string"}
                                },
                                "final_answer": {"type": "string", "description": "最终标准结论/答案（LaTeX 格式）"},
                            },
                            "required": ["label", "question_content", "core_method", "solution_steps", "final_answer"],
                            "additionalProperties": False,
                        },
                    },
                    "knowledge_points": {
                        "type": "array",
                        "description": "所考查的高中数学核心知识点列表",
                        "items": {"type": "string"}
                    },
                    "method_summary": {
                        "type": "string",
                        "description": "名师解题思路点拨与方法归纳",
                    },
                },
                "required": ["question_stem", "subquestions", "knowledge_points", "method_summary"],
                "additionalProperties": False,
            },
            system="你是一位资深高中数学名师。请针对给出的题目题干，严格按照高考阅卷满分规范，输出条理清晰、步骤严谨、无跳步的完整标准解答。",
            user=f"请为以下高中数学题生成完整标准解答：\n\n{question_stem}",
        )


