"""Minimal DeepSeek JSON client. It never knows Neo4j or SymPy details."""

# from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from errors import ConfigurationError, UpstreamServiceError


# DeepSeek strict function calling is currently exposed through the beta URL.
API_URL = "https://api.deepseek.com/beta/chat/completions"
SAFE_EXPRESSION = re.compile(r"^[0-9xX+\-*/().\s]+$")


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if not self.api_key:
            raise ConfigurationError("未设置 DEEPSEEK_API_KEY，无法进行题目解析。")

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
        request = Request(
            API_URL,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except HTTPError as error:
            raise UpstreamServiceError(
                f"DeepSeek 题目解析服务请求失败（HTTP {error.code}），请稍后重试。"
            ) from error
        except URLError as error:
            raise UpstreamServiceError("无法连接 DeepSeek 题目解析服务，请检查网络后重试。") from error

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
                    elif char == 'n' and (i + 1 >= n or s[i+1] in {' ', '\n', '\r', '\t', '"', ',', '}', ']'}):
                        # Genuine newline escape in JSON
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

    def normalize_ocr_math_steps(self, raw_ocr_text: str) -> dict:
        """Extract problem stem and perform mathematical semantic normalization on OCR steps."""
        return self._tool_call(
            "normalize_ocr_math_steps",
            "将原始 OCR 数学文本忠实拆解为步骤，区分格式修正与代数实质内容，检测因果断裂与漏写疑点",
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
                                    "description": "忠实还原卷面文字的步骤描述（严禁私自篡改代数实质内容）",
                                },
                                "math_expression_latex": {
                                    "type": "string",
                                    "description": "本步卷面原样呈现的 LaTeX 算式",
                                },
                                "step_intent": {
                                    "type": "string",
                                    "description": "步骤意图（如：声明奇函数性质、代入求参b、代入求参a、得出解析式）",
                                },
                                "has_discontinuity": {
                                    "type": "boolean",
                                    "description": "是否检测到代数断裂/漏写未知数（如本式算不通但后文却解出未知数）",
                                },
                                "pedagogical_warning": {
                                    "type": "string",
                                    "description": "若学生纸面上确实漏写，给出的教学扣分警示与不严谨诊断",
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
                                "has_discontinuity",
                                "pedagogical_warning",
                                "ocr_fix_suggestion",
                                "format_fix_note",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "overall_summary": {
                        "type": "string",
                        "description": "整体规范化与疑点分析总结",
                    },
                },
                "required": ["question_stem", "steps", "overall_summary"],
                "additionalProperties": False,
            },
            system="你是一个中学数学助教。请将原始 OCR 数学文本拆解为题目题干与步骤链，检测步骤中的代数断裂与格式问题。",
            user=f"请对以下原始 OCR 文本进行结构化与断裂检测分析：\n\n{raw_ocr_text}",
        )


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


