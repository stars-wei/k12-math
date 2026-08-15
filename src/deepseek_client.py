"""Minimal DeepSeek JSON client. It never knows Neo4j or SymPy details."""

from __future__ import annotations

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
            with urlopen(request, timeout=30) as response:
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
            result = json.loads(tool_call["function"]["arguments"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise UpstreamServiceError("DeepSeek 未返回可用的题目解析结果，请稍后重试。") from error
        if not isinstance(result, dict):
            raise UpstreamServiceError("DeepSeek 返回的题目解析格式无效，请稍后重试。")
        return result

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

    def select_id(self, system: str, user: str, candidate_ids: list[str]) -> dict:
        """Select one graph ID using a dynamic, strict JSON-schema enum."""
        return self._tool_call(
            "select_candidate",
            "Select exactly one ID from the supplied candidates.",
            {
                "type": "object",
                "properties": {
                    "selected_id": {"type": "string", "enum": candidate_ids},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["selected_id", "reason", "confidence"],
                "additionalProperties": False,
            },
            system,
            user,
        )
