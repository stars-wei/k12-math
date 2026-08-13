"""Minimal DeepSeek JSON client. It never knows Neo4j or SymPy details."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from errors import ConfigurationError, UpstreamServiceError


# DeepSeek strict function calling is currently exposed through the beta URL.
API_URL = "https://api.deepseek.com/beta/chat/completions"


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
            tool_call = payload["choices"][0]["message"]["tool_calls"][0]
            if tool_call["function"]["name"] != tool_name:
                raise KeyError("unexpected function")
            result = json.loads(tool_call["function"]["arguments"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise UpstreamServiceError("DeepSeek 未返回可用的题目解析结果，请稍后重试。") from error
        if not isinstance(result, dict):
            raise UpstreamServiceError("DeepSeek 返回的题目解析格式无效，请稍后重试。")
        return result

    def extract_problems(self, question: str) -> dict:
        """Extract every independently solvable function expression in the question."""
        return self._tool_call(
            "extract_problem_items",
            "Extract all function expressions needed by the local quadratic solver.",
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "expression_sympy": {"type": "string"},
                            },
                            "required": ["label", "expression_sympy"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["items"],
                "additionalProperties": False,
            },
            system=(
                "你是数学题输入解析器。找出题目要求处理的每一个函数，按题目顺序分别输出。"
                "label 保存题号，如（1）、（2）；没有题号时输出空字符串。"
                "expression_sympy 输出不含 y= 的 SymPy 表达式。"
                "只能使用变量 x、数字、+、-、*、/、**、括号和小数点；幂必须使用 **。"
                "忽略题号、列表编号和 Markdown 标记；不要添加 1* 等冗余单位因子。"
                "例如二分之一乘 x 平方应输出 x**2/2。"
                "公共的解题要求不是函数，不要把同一个函数因多个题型重复输出。"
                "不要计算、不要复述或改写题意、不要输出 LaTeX。"
            ),
            user=question,
        )

    def extract_problem(self, question: str) -> dict:
        """Backward-compatible single-expression wrapper."""
        result = self.extract_problems(question)
        items = result.get("items")
        if not isinstance(items, list) or not items:
            raise RuntimeError("DeepSeek 未提取出函数表达式")
        return {"expression_sympy": items[0]["expression_sympy"]}

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
