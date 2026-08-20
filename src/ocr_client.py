"""SiliconFlow DeepSeek-OCR client used by the local web entry point."""

from __future__ import annotations

import base64
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from errors import ConfigurationError, UpstreamServiceError


API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OCR_MODEL", "PaddlePaddle/PaddleOCR-VL-1.5")
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}


class OcrClient:
    """Turn an in-memory image uploaded by the browser into plain problem text."""

    def __init__(self, api_key: str | None = None, default_model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.default_model = default_model
        if not self.api_key:
            raise ConfigurationError("未设置 SILICONFLOW_API_KEY，无法识别题目截图。")

    def transcribe(self, image: bytes, media_type: str, model: str | None = None) -> str:
        if media_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError("只支持 PNG、JPEG 或 WEBP 图片。")
        if not image:
            raise ValueError("上传的图片为空。")
        target_model = model or self.default_model
        encoded = base64.b64encode(image).decode("ascii")

        prompt = "请识别图中的所有文字和数学公式，使用 LaTeX 格式输出。"

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}", "detail": "high"}},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0,
            "frequency_penalty": 0.1,
            "max_tokens": 2048,
        }
        request = Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        result = None
        for attempt in range(2):
            try:
                with urlopen(request, timeout=30) as response:
                    result = json.load(response)
                    break
            except Exception as error:
                if attempt == 1:
                    if isinstance(error, HTTPError):
                        raise UpstreamServiceError(
                            f"题目图片识别服务请求失败（HTTP {error.code}），请稍后重试。"
                        ) from error
                    raise UpstreamServiceError("无法连接题目图片识别服务，请检查网络后重试。") from error
                import time
                time.sleep(0.5)
        try:
            text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise UpstreamServiceError("题目图片识别服务未返回文字结果，请换一张更清晰的图片。") from error
        if not isinstance(text, str) or not text.strip():
            raise UpstreamServiceError("未能从图片中识别出题干，请换一张更清晰的图片。")
        
        cleaned = self._clean_repetitive_tail(text.strip())
        return cleaned

    @staticmethod
    def _clean_repetitive_tail(text: str) -> str:
        """Deduplicate consecutive identical clauses and strip trailing margin noise."""
        prev = None
        curr = text
        while prev != curr:
            prev = curr
            # Clean repeated ∴/∵ \( expr \) , ∴ \( expr \)
            curr = re.sub(r'([∴∵]?\s*\\\(.+?\\\))[,，;；\s]+\1', r'\1', curr)
            # Clean general repeated phrases before comma/period
            curr = re.sub(r'([^\n,，;；]+)[,，;；\s]+\1(?=[,，;；。\s]|$)', r'\1', curr)
            # Clean repeated standalone identical lines
            curr = re.sub(r'(^|\n)([^\n]+)\n\2(?=\n|$)', r'\1\2', curr)

        # Filter out trailing isolated noise line numbers like "1." or "1. 2. 3."
        curr = re.sub(r'\n+\s*(?:\d+[\.、\s]*)+$', '', curr)
        return curr.strip()
