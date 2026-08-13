"""SiliconFlow DeepSeek-OCR client used by the local web entry point."""

from __future__ import annotations

import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from errors import ConfigurationError, UpstreamServiceError


API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-OCR"
OCR_PROMPT = "<image>\nFree OCR."
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}


class OcrClient:
    """Turn an in-memory image uploaded by the browser into plain problem text."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ConfigurationError("未设置 SILICONFLOW_API_KEY，无法识别题目截图。")

    def transcribe(self, image: bytes, media_type: str) -> str:
        if media_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError("只支持 PNG、JPEG 或 WEBP 图片。")
        if not image:
            raise ValueError("上传的图片为空。")
        encoded = base64.b64encode(image).decode("ascii")
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}", "detail": "high"}},
                {"type": "text", "text": OCR_PROMPT},
            ]}],
            "temperature": 0,
            "max_tokens": 1500,
        }
        request = Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                result = json.load(response)
        except HTTPError as error:
            raise UpstreamServiceError(
                f"题目图片识别服务请求失败（HTTP {error.code}），请稍后重试。"
            ) from error
        except URLError as error:
            raise UpstreamServiceError("无法连接题目图片识别服务，请检查网络后重试。") from error
        try:
            text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise UpstreamServiceError("题目图片识别服务未返回文字结果，请换一张更清晰的图片。") from error
        if not isinstance(text, str) or not text.strip():
            raise UpstreamServiceError("未能从图片中识别出题干，请换一张更清晰的图片。")
        return text.strip()
