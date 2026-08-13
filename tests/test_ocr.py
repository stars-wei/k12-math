"""Use SiliconFlow DeepSeek-OCR to transcribe one local math-problem image.

Run:
    python test_ocr.py "D:\\Downloads\\image\\下载.png"

Before running, set the SILICONFLOW_API_KEY environment variable.
"""

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# DeepSeek-OCR 的无布局 OCR 模式：只返回文字，不返回文本框坐标。
OCR_PROMPT = "<image>\nFree OCR."


def transcribe(image_path: Path) -> str:
    """Read one image, call DeepSeek-OCR, and return its text response."""
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("未设置环境变量 SILICONFLOW_API_KEY。")
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片：{image_path}")

    media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:{media_type};base64,{encoded}"

    payload = {
        "model": "deepseek-ai/DeepSeek-OCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1500,
    }
    request = Request(
        "https://api.siliconflow.cn/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SiliconFlow API 请求失败：HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"无法连接 SiliconFlow API：{error.reason}") from error

    return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("用法：python test_ocr.py <图片路径>")
    print(transcribe(Path(sys.argv[1])))
