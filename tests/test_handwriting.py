"""Test DeepSeek-OCR on handwritten text images.

Usage:
    python tests/test_handwriting.py <image_path>
"""

import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

# Load .env
load_dotenv()

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-OCR"
OCR_PROMPT = "<image>\nFree OCR."


def test_ocr(image_path: Path) -> dict:
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("未在环境变量或 .env 中找到 SILICONFLOW_API_KEY。")
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到测试图片: {image_path}")

    media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{media_type};base64,{encoded}"

    payload = {
        "model": MODEL,
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

    req = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    print(f"[*] 正在调用 DeepSeek-OCR 识别图片: {image_path.name} ({len(image_bytes)/1024:.1f} KB)...")
    start_time = time.time()
    try:
        with urlopen(req, timeout=90) as resp:
            result = json.load(resp)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API 请求失败 (HTTP {e.code}): {detail}") from e
    except URLError as e:
        raise RuntimeError(f"网络连接错误: {e.reason}") from e

    elapsed = time.time() - start_time
    content = result["choices"][0]["message"]["content"].strip()

    return {
        "text": content,
        "elapsed_seconds": round(elapsed, 2),
        "char_count": len(content),
        "lines": [line.strip() for line in content.split("\n") if line.strip()],
    }


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if len(sys.argv) < 2:
        print("用法: python tests/test_handwriting.py <图片路径>")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    res = test_ocr(target_path)
    print("\n" + "=" * 50)
    print(f"[OK] 识别完成！耗时: {res['elapsed_seconds']} 秒 | 识别字符数: {res['char_count']}")
    print("=" * 50)
    print("【DeepSeek-OCR 原始转写内容】：")
    print(res["text"])
    print("=" * 50)
