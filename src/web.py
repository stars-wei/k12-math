"""Local web entry point for the graph-driven quadratic solver."""

from __future__ import annotations

import argparse
import html
import os
import re
from email.parser import BytesParser
from email.policy import default
from getpass import getpass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from deepseek_client import DeepSeekClient
from errors import friendly_message
from multi_solver import ProblemItemOutcome, render_problem_items, solve_all_tasks
from ocr_client import OcrClient
from problem import Problem


INPUT_TEMPLATE = Path(__file__).with_name("templates") / "input.html"
CONFIRM_TEMPLATE = Path(__file__).with_name("templates") / "confirm.html"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def input_page(question: str = "", error: str = "") -> str:
    """Render the small local input page without adding a web framework."""
    return (
        INPUT_TEMPLATE.read_text(encoding="utf-8")
        .replace("{{question}}", html.escape(question))
        .replace("{{error}}", html.escape(error))
    )


def confirm_page(question: str) -> str:
    """Render the OCR result for user confirmation before any solving call."""
    return CONFIRM_TEMPLATE.read_text(encoding="utf-8").replace("{{question}}", html.escape(question))


def build_problem(question: str, extracted: dict) -> Problem:
    """Preserve the source question and attach one validated extracted item."""
    return Problem(
        question_text=question,
        expression_sympy=extracted["target_expression"],
        item_question_text=extracted["question_text"],
        reference_expressions=tuple(extracted["reference_expressions"]),
    )


def normalize_item_label(label: object, index: int, total: int) -> str:
    """Render stable local numbering instead of trusting model punctuation."""
    if total <= 1:
        return ""
    match = re.search(r"\d+", label) if isinstance(label, str) else None
    number = match.group(0) if match else str(index)
    return f"（{number}）"


def multipart_form(content_type: str, body: bytes) -> tuple[str, bytes, str]:
    """Read the two fields in the upload form without adding a web framework."""
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    question, image, image_type = "", b"", ""
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        value = part.get_payload(decode=True) or b""
        if name == "question":
            question = value.decode(part.get_content_charset() or "utf-8", errors="replace").strip()
        elif name == "image" and part.get_filename():
            image, image_type = value, part.get_content_type()
    return question, image, image_type


def make_handler(password: str, url: str):
    class Handler(BaseHTTPRequestHandler):
        def send_html(self, page: str, status: int = 200) -> None:
            data = page.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:
                # The learner closed or refreshed the page before the response.
                # This is normal for a local development server.
                return

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/":
                self.send_error(404)
                return
            self.send_html(input_page())

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/prepare", "/solve"}:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_UPLOAD_BYTES:
                self.send_html(input_page(error="图片不能超过 10 MB。"), status=400)
                return
            body = self.rfile.read(length)
            try:
                if self.path == "/prepare":
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.startswith("multipart/form-data"):
                        raise ValueError("上传表单格式无效。")
                    question, image, image_type = multipart_form(content_type, body)
                    if image:
                        question = OcrClient().transcribe(image, image_type)
                    if not question:
                        raise ValueError("请输入题干或上传题目图片。")
                    self.send_html(confirm_page(question))
                    return

                form = parse_qs(body.decode("utf-8"))
                question = form.get("question", [""])[0].strip()
                if not question:
                    raise ValueError("题干不能为空。")
                client = DeepSeekClient()
                extracted = client.extract_problems(question)
                extracted_items = extracted.get("items")
                if not isinstance(extracted_items, list) or not extracted_items:
                    raise ValueError("未能从题目中提取出函数表达式。")
                item_results: list[ProblemItemOutcome] = []
                total = len(extracted_items)
                for index, item in enumerate(extracted_items, start=1):
                    problem = build_problem(question, item)
                    outcomes = solve_all_tasks(problem, password, url, client)
                    label = normalize_item_label(item.get("label"), index, total)
                    item_results.append(ProblemItemOutcome(label, problem, outcomes))
                self.send_html(render_problem_items(question, item_results))
            except Exception as error:
                self.send_html(input_page(question, friendly_message(error)), status=400)

        def log_message(self, format: str, *args: object) -> None:
            print(format % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="本地数学解题网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--url", default="http://localhost:7474/db/math/tx/commit")
    args = parser.parse_args()

    password = os.getenv("NEO4J_PASSWORD") or getpass("Neo4j password: ")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(password, args.url))
    print(f"打开 http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
