"""Local web entry point for the graph-driven quadratic solver."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from email.parser import BytesParser
from email.policy import default
from getpass import getpass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from dotenv import load_dotenv

from deepseek_client import DeepSeekClient
from errors import friendly_message
from multi_solver import ProblemItemOutcome, render_problem_items, solve_all_tasks
from ocr_client import OcrClient
from problem import Problem


INPUT_TEMPLATE = Path(__file__).with_name("templates") / "input.html"
CONFIRM_TEMPLATE = Path(__file__).with_name("templates") / "confirm.html"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    question, image, image_type = "", b"", "image/jpeg"
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        value = part.get_payload(decode=True) or b""
        if name == "question":
            question = value.decode(part.get_content_charset() or "utf-8", errors="replace").strip()
        elif name == "image":
            image = value
            ctype = part.get_content_type()
            if ctype and ctype not in {"application/octet-stream", "text/plain"}:
                image_type = ctype
    return question, image, image_type


OCR_TEST_TEMPLATE = Path(__file__).with_name("templates") / "ocr_test.html"
GRADE_TEMPLATE = Path(__file__).with_name("templates") / "grade.html"


def make_handler(password: str, url: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            sys.stdout.write(f"[{self.log_date_time_string()}] {self.client_address[0]} - {format % args}\n")
            sys.stdout.flush()

        def send_html(self, page: str, status: int = 200) -> None:
            try:
                data = page.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                print(f"📤 [RESPONSE] {self.path} {status} HTML ({len(data)} bytes)", flush=True)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        def send_json(self, payload: dict, status: int = 200) -> None:
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                print(f"📤 [RESPONSE] {self.path} {status} JSON ({len(data)} bytes)", flush=True)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        def do_GET(self) -> None:  # noqa: N802
            print(f"📥 [REQUEST] GET {self.path} (来自 {self.client_address[0]})", flush=True)
            if self.path in {"/ocr_test", "/test"}:
                self.send_html(OCR_TEST_TEMPLATE.read_text(encoding="utf-8"))
                return
            if self.path == "/grade":
                self.send_html(GRADE_TEMPLATE.read_text(encoding="utf-8"))
                return
            if self.path != "/":
                self.send_error(404)
                return
            self.send_html(input_page())

        def do_POST(self) -> None:  # noqa: N802
            print(f"📥 [REQUEST] POST {self.path} (来自 {self.client_address[0]})", flush=True)
            if self.path not in {"/prepare", "/solve", "/api/ocr_test", "/api/grade", "/api/normalize_ocr", "/api/grade_steps", "/api/grade_photo"}:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_UPLOAD_BYTES:
                if self.path in {"/api/ocr_test", "/api/grade", "/api/normalize_ocr", "/api/grade_steps", "/api/grade_photo"}:
                    self.send_json({"error": "上传数据不能超过 10 MB。"}, status=400)
                else:
                    self.send_html(input_page(error="图片不能超过 10 MB。"), status=400)
                return
            body = self.rfile.read(length)
            question = ""
            try:
                if self.path == "/api/grade_photo":
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.startswith("multipart/form-data"):
                        raise ValueError("上传表单格式无效。")
                    import time
                    selected_model, image, image_type = multipart_form(content_type, body)
                    if not image:
                        raise ValueError("未接收到图片数据。")
                    
                    t0 = time.time()
                    model_to_use = selected_model.strip() if selected_model and ("Paddle" in selected_model or "DeepSeek" in selected_model) else None
                    
                    # 1. OCR 识别
                    raw_text = OcrClient().transcribe(image, image_type, model=model_to_use)
                    t_ocr = time.time() - t0
                    
                    # 2. 语义规范化与题干/步骤分离
                    ds_client = DeepSeekClient()
                    norm_res = ds_client.normalize_ocr_math_steps(raw_text)
                    t_norm = time.time() - t0 - t_ocr
                    
                    # 3. 符号数学智能批改
                    stem = norm_res.get("question_stem", "")
                    steps = norm_res.get("steps", [])
                    from grader import grade_normalized_steps
                    report = grade_normalized_steps(stem, steps, ds_client)
                    t_grade = time.time() - t0 - t_ocr - t_norm
                    
                    total_time = time.time() - t0
                    self.send_json({
                        "raw_ocr_text": raw_text,
                        "question_stem": stem,
                        "normalized_steps": steps,
                        "overall_summary": norm_res.get("overall_summary", ""),
                        "grading_report": report,
                        "model_used": model_to_use or OcrClient().default_model,
                        "timings": {
                            "ocr_seconds": round(t_ocr, 2),
                            "normalization_seconds": round(t_norm, 2),
                            "grading_seconds": round(t_grade, 2),
                            "total_seconds": round(total_time, 2)
                        }
                    })
                    return

                if self.path == "/api/normalize_ocr":
                    payload = json.loads(body.decode("utf-8"))
                    raw_text = payload.get("raw_text", "").strip()
                    if not raw_text:
                        raise ValueError("待纠错的 OCR 文本不能为空。")

                    client = DeepSeekClient()
                    res = client.normalize_ocr_math_steps(raw_text)
                    self.send_json(res)
                    return

                if self.path == "/api/grade_steps":
                    payload = json.loads(body.decode("utf-8"))
                    question_stem = payload.get("question_stem", "").strip()
                    steps = payload.get("steps", [])
                    if not question_stem or not steps:
                        raise ValueError("题干与规范步骤链不能为空。")
                    from grader import grade_normalized_steps
                    client = DeepSeekClient()
                    report_dict = grade_normalized_steps(question_stem, steps, client)
                    self.send_json(report_dict)
                    return

                if self.path == "/api/grade":
                    payload = json.loads(body.decode("utf-8"))
                    question = payload.get("question", "").strip()
                    steps_text = payload.get("steps_text", "").strip()
                    if not question or not steps_text:
                        raise ValueError("题目和解题步骤不能为空。")

                    from grader import grade_solution, structure_student_steps

                    client = DeepSeekClient()
                    extracted = client.extract_problems(question)
                    items = extracted.get("items", [])
                    if not items:
                        raise ValueError("未能从题干中识别出二次函数表达式。")

                    target_expr = items[0]["target_expression"]
                    problem = Problem(
                        question_text=question,
                        expression_sympy=target_expr,
                    )
                    student_steps = structure_student_steps(question, steps_text, client)
                    report = grade_solution(problem, student_steps)
                    self.send_json(report.to_dict())
                    return

                if self.path == "/api/ocr_test":
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.startswith("multipart/form-data"):
                        raise ValueError("上传表单格式无效。")
                    import time
                    selected_model, image, image_type = multipart_form(content_type, body)
                    if not image:
                        raise ValueError("未接收到图片数据。")
                    t0 = time.time()
                    model_to_use = selected_model.strip() if selected_model and ("Paddle" in selected_model or "DeepSeek" in selected_model) else None
                    text = OcrClient().transcribe(image, image_type, model=model_to_use)
                    elapsed = time.time() - t0
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    self.send_json({
                        "text": text,
                        "elapsed_seconds": round(elapsed, 2),
                        "char_count": len(text),
                        "lines": lines,
                        "model_used": model_to_use or OcrClient().default_model,
                    })
                    return

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
                import traceback
                print(f"❌ [ERROR] {self.path} 处理失败: {error}", flush=True)
                traceback.print_exc()
                if self.path.startswith("/api/"):
                    self.send_json({"error": friendly_message(error)}, status=400)
                else:
                    self.send_html(input_page(question, friendly_message(error)), status=400)

        def log_message(self, format: str, *args: object) -> None:
            print(format % args)

    return Handler


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    default_host = os.getenv("HOST", "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1")
    default_port = int(os.getenv("PORT", "7860" if os.getenv("SPACE_ID") else "8000"))
    parser = argparse.ArgumentParser(description="本地数学解题网页")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--url", default="http://localhost:7474/db/math/tx/commit")
    args = parser.parse_args()

    password = os.getenv("NEO4J_PASSWORD", "")
    if not password and not os.getenv("SPACE_ID") and sys.stdin.isatty():
        try:
            password = getpass("Neo4j password: ")
        except Exception:
            password = ""
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
