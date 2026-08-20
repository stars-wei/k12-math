"""Local web entry point for the graph-driven quadratic solver."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

from deepseek_client import DeepSeekClient
from errors import friendly_message
from grading_trace import create_trace_recorder, image_metadata
from ocr_client import OcrClient
from problem import Problem


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


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


def multipart_form(content_type: str, body: bytes) -> tuple[dict[str, str], bytes, str, str]:
    """Read all fields and the image in the upload form without adding a web framework."""
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    fields: dict[str, str] = {}
    image, image_type, image_filename = b"", "image/jpeg", ""
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        value = part.get_payload(decode=True) or b""
        if name == "image":
            image = value
            image_filename = part.get_filename() or ""
            ctype = part.get_content_type()
            if ctype and ctype not in {"application/octet-stream", "text/plain"}:
                image_type = ctype
        elif name:
            fields[name] = value.decode(part.get_content_charset() or "utf-8", errors="replace").strip()
    return fields, image, image_type, image_filename


STUDIO_TEMPLATE = Path(__file__).with_name("templates") / "studio.html"


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            sys.stdout.write(f"[{self.log_date_time_string()}] {self.client_address[0]} - {format % args}\n")
            sys.stdout.flush()

        def start_text_trace(self, payload: dict) -> None:
            if self.trace is not None and self.trace.start("text"):
                self.trace.event("request_input", {"endpoint": self.path, "payload": payload})

        def deepseek_client(self) -> DeepSeekClient:
            return DeepSeekClient(
                trace_callback=self.trace.model_event if self.trace is not None else None
            )

        def send_html(self, page: str, status: int = 200) -> None:
            try:
                data = page.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                print(f"📤 [RESPONSE] {self.path} {status} HTML ({len(data)} bytes)", flush=True)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        def send_json(self, payload: dict, status: int = 200) -> None:
            trace = getattr(self, "trace", None)
            if trace is not None:
                if trace.trace_id is not None:
                    payload.setdefault("trace_id", str(trace.trace_id))
                if not trace.finalized:
                    if status >= 400:
                        trace.fail(payload.get("error", f"HTTP {status}"))
                    else:
                        trace.complete(payload)
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                print(f"📤 [RESPONSE] {self.path} {status} JSON ({len(data)} bytes)", flush=True)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        def do_GET(self) -> None:  # noqa: N802
            print(f"📥 [REQUEST] GET {self.path} (来自 {self.client_address[0]})", flush=True)
            if self.path in {"/", "/studio", "/grade", "/ocr_test", "/test"}:
                self.send_html(STUDIO_TEMPLATE.read_text(encoding="utf-8"))
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            self.trace = create_trace_recorder()
            print(f"📥 [REQUEST] POST {self.path} (来自 {self.client_address[0]})", flush=True)
            if self.path not in {"/api/ocr_test", "/api/grade", "/api/normalize_ocr", "/api/grade_steps", "/api/grade_photo", "/api/solve_standard", "/api/analyze"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json({"error": "请求长度格式无效。"}, status=400)
                return
            if length > MAX_UPLOAD_BYTES:
                self.send_json({"error": "上传数据不能超过 10 MB。"}, status=400)
                return
            body = self.rfile.read(length)
            try:
                if self.path in {"/api/grade_photo", "/api/analyze"}:
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.startswith("multipart/form-data"):
                        raise ValueError("上传表单格式无效。")
                    fields, image, image_type, image_filename = multipart_form(content_type, body)
                    if not image:
                        raise ValueError("未接收到图片数据。")
                    
                    intent = fields.get("intent", "auto").strip().lower()
                    if intent not in {"auto", "grade", "solve"}:
                        raise ValueError("处理方式无效。")
                    selected_model = fields.get("model", "") or fields.get("question", "")
                    model_to_use = selected_model.strip() if selected_model and ("Paddle" in selected_model or "DeepSeek" in selected_model) else None

                    metadata = image_metadata(image_filename, image_type, image)
                    if self.trace is not None and self.trace.start("image", **metadata):
                        self.trace.event(
                            "request_input",
                            {
                                "endpoint": self.path,
                                "intent": intent,
                                "ocr_mode": "dual",
                                "primary_ocr_model": model_to_use or OcrClient().default_model,
                                "image": metadata,
                            },
                        )

                    t0 = time.time()
                    # 1. 两套 OCR 并行识别
                    ocr_pair = OcrClient().transcribe_pair(
                        image,
                        image_type,
                        primary_model=model_to_use,
                    )
                    raw_text = ocr_pair["primary_text"]
                    secondary_ocr_text = ocr_pair["secondary_text"]
                    t_ocr = time.time() - t0
                    if self.trace is not None:
                        self.trace.event(
                            "ocr_result",
                            {"role": "primary", "text": raw_text},
                            duration_ms=round(t_ocr * 1000),
                            model_name=ocr_pair["primary_model"],
                        )
                        if secondary_ocr_text:
                            self.trace.event(
                                "ocr_result",
                                {"role": "secondary", "text": secondary_ocr_text},
                                duration_ms=round(t_ocr * 1000),
                                model_name=ocr_pair["secondary_model"],
                            )
                        elif ocr_pair["secondary_error"]:
                            self.trace.event(
                                "ocr_secondary_error",
                                {
                                    "model": ocr_pair["secondary_model"],
                                    "message": ocr_pair["secondary_error"],
                                },
                            )
                    ds_client = DeepSeekClient(
                        trace_callback=self.trace.model_event if self.trace is not None else None
                    )

                    # 若选择【纯题干标准求解】模式，直接生成高中数学高考满分标准步骤推导
                    if intent == "solve":
                        sol_res = ds_client.generate_standard_solution(raw_text)
                        if self.trace is not None:
                            self.trace.event("standard_solution", sol_res)
                        t_solve = time.time() - t0 - t_ocr
                        total_time = time.time() - t0
                        self.send_json({
                            "mode": "solve",
                            "raw_ocr_text": raw_text,
                            "secondary_ocr_text": secondary_ocr_text,
                            "ocr_models": [ocr_pair["primary_model"], ocr_pair["secondary_model"]],
                            "question_stem": sol_res.get("question_stem", raw_text),
                            "solution_data": sol_res,
                            "model_used": ocr_pair["primary_model"],
                            "timings": {
                                "ocr_seconds": round(t_ocr, 2),
                                "solver_seconds": round(t_solve, 2),
                                "total_seconds": round(total_time, 2)
                            }
                        })
                        return

                    # 2. 语义规范化与题干/步骤分离
                    norm_res = ds_client.normalize_ocr_math_steps(
                        raw_text,
                        secondary_ocr_text=secondary_ocr_text,
                    )
                    if self.trace is not None:
                        self.trace.event("normalized_steps", norm_res)
                    t_norm = time.time() - t0 - t_ocr
                    stem = norm_res.get("question_stem", "")
                    steps = norm_res.get("steps", [])

                    # 若智能自适应模式下未检测到手写解题步骤，自动切为标准求解
                    if intent == "auto" and (not steps or len(steps) == 0):
                        sol_res = ds_client.generate_standard_solution(stem or raw_text)
                        if self.trace is not None:
                            self.trace.event("standard_solution", sol_res)
                        t_solve = time.time() - t0 - t_ocr - t_norm
                        total_time = time.time() - t0
                        self.send_json({
                            "mode": "solve",
                            "raw_ocr_text": raw_text,
                            "secondary_ocr_text": secondary_ocr_text,
                            "ocr_models": [ocr_pair["primary_model"], ocr_pair["secondary_model"]],
                            "question_stem": sol_res.get("question_stem", stem or raw_text),
                            "solution_data": sol_res,
                            "model_used": ocr_pair["primary_model"],
                            "timings": {
                                "ocr_seconds": round(t_ocr, 2),
                                "solver_seconds": round(t_solve, 2),
                                "total_seconds": round(total_time, 2)
                            }
                        })
                        return

                    # 3. 符号数学智能批改
                    from grader import grade_normalized_steps
                    report = grade_normalized_steps(stem, steps, ds_client)
                    if self.trace is not None:
                        self.trace.event("grading_report", report)
                    t_grade = time.time() - t0 - t_ocr - t_norm
                    total_time = time.time() - t0
                    self.send_json({
                        "mode": "grade",
                        "raw_ocr_text": raw_text,
                        "secondary_ocr_text": secondary_ocr_text,
                        "ocr_models": [ocr_pair["primary_model"], ocr_pair["secondary_model"]],
                        "question_stem": stem,
                        "normalized_steps": steps,
                        "overall_summary": norm_res.get("overall_summary", ""),
                        "grading_report": report,
                        "model_used": ocr_pair["primary_model"],
                        "timings": {
                            "ocr_seconds": round(t_ocr, 2),
                            "normalization_seconds": round(t_norm, 2),
                            "grading_seconds": round(t_grade, 2),
                            "total_seconds": round(total_time, 2)
                        }
                    })
                    return

                if self.path == "/api/solve_standard":
                    payload = json.loads(body.decode("utf-8"))
                    self.start_text_trace(payload)
                    stem = payload.get("question_stem", "").strip()
                    if not stem:
                        raise ValueError("题干不能为空。")
                    ds_client = self.deepseek_client()
                    sol_res = ds_client.generate_standard_solution(stem)
                    if self.trace is not None:
                        self.trace.event("standard_solution", sol_res)
                    self.send_json(sol_res)
                    return

                if self.path == "/api/normalize_ocr":
                    payload = json.loads(body.decode("utf-8"))
                    self.start_text_trace(payload)
                    raw_text = payload.get("raw_text", "").strip()
                    if not raw_text:
                        raise ValueError("待纠错的 OCR 文本不能为空。")

                    client = self.deepseek_client()
                    res = client.normalize_ocr_math_steps(raw_text)
                    if self.trace is not None:
                        self.trace.event("normalized_steps", res)
                    self.send_json(res)
                    return

                if self.path == "/api/grade_steps":
                    payload = json.loads(body.decode("utf-8"))
                    self.start_text_trace(payload)
                    question_stem = payload.get("question_stem", "").strip()
                    steps = payload.get("steps", [])
                    if not question_stem or not steps:
                        raise ValueError("题干与规范步骤链不能为空。")
                    from grader import grade_normalized_steps
                    client = self.deepseek_client()
                    report_dict = grade_normalized_steps(question_stem, steps, client)
                    if self.trace is not None:
                        self.trace.event("grading_report", report_dict)
                    self.send_json(report_dict)
                    return

                if self.path == "/api/grade":
                    payload = json.loads(body.decode("utf-8"))
                    self.start_text_trace(payload)
                    question = payload.get("question", "").strip()
                    steps_text = payload.get("steps_text", "").strip()
                    if not question or not steps_text:
                        raise ValueError("题目和解题步骤不能为空。")

                    from grader import grade_solution, structure_student_steps

                    client = self.deepseek_client()
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
                    report_dict = report.to_dict()
                    if self.trace is not None:
                        self.trace.event("grading_report", report_dict)
                    self.send_json(report_dict)
                    return

                if self.path == "/api/ocr_test":
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.startswith("multipart/form-data"):
                        raise ValueError("上传表单格式无效。")
                    fields, image, image_type, image_filename = multipart_form(content_type, body)
                    if not image:
                        raise ValueError("未接收到图片数据。")
                    t0 = time.time()
                    selected_model = fields.get("model", "") or fields.get("question", "")
                    model_to_use = selected_model.strip() if selected_model and ("Paddle" in selected_model or "DeepSeek" in selected_model) else None

                    metadata = image_metadata(image_filename, image_type, image)
                    if self.trace is not None and self.trace.start("image", **metadata):
                        self.trace.event(
                            "request_input",
                            {
                                "endpoint": self.path,
                                "ocr_mode": "dual",
                                "primary_ocr_model": model_to_use or OcrClient().default_model,
                                "image": metadata,
                            },
                        )
                    ocr_pair = OcrClient().transcribe_pair(
                        image,
                        image_type,
                        primary_model=model_to_use,
                    )
                    text = ocr_pair["primary_text"]
                    secondary_text = ocr_pair["secondary_text"]
                    elapsed = time.time() - t0
                    if self.trace is not None:
                        self.trace.event(
                            "ocr_result",
                            {"role": "primary", "text": text},
                            duration_ms=round(elapsed * 1000),
                            model_name=ocr_pair["primary_model"],
                        )
                        if secondary_text:
                            self.trace.event(
                                "ocr_result",
                                {"role": "secondary", "text": secondary_text},
                                duration_ms=round(elapsed * 1000),
                                model_name=ocr_pair["secondary_model"],
                            )
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    self.send_json({
                        "text": text,
                        "secondary_text": secondary_text,
                        "elapsed_seconds": round(elapsed, 2),
                        "char_count": len(text),
                        "lines": lines,
                        "model_used": ocr_pair["primary_model"],
                        "ocr_models": [ocr_pair["primary_model"], ocr_pair["secondary_model"]],
                    })
                    return
            except Exception as error:
                import traceback
                print(f"❌ [ERROR] {self.path} 处理失败: {error}", flush=True)
                traceback.print_exc()
                if self.trace is not None:
                    self.trace.fail(error)
                self.send_json({"error": friendly_message(error)}, status=400)

    return Handler


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    default_host = os.getenv("HOST", "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1")
    default_port = int(os.getenv("PORT", "7860" if os.getenv("SPACE_ID") else "8000"))
    parser = argparse.ArgumentParser(description="本地数学解题网页")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler())
    print(f"打开 http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
