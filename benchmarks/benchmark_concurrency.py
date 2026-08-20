"""API 并发与压力测试工具 (API Concurrency & Stress Benchmark)

支持测试模式：
1. local_ocr      : 本地后端 OCR 测试接口 (/api/ocr_test)
2. local_grade    : 本地后端拍照一键批改全链路接口 (/api/grade_photo)
3. upstream_ocr   : 直连 SiliconFlow 视觉大模型 API (测试云端并发与 429 阈值)
4. upstream_ds    : 直连 DeepSeek 题目解析 API

使用示例：
  python benchmarks/benchmark_concurrency.py --target local_ocr --concurrency 5 --total 10
  python benchmarks/benchmark_concurrency.py --target local_grade --concurrency 3 --total 6
  python benchmarks/benchmark_concurrency.py --target upstream_ocr --concurrency 10 --total 20
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# 加载 .env 环境变量
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


@dataclass
class RequestMetric:
    req_id: int
    success: bool
    status_code: int
    elapsed_seconds: float
    error_msg: str = ""
    response_preview: str = ""


@dataclass
class BenchmarkSummary:
    target: str
    concurrency: int
    total_requests: int
    total_time_seconds: float
    metrics: list[RequestMetric] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for m in self.metrics if m.success)

    @property
    def fail_count(self) -> int:
        return sum(1 for m in self.metrics if not m.success)

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_requests * 100) if self.total_requests else 0.0

    @property
    def qps(self) -> float:
        return (self.total_requests / self.total_time_seconds) if self.total_time_seconds > 0 else 0.0

    @property
    def latencies(self) -> list[float]:
        return sorted([m.elapsed_seconds for m in self.metrics if m.success])

    def percentile(self, p: float) -> float:
        lats = self.latencies
        if not lats:
            return 0.0
        idx = int(len(lats) * p)
        idx = min(idx, len(lats) - 1)
        return lats[idx]

    def print_report(self) -> None:
        lats = self.latencies
        min_lat = min(lats) if lats else 0.0
        max_lat = max(lats) if lats else 0.0
        avg_lat = (sum(lats) / len(lats)) if lats else 0.0
        p50 = self.percentile(0.50)
        p90 = self.percentile(0.90)
        p95 = self.percentile(0.95)
        p99 = self.percentile(0.99)

        # 统计错误类型
        error_stats = {}
        for m in self.metrics:
            if not m.success:
                cat = f"HTTP {m.status_code}" if m.status_code else m.error_msg[:30]
                error_stats[cat] = error_stats.get(cat, 0) + 1

        print("\n" + "=" * 65)
        print("                 🎯 API 并发压测结果报告")
        print("=" * 65)
        print(f" 📌 测试目标     : {self.target}")
        print(f" 👥 并发工作线程 : {self.concurrency}")
        print(f" 📊 总请求次数   : {self.total_requests}")
        print(f" ⏱️ 压测总耗时   : {self.total_time_seconds:.2f} 秒")
        print(f" ⚡ 吞吐量 (QPS) : {self.qps:.2f} req/s")
        print("-" * 65)
        print(f" ✅ 成功请求数   : {self.success_count} / {self.total_requests} ({self.success_rate:.1f}%)")
        print(f" ❌ 失败请求数   : {self.fail_count}")
        if error_stats:
            print(" 🚨 失败原因分布 :")
            for err, count in error_stats.items():
                print(f"    • {err}: {count} 次")
        print("-" * 65)
        print(" 📈 成功响应延迟分布 (Latency Percentiles):")
        print(f"    • 最低耗时 (Min) : {min_lat:.3f} s")
        print(f"    • 平均耗时 (Avg) : {avg_lat:.3f} s")
        print(f"    • 中位数   (P50) : {p50:.3f} s")
        print(f"    • 90分位   (P90) : {p90:.3f} s")
        print(f"    • 95分位   (P95) : {p95:.3f} s")
        print(f"    • 99分位   (P99) : {max_lat:.3f} s")
        print(f"    • 最高耗时 (Max) : {max_lat:.3f} s")
        print("=" * 65 + "\n")


def send_single_request(
    req_id: int,
    target: str,
    base_url: str,
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    timeout: int = 60,
) -> RequestMetric:
    """执行单次测试请求并采集指标"""
    t0 = time.time()
    boundary = f"----BenchmarkBoundary{req_id}{int(time.time()*1000)}"

    try:
        if target in {"local_ocr", "local_grade"}:
            endpoint = "/api/ocr_test" if target == "local_ocr" else "/api/grade_photo"
            url = f"{base_url.rstrip('/')}{endpoint}"

            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="question"\r\n\r\n'
                f"PaddlePaddle/PaddleOCR-VL-1.5\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
                f"Content-Type: {media_type}\r\n\r\n"
            ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = time.time() - t0
                data = json.load(resp)
                preview = str(data)[:60]
                return RequestMetric(
                    req_id=req_id,
                    success=True,
                    status_code=resp.status,
                    elapsed_seconds=elapsed,
                    response_preview=preview,
                )

        elif target == "upstream_ocr":
            api_key = os.getenv("SILICONFLOW_API_KEY", "")
            if not api_key:
                raise ValueError("未在 .env 中找到 SILICONFLOW_API_KEY")
            url = "https://api.siliconflow.cn/v1/chat/completions"
            encoded = base64.b64encode(image_bytes).decode("ascii")

            payload = {
                "model": "PaddlePaddle/PaddleOCR-VL-1.5",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}", "detail": "high"}},
                        {"type": "text", "text": "请识别图中的所有文字和数学公式，使用 LaTeX 格式输出。"},
                    ],
                }],
                "temperature": 0,
                "max_tokens": 1024,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = time.time() - t0
                data = json.load(resp)
                text = data["choices"][0]["message"]["content"]
                return RequestMetric(
                    req_id=req_id,
                    success=True,
                    status_code=resp.status,
                    elapsed_seconds=elapsed,
                    response_preview=text[:60],
                )

        elif target == "upstream_ds":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ValueError("未在 .env 中找到 DEEPSEEK_API_KEY")
            url = "https://api.deepseek.com/chat/completions"

            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "求函数 y=x^2-4x+3 的顶点与对称轴。"}],
                "temperature": 0,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = time.time() - t0
                data = json.load(resp)
                text = data["choices"][0]["message"]["content"]
                return RequestMetric(
                    req_id=req_id,
                    success=True,
                    status_code=resp.status,
                    elapsed_seconds=elapsed,
                    response_preview=text[:60],
                )
        else:
            raise ValueError(f"未知的测试目标: {target}")

    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        err_body = e.read().decode("utf-8", errors="replace")[:100]
        return RequestMetric(
            req_id=req_id,
            success=False,
            status_code=e.code,
            elapsed_seconds=elapsed,
            error_msg=f"HTTP {e.code}: {err_body}",
        )
    except Exception as e:
        elapsed = time.time() - t0
        return RequestMetric(
            req_id=req_id,
            success=False,
            status_code=0,
            elapsed_seconds=elapsed,
            error_msg=str(e),
        )


def run_benchmark(
    target: str,
    concurrency: int,
    total: int,
    base_url: str,
    image_path: str,
    timeout: int = 60,
) -> BenchmarkSummary:
    """启动多线程并发压测"""
    img_file = Path(image_path)
    if not img_file.exists():
        # 如果指定图片不存在，生成一张测试图
        print(f"⚠️ 指定图片不存在: {image_path}，使用默认示例图")
        sample_path = Path(__file__).parent / "sample_math.jpg"
        if sample_path.exists():
            img_file = sample_path
        else:
            raise FileNotFoundError(f"找不到测试图片: {image_path}")

    image_bytes = img_file.read_bytes()
    media_type = "image/png" if img_file.suffix.lower() == ".png" else "image/jpeg"

    print("\n🚀 正在启动并发压力测试...")
    print(f"   • 测试目标: {target}")
    print(f"   • 并发数  : {concurrency} workers")
    print(f"   • 请求总数: {total} requests")
    print(f"   • 测试图片: {img_file.name} ({len(image_bytes)/1024:.1f} KB)")
    print(f"   • 超时阈值: {timeout}s\n")

    summary = BenchmarkSummary(
        target=target,
        concurrency=concurrency,
        total_requests=total,
        total_time_seconds=0.0,
    )

    t_start = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                send_single_request,
                req_id=i + 1,
                target=target,
                base_url=base_url,
                image_bytes=image_bytes,
                media_type=media_type,
                timeout=timeout,
            ): i + 1
            for i in range(total)
        }

        for future in as_completed(futures):
            req_id = futures[future]
            metric = future.result()
            summary.metrics.append(metric)
            completed_count += 1

            status_mark = "✅ 成功" if metric.success else f"❌ 失败 ({metric.error_msg[:25]})"
            print(
                f"[{completed_count:02d}/{total:02d}] 请求 #{metric.req_id:02d} -> "
                f"{status_mark} | 耗时: {metric.elapsed_seconds:.2f}s"
            )

    summary.total_time_seconds = time.time() - t_start
    summary.print_report()
    return summary


def main():
    parser = argparse.ArgumentParser(description="K12 数学 OCR 与智能批改 API 并发压测工具")
    parser.add_argument(
        "--target",
        "-t",
        choices=["local_ocr", "local_grade", "upstream_ocr", "upstream_ds"],
        default="local_ocr",
        help="压测目标: local_ocr(本地OCR), local_grade(本地整卷批改), upstream_ocr(云端VLM), upstream_ds(云端DeepSeek)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="并发工作线程数 (默认: 5)",
    )
    parser.add_argument(
        "--total",
        "-n",
        type=int,
        default=10,
        help="总请求次数 (默认: 10)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="本地服务基础 URL (默认: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--image",
        default=str(Path(__file__).parent / "sample_math.jpg"),
        help="用于测试的手写图片路径",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="单次请求超时时间(秒)",
    )

    args = parser.parse_args()
    run_benchmark(
        target=args.target,
        concurrency=args.concurrency,
        total=args.total,
        base_url=args.url,
        image_path=args.image,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
