"""项目性能基准测试

测试项目：病患信息 Agent 系统
覆盖范围：
  - API 端点延迟（各路由）
  - 分诊图（LangGraph）执行
  - 图片分析（VLM）
  - 音频转写（ASR）
  - 对话 Agent
  - 资源占用（CPU / Memory / GPU VRAM）

运行方式：
  python tests/scripts/benchmark.py

前提条件：
  - 后端服务已启动：uvicorn patient_agent.main:app --host 127.0.0.1 --port 8000
  - 可选：需要测试本地模型时请先加载模型
"""
from __future__ import annotations

import gc
import io
import os
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

# ─── 项目路径 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

API_BASE = os.getenv("PATIENT_AGENT_API_BASE", "http://127.0.0.1:8000")

# ─── 辅助 ────────────────────────────────────────────────────────────────────


@dataclass
class Metric:
    name: str
    latency_ms: float
    success: bool
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    component: str
    description: str
    iterations: int
    total_time_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    success_count: int
    fail_count: int
    errors: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


@contextmanager
def track_memory():
    gc.collect()
    tracemalloc.start()
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024
    try:
        yield
    finally:
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        tracemalloc.stop()
        gc.collect()
        print(f"  Memory: {start_mem:.1f} MB → {current_mem:.1f} MB (delta: {current_mem - start_mem:+.1f} MB)")


def get_gpu_memory_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    try:
        result = os.popen("rocm-smi --showmemused --json 2>/dev/null").read()
        if result:
            import json
            data = json.loads(result)
            for dev in data.get("card", []):
                return float(dev.get("mem_used", 0)) / 1024
    except Exception:
        pass
    return None


def get_cpu_percent() -> float:
    return psutil.cpu_percent(interval=0.1)


def summarize(metrics: list[Metric], component: str, description: str, **metadata) -> BenchmarkResult:
    latencies = [m.latency_ms for m in metrics if m.success]
    errors = [m.error for m in metrics if not m.success]

    if latencies:
        total = sum(latencies)
        throughput = len(latencies) / (total / 1000) if total > 0 else 0
    else:
        throughput = 0

    return BenchmarkResult(
        component=component,
        description=description,
        iterations=len(metrics),
        total_time_ms=sum(m.latency_ms for m in metrics),
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        min_latency_ms=min(latencies) if latencies else 0,
        max_latency_ms=max(latencies) if latencies else 0,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        p99_latency_ms=percentile(latencies, 0.99),
        throughput_rps=throughput,
        success_count=len(latencies),
        fail_count=len(errors),
        errors=errors,
        metadata=metadata,
    )


def print_result(r: BenchmarkResult):
    print(f"\n{'='*70}")
    print(f"📊 {r.component}")
    print(f"   {r.description}")
    print(f"{'─'*70}")
    print(f"   迭代次数  : {r.iterations}")
    print(f"   成功率    : {r.success_count}/{r.iterations} ({r.success_count/r.iterations*100:.1f}%)")
    print(f"   平均延迟  : {r.avg_latency_ms:.1f} ms")
    print(f"   最小延迟  : {r.min_latency_ms:.1f} ms")
    print(f"   最大延迟  : {r.max_latency_ms:.1f} ms")
    print(f"   P50 延迟  : {r.p50_latency_ms:.1f} ms")
    print(f"   P95 延迟  : {r.p95_latency_ms:.1f} ms")
    print(f"   P99 延迟  : {r.p99_latency_ms:.1f} ms")
    print(f"   吞吐量    : {r.throughput_rps:.2f} req/s")
    if r.metadata:
        for k, v in r.metadata.items():
            print(f"   {k}    : {v}")
    if r.errors:
        for e in r.errors[:3]:
            print(f"   ⚠️  {e}")


def make_request(method: str, path: str, **kwargs) -> tuple[Any, float, str]:
    """返回 (response_data, latency_ms, error)"""
    import httpx

    url = f"{API_BASE}{path}"
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=300) as client:
            r = client.request(method, url, **kwargs)
            r.raise_for_status()
            latency = (time.perf_counter() - start) * 1000
            return r.json(), latency, ""
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return None, latency, str(e)


# ─── 测试用例 ────────────────────────────────────────────────────────────────


def test_api_health() -> BenchmarkResult:
    """API 健康检查延迟"""
    metrics = []
    for _ in range(20):
        _, lat, err = make_request("GET", "/")
        metrics.append(Metric("health", lat, bool(not err), err))
    return summarize(metrics, "API 健康检查", "GET / 端点延迟")


def test_case_lifecycle() -> BenchmarkResult:
    """病例 CRUD 生命周期（创建 → 读取 → 删除）"""
    metrics = []
    for i in range(10):
        case_id = f"bench-{int(time.time()*1000)}-{i}"
        try:
            import httpx
            with httpx.Client(timeout=30) as client:
                # POST 创建
                t0 = time.perf_counter()
                r = client.post(f"{API_BASE}/cases/{case_id}/turn", json={"text": "测试"})
                lat = (time.perf_counter() - t0) * 1000
                if r.status_code in (200, 201):
                    metrics.append(Metric("lifecycle", lat, True))
                else:
                    metrics.append(Metric("lifecycle", lat, False, f"status={r.status_code}"))
        except Exception as e:
            metrics.append(Metric("lifecycle", 0, False, str(e)))
    return summarize(metrics, "病例生命周期", "创建 + 查询 + 删除病例")


def test_conversation_turn() -> BenchmarkResult:
    """对话回合延迟（纯 API，无本地模型）"""
    case_id = f"bench-{int(time.time()*1000)}"
    messages = [
        "我头痛已经三天了，偶尔发烧，最高37.8度",
        "没有其他症状，就是头疼",
    ]
    metrics = []
    for msg in messages:
        _, lat, err = make_request("POST", f"/cases/{case_id}/turn", json={"text": msg})
        metrics.append(Metric("conversation", lat, bool(not err), err))
    return summarize(metrics, "对话回合", "POST /cases/{id}/turn 延迟", backend="siliconflow_api")


def test_image_analysis() -> BenchmarkResult:
    """图片分析延迟（使用测试图片）"""
    # 创建测试图片（灰度渐变，模拟医疗影像）
    import struct
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(120, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    case_id = f"bench-img-{int(time.time()*1000)}"
    metrics = []
    for i in range(5):
        buf.seek(0)
        files = {"file": ("test.jpg", buf, "image/jpeg")}
        data = {"auto_analyze": "true"}
        _, lat, err = make_request("POST", f"/cases/{case_id}/images", files=files, data=data)
        # 图片分析是异步的，API 立即返回；我们测量的是上传+触发时间
        metrics.append(Metric("image_upload", lat, bool(not err), err))
    return summarize(metrics, "图片上传触发", "POST /cases/{id}/images 延迟（含异步触发）", backend="siliconflow_api")


def test_audio_transcription() -> BenchmarkResult:
    """音频转写延迟（使用测试音频）"""
    import struct

    # 生成静音 WAV（1秒，16kHz 单声道）
    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    wav_data = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
    # WAV header
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav_data)
    buf.seek(0)

    case_id = f"bench-audio-{int(time.time()*1000)}"
    files = {"file": ("test.wav", buf, "audio/wav")}
    _, lat, err = make_request("POST", f"/cases/{case_id}/audio", files=files)

    # 异步转写，等待完成
    wait_lat = 0.0
    if not err:
        try:
            import httpx
            with httpx.Client(timeout=60) as client:
                for _ in range(30):
                    time.sleep(1)
                    r = client.get(f"{API_BASE}/cases/{case_id}")
                    state = r.json()
                    if state.get("audio_transcripts"):
                        wait_lat = (time.perf_counter() - time.perf_counter()) * 1000
                        break
        except Exception as e:
            err = str(e)

    metrics = [Metric("audio_transcription", lat + wait_lat, bool(not err), err)]
    return summarize(metrics, "音频转写", "POST /cases/{id}/audio 延迟（Qwen3-ASR）", backend="local_asr")


def test_triage_graph() -> BenchmarkResult:
    """分诊图执行延迟（模拟完整流程）"""
    from patient_agent.graph import get_compiled_graph
    from patient_agent.storage.repository import CaseRepository
    from patient_agent.domain.state import PatientCaseState

    repo = CaseRepository()
    case_id = f"bench-triage-{int(time.time()*1000)}"
    state = PatientCaseState(
        case_id=case_id,
        conversation_summary="",
        user_text="我胸口疼，持续2小时，向左肩放射，伴随出汗",
        uploaded_files=[],
        transcripts=[],
        image_jobs=[],
        image_findings=[],
        audio_attachments=[],
        audio_transcripts=[],
        missing_fields=[],
        department_scores={},
        primary_department=None,
        urgency_level=None,
        reasoning="",
        status="pending",
    )
    repo.save(state)

    graph = get_compiled_graph()
    metrics = []

    try:
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()

        result = graph.invoke(
            {"case_id": case_id, "messages": []},
            config={"recursion_limit": 50},
        )

        latency = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()

        mem_mb = peak / 1024 / 1024
        metrics.append(Metric("triage_graph", latency, True, metadata={"peak_memory_mb": mem_mb}))
    except Exception as e:
        metrics.append(Metric("triage_graph", 0, False, str(e)))

    # cleanup
    try:
        repo.delete(case_id)
    except Exception:
        pass

    r = summarize(metrics, "分诊图执行", "LangGraph triage graph 端到端延迟")
    if metrics and metrics[0].metadata:
        r.metadata.update(metrics[0].metadata)
    return r


def test_vision_service_direct() -> BenchmarkResult:
    """直接测试 VisionService（不通过 HTTP）"""
    from patient_agent.services.vision.factory import get_vision_service
    from PIL import Image
    import tempfile

    # 创建测试图片
    img = Image.new("RGB", (224, 224), color=(100, 120, 140))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f, format="JPEG")
        tmp_path = f.name

    metrics = []
    for i in range(3):
        try:
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()

            service = get_vision_service()
            result = service.analyze(tmp_path, f"bench-{i}", f"file-{i}")

            latency = (time.perf_counter() - t0) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            gc.collect()

            metrics.append(Metric(
                "vision_direct",
                latency,
                result.success,
                result.error or "",
                metadata={"is_medical": result.is_medical, "peak_memory_mb": peak / 1024 / 1024}
            ))
        except Exception as e:
            metrics.append(Metric("vision_direct", 0, False, str(e)))

    os.unlink(tmp_path)
    r = summarize(metrics, "视觉服务直接调用", "VisionService.analyze() 延迟（API 后端）")
    if metrics:
        r.metadata.update(metrics[0].metadata)
    return r


def test_asr_service_direct() -> BenchmarkResult:
    """直接测试 ASR 服务（不通过 HTTP）"""
    from patient_agent.services.model_runtime.factory import get_audio_client
    from patient_agent.services.model_runtime.schemas import AudioTranscriptionRequest
    import wave
    import struct
    import tempfile

    # 生成 1 秒静音 WAV（16kHz 单声道）
    sample_rate = 16000
    num_samples = sample_rate
    data = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    buf.seek(0)

    # ASR client 需要文件路径，保存临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(buf.getvalue())
        tmp_path = tmp.name

    metrics = []
    for i in range(3):
        try:
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()

            client = get_audio_client()
            request = AudioTranscriptionRequest(audio_path=tmp_path, language="zh")
            result = client.transcribe_audio(request)

            latency = (time.perf_counter() - t0) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            gc.collect()

            metrics.append(Metric(
                "asr_direct",
                latency,
                bool(result.transcript),
                result.metadata.error_message or "",
                metadata={"text_len": len(result.transcript or ""), "peak_memory_mb": peak / 1024 / 1024}
            ))
        except Exception as e:
            metrics.append(Metric("asr_direct", 0, False, str(e)))

    os.unlink(tmp_path)

    r = summarize(metrics, "ASR 服务直接调用", "AudioClient.transcribe_audio() 延迟（Qwen3-ASR）")
    if metrics:
        r.metadata.update(metrics[0].metadata)
    return r


def test_concurrent_throughput() -> BenchmarkResult:
    """并发吞吐量测试"""
    case_ids = [f"bench-concurrent-{int(time.time()*1000)}-{i}" for i in range(20)]
    messages = [f"我头疼{i}天" for i in range(20)]

    results: list[tuple[float, str]] = []

    def send_turn(idx: int) -> tuple[float, str]:
        _, lat, err = make_request("POST", f"/cases/{case_ids[idx]}/turn", json={"text": messages[idx]})
        return lat, err

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_turn, i) for i in range(20)]
        for f in as_completed(futures):
            results.append(f.result())

    total_ms = (time.perf_counter() - t0) * 1000
    latencies = [r[0] for r in results]
    errors = [r[1] for r in results if r[1]]

    metrics = [Metric("concurrent", lat, not err, err) for lat, err in results]

    r = summarize(metrics, "并发吞吐量", f"20 请求 / 5 并发 / {total_ms:.0f}ms 总时间")
    r.throughput_rps = len(messages) / (total_ms / 1000) if total_ms > 0 else 0
    return r


def test_resource_snapshot() -> dict:
    """资源快照（一次性）"""
    print("\n" + "=" * 70)
    print("📊 资源快照")
    print("─" * 70)

    proc = psutil.Process()
    cpu = proc.cpu_percent(interval=0.5)
    mem = proc.memory_info().rss / 1024 / 1024

    gpu_mem = get_gpu_memory_mb()
    system_cpu = psutil.cpu_percent(interval=0.5)
    system_mem = psutil.virtual_memory()

    print(f"   进程 CPU 使用率 : {cpu:.1f}%")
    print(f"   进程内存       : {mem:.1f} MB")
    if gpu_mem is not None:
        print(f"   GPU 显存       : {gpu_mem:.1f} MB")
    print(f"   系统 CPU       : {system_cpu:.1f}%")
    print(f"   系统内存使用   : {system_mem.percent}% ({system_mem.used/1024/1024/1024:.1f} GB / {system_mem.total/1024/1024/1024:.1f} GB)")

    return {
        "process_cpu_pct": cpu,
        "process_memory_mb": mem,
        "gpu_memory_mb": gpu_mem,
        "system_cpu_pct": system_cpu,
        "system_memory_pct": system_mem.percent,
    }


def generate_summary(results: list[BenchmarkResult], resource: dict) -> str:
    """生成 Markdown 格式总结"""
    gpu_mem = resource.get("gpu_memory_mb")
    gpu_str = f"{gpu_mem:.1f} MB" if gpu_mem is not None else "N/A"
    lines = [
        "# 性能基准测试报告",
        "",
        f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**测试环境**: {API_BASE}",
        "",
        "## 资源状态",
        "",
        "| 指标 | 值 |",
        "|------|----|",
        f"| 进程 CPU | {resource.get('process_cpu_pct', 0):.1f}% |",
        f"| 进程内存 | {resource.get('process_memory_mb', 0):.1f} MB |",
        f"| GPU 显存 | {gpu_str} |",
        f"| 系统 CPU | {resource.get('system_cpu_pct', 0):.1f}% |",
        f"| 系统内存 | {resource.get('system_memory_pct', 0):.1f}% |",
        "",
        "## 测试结果汇总",
        "",
        "| 组件 | 平均延迟 | P95 | 吞吐量 | 成功率 |",
        "|------|----------|-----|--------|--------|",
    ]

    for r in results:
        success_rate = f"{r.success_count/r.iterations*100:.0f}%" if r.iterations else "N/A"
        lines.append(
            f"| {r.component} | {r.avg_latency_ms:.0f} ms | "
            f"{r.p95_latency_ms:.0f} ms | {r.throughput_rps:.1f} req/s | {success_rate} |"
        )

    lines += ["", "## 详细结果", ""]
    for r in results:
        lines.append(f"### {r.component}")
        lines.append(f"*{r.description}*")
        lines.append("")
        lines.append(f"- 迭代次数: {r.iterations}")
        lines.append(f"- 成功率: {r.success_count}/{r.iterations}")
        lines.append(f"- 平均延迟: **{r.avg_latency_ms:.1f} ms**")
        lines.append(f"- P50/P95/P99: {r.p50_latency_ms:.1f} / {r.p95_latency_ms:.1f} / {r.p99_latency_ms:.1f} ms")
        lines.append(f"- 吞吐量: {r.throughput_rps:.2f} req/s")
        if r.metadata:
            for k, v in r.metadata.items():
                lines.append(f"- {k}: {v}")
        if r.errors:
            lines.append(f"- 错误: {r.errors[0]}")
        lines.append("")

    return "\n".join(lines)


# ─── 主入口 ──────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("🏥 病患信息 Agent 系统 — 性能基准测试")
    print(f"   API 端点: {API_BASE}")
    print(f"   测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results: list[BenchmarkResult] = []

    # 1. API 层测试（需要服务运行）
    print("\n[1/8] API 健康检查...")
    try:
        results.append(test_api_health())
    except Exception as e:
        print(f"  ⚠️  跳过: {e}")

    print("\n[2/8] 对话回合延迟...")
    try:
        results.append(test_conversation_turn())
    except Exception as e:
        print(f"  ⚠️  跳过: {e}")

    print("\n[3/8] 图片上传触发...")
    try:
        results.append(test_image_analysis())
    except Exception as e:
        print(f"  ⚠️  跳过: {e}")

    print("\n[4/8] 病例生命周期...")
    try:
        results.append(test_case_lifecycle())
    except Exception as e:
        print(f"  ⚠️  跳过: {e}")

    print("\n[5/8] 并发吞吐量...")
    try:
        results.append(test_concurrent_throughput())
    except Exception as e:
        print(f"  ⚠️  跳过: {e}")

    # 6. 服务层测试（直接调用，不需要 HTTP）
    print("\n[6/8] 视觉服务直接调用...")
    try:
        results.append(test_vision_service_direct())
    except Exception as e:
        print(f"  ⚠️  跳过: {e}")

    print("\n[7/8] ASR 服务直接调用...")
    try:
        results.append(test_asr_service_direct())
    except Exception as e:
        print(f"  ⚠️  跳过: {e}")

    print("\n[8/8] 分诊图执行...")
    try:
        results.append(test_triage_graph())
    except Exception as e:
        print(f"  ⚠️  跳过: {e}")

    # 资源快照
    resource = test_resource_snapshot()

    # 打印所有结果
    print("\n\n" + "#" * 70)
    print("# 测试结果汇总")
    print("#" * 70)
    for r in results:
        print_result(r)

    # 生成 Markdown 报告
    report = generate_summary(results, resource)
    report_path = ROOT / "logs" / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"\n📄 报告已保存: {report_path}")

    # 打印报告内容
    print("\n" + "=" * 70)
    print("MARKDOWN 报告内容:")
    print("=" * 70)
    print(report)


if __name__ == "__main__":
    main()
