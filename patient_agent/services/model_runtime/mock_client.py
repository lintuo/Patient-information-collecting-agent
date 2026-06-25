"""Mock 模型运行时 — 固定返回合理模拟结果，无需任何外部依赖。

适用于：
    - 本地开发调试（无 API key、无 GPU）
    - CI / 单元测试
    - 演示环境
"""

from __future__ import annotations

import logging
import time

from patient_agent.services.model_runtime.base import ModelRuntimeClient
from patient_agent.services.model_runtime.schemas import (
    AudioTranscriptionRequest,
    AudioTranscriptionResult,
    ChatMessage,
    ChatRequest,
    ChatResult,
    ImageAnalysisRequest,
    ImageAnalysisResult,
    ModelCallMetadata,
)

logger = logging.getLogger(__name__)


def _make_metadata(
    latency_ms: int | None = None,
    error_message: str | None = None,
) -> ModelCallMetadata:
    return ModelCallMetadata(
        provider="mock",
        backend="mock",
        model="mock",
        device="none",
        latency_ms=latency_ms,
        error_message=error_message,
    )


class MockModelRuntimeClient(ModelRuntimeClient):
    """固定返回合理模拟结果的 Mock 实现，三项能力全支持。"""

    def chat(self, request: ChatRequest) -> ChatResult:
        t0 = time.perf_counter()
        time.sleep(0.05)  # 模拟网络延迟

        user_text = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_text = msg.content
                break

        # 模拟一些与症状相关的对话
        if not user_text:
            reply = "您好，我是病患信息助手。请描述您的主要不适症状。"
        elif any(kw in user_text for kw in ["胸", "胸口", "胸痛", "疼"]):
            reply = (
                "我已记录您描述的胸部不适。请问疼痛是持续性的还是间歇性的？"
                "有没有放射到其他部位（如手臂、肩膀、后背）？"
                "有没有伴随出汗、恶心或呼吸困难等症状？"
            )
        elif any(kw in user_text for kw in ["头", "头痛", "头晕"]):
            reply = (
                "我已记录您描述的头部不适。请问头痛持续多久了？"
                "疼痛程度如何（1-10分）？有没有伴随恶心、呕吐、视物模糊或四肢无力？"
            )
        elif any(kw in user_text for kw in ["腹", "肚子", "腹痛", "胃"]):
            reply = (
                "我已记录您的腹部症状。请问具体是哪个位置疼（上腹/中腹/下腹）？"
                "疼痛是钝痛、刺痛还是绞痛？有没有恶心呕吐或大便异常？"
            )
        elif any(kw in user_text for kw in ["发", "发烧", "发热", "体温"]):
            reply = (
                "我已记录您的发热情况。请问目前体温大概是多少？"
                "有没有咳嗽、咽痛、鼻塞流涕、肌肉酸痛等伴随症状？"
                "发热持续多久了？"
            )
        elif any(kw in user_text for kw in ["皮", "皮肤", "疹", "红斑"]):
            reply = (
                "我已记录您的皮肤症状。请问皮疹出现在哪些部位？"
                "有没有瘙痒、疼痛或渗出？疹子是一直存在还是时有时无？"
            )
        else:
            reply = (
                "好的，我已记录您的信息。请问还有没有其他需要补充的？"
                "比如症状出现的时间、频率、加重或缓解的因素等。"
            )

        if request.system_prompt:
            reply = f"[系统提示]: {request.system_prompt[:50]}...\n{reply}"

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return ChatResult(
            text=reply,
            metadata=_make_metadata(latency_ms=latency_ms),
        )

    def analyze_image(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        t0 = time.perf_counter()
        time.sleep(0.08)  # 模拟模型推理延迟

        image_name = request.image_path.split("/")[-1] if request.image_path else "图片"

        return ImageAnalysisResult(
            summary=(
                f"mock 图像分析结果：图片「{image_name}」中可见皮肤红斑样改变，"
                "局部有轻微隆起，边界欠清，颜色呈淡红色。未观察到明显水疱或溃疡形成。"
            ),
            modality_guess="皮肤照片",
            visible_findings=[
                "皮肤红斑",
                "局部皮疹样改变",
                "轻微隆起，边界欠清",
                "淡红色斑片",
            ],
            abnormal_findings=[
                "局部皮肤红斑样改变（需结合临床判断）",
            ],
            red_flags=[],
            suggested_departments=[
                "皮肤科",
                "全科医学科",
            ],
            limitations=[
                "这是 mock 结果，未调用真实图像模型",
                "仅用于开发调试，不能用于临床决策",
            ],
            confidence="low",
            metadata=_make_metadata(latency_ms=int((time.perf_counter() - t0) * 1000)),
        )

    def transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResult:
        t0 = time.perf_counter()
        time.sleep(0.06)  # 模拟 ASR 延迟

        audio_name = request.audio_path.split("/")[-1] if request.audio_path else "音频"

        return AudioTranscriptionResult(
            transcript="我胸口疼，持续两个小时，疼痛大概七分。",
            language="zh",
            confidence=0.92,
            metadata=_make_metadata(
                latency_ms=int((time.perf_counter() - t0) * 1000),
            ),
        )

    def health_check(self) -> dict:
        return {
            "ok": True,
            "backend": "mock",
            "model": "mock",
            "device": "none",
            "note": "Mock backend always returns simulated results",
        }
