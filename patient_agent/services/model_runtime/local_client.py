"""本地模型接入 — 支持 local_http 和 local_placeholder 模式。

环境变量：
    PATIENT_AGENT_LOCAL_MODE          local_http | local_placeholder
    PATIENT_AGENT_LOCAL_BASE_URL      本地 OpenAI-compatible 服务地址，例如 http://127.0.0.1:8001/v1
    PATIENT_AGENT_LOCAL_API_KEY       本地服务密钥（可选，默认 "local"）
    PATIENT_AGENT_LOCAL_CHAT_MODEL    本地对话模型名
    PATIENT_AGENT_LOCAL_VISION_MODEL 本地 VLM 模型名
    PATIENT_AGENT_LOCAL_ASR_MODEL     本地 ASR 模型名
    PATIENT_AGENT_LOCAL_DEVICE       运行设备，例如 local / cuda / hip:0 / cpu
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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
load_dotenv()


class LocalModelRuntimeClient(ModelRuntimeClient):
    """本地模型运行时，支持 local_http 和 local_placeholder 模式。"""

    def __init__(self) -> None:
        self.mode = os.getenv("PATIENT_AGENT_LOCAL_MODE", "local_http")
        self.base_url = os.getenv(
            "PATIENT_AGENT_LOCAL_BASE_URL", "http://127.0.0.1:8001/v1"
        ).rstrip("/")
        self.api_key = os.getenv("PATIENT_AGENT_LOCAL_API_KEY", "local")
        self.chat_model = os.getenv("PATIENT_AGENT_LOCAL_CHAT_MODEL", "")
        self.vision_model = os.getenv("PATIENT_AGENT_LOCAL_VISION_MODEL", "")
        self.asr_model = os.getenv("PATIENT_AGENT_LOCAL_ASR_MODEL", "")
        self.device = os.getenv("PATIENT_AGENT_LOCAL_DEVICE", "local")
        self._client: Any = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=float(os.getenv("PATIENT_AGENT_LOCAL_TIMEOUT", "120")),
                max_retries=int(os.getenv("PATIENT_AGENT_LOCAL_MAX_RETRIES", "2")),
            )
        return self._client

    def _make_metadata(
        self,
        model: str | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
    ) -> ModelCallMetadata:
        return ModelCallMetadata(
            provider="openai_compatible",
            backend=self.mode,
            model=model or self.chat_model,
            device=self.device,
            latency_ms=latency_ms,
            error_message=error_message,
        )

    def _require_http(self) -> Any:
        """Ensure local_http mode is active."""
        if self.mode != "local_http":
            raise NotImplementedError(
                f"local_placeholder 模式（{self.mode}）当前不支持此操作。"
                " 请配置 PATIENT_AGENT_LOCAL_MODE=local_http 并提供对应模型服务。"
            )
        return self._get_client()

    # -------------------------------------------------------------------------
    # chat
    # -------------------------------------------------------------------------
    def chat(self, request: ChatRequest) -> ChatResult:
        t0 = time.perf_counter()

        if self.mode == "local_placeholder":
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ChatResult(
                text="[错误] local_placeholder 模式不支持 chat，请配置 local_http 或使用 mock backend。",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="local_placeholder does not support chat",
                ),
            )

        if not self.chat_model:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ChatResult(
                text="[错误] PATIENT_AGENT_LOCAL_CHAT_MODEL 未配置",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="PATIENT_AGENT_LOCAL_CHAT_MODEL not configured",
                ),
            )

        try:
            client = self._require_http()

            messages: list[dict] = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            for msg in request.messages:
                messages.append({"role": msg.role, "content": msg.content})
            if not messages:
                messages.append({"role": "user", "content": "(空消息)"})

            kwargs: dict[str, Any] = {"model": self.chat_model, "messages": messages}
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens

            response = client.chat.completions.create(**kwargs)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            content = response.choices[0].message.content or ""
            raw = response.model_dump() if hasattr(response, "model_dump") else None

            return ChatResult(
                text=content,
                metadata=self._make_metadata(model=self.chat_model, latency_ms=latency_ms),
                raw_response=raw,
            )

        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.exception("Local chat error")
            return ChatResult(
                text=f"[本地模型错误] {type(exc).__name__}: {exc}",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message=f"{type(exc).__name__}: {exc}",
                ),
            )

    # -------------------------------------------------------------------------
    # analyze_image
    # -------------------------------------------------------------------------
    def analyze_image(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        t0 = time.perf_counter()

        if self.mode == "local_placeholder":
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ImageAnalysisResult(
                summary="[错误] local_placeholder 模式不支持图像分析，请配置 local_http 或使用 mock backend。",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="local_placeholder does not support image analysis",
                ),
            )

        if not self.vision_model:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ImageAnalysisResult(
                summary="[错误] PATIENT_AGENT_LOCAL_VISION_MODEL 未配置",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="PATIENT_AGENT_LOCAL_VISION_MODEL not configured",
                ),
            )

        image_path = Path(request.image_path)
        if not image_path.exists():
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ImageAnalysisResult(
                summary=f"[错误] 图片文件不存在: {request.image_path}",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message=f"File not found: {request.image_path}",
                ),
            )

        try:
            client = self._require_http()

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            mime = request.mime_type or "image/jpeg"
            prompt = request.prompt or (
                "请分析这张图片（医疗相关），返回 JSON：\n"
                '{"summary":"描述","modality_guess":"类型",'
                '"visible_findings":["发现"],"abnormal_findings":["异常"],'
                '"red_flags":["红旗"],"suggested_departments":["科室"],'
                '"limitations":["限制"],"confidence":"high/medium/low"}'
            )

            response = client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{image_data}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=int(os.getenv("PATIENT_AGENT_LOCAL_VISION_MAX_TOKENS", "1024")),
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            content = response.choices[0].message.content or ""
            raw = response.model_dump() if hasattr(response, "model_dump") else None

            try:
                json_str = content.strip()
                if json_str.startswith("```"):
                    json_str = json_str.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                parsed = json.loads(json_str)
                return ImageAnalysisResult(
                    summary=parsed.get("summary", content),
                    modality_guess=parsed.get("modality_guess"),
                    visible_findings=parsed.get("visible_findings", []),
                    abnormal_findings=parsed.get("abnormal_findings", []),
                    red_flags=parsed.get("red_flags", []),
                    suggested_departments=parsed.get("suggested_departments", []),
                    limitations=parsed.get("limitations", []),
                    confidence=parsed.get("confidence"),
                    metadata=self._make_metadata(
                        model=self.vision_model, latency_ms=latency_ms
                    ),
                    raw_response=raw,
                )
            except (json.JSONDecodeError, IndexError, AttributeError) as exc:
                return ImageAnalysisResult(
                    summary=content,
                    modality_guess=None,
                    visible_findings=[],
                    abnormal_findings=[],
                    red_flags=[],
                    suggested_departments=[],
                    limitations=[f"JSON 解析失败: {exc}"],
                    confidence=None,
                    metadata=self._make_metadata(
                        model=self.vision_model,
                        latency_ms=latency_ms,
                        error_message=f"JSON parse failed: {exc}",
                    ),
                    raw_response=raw,
                )

        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.exception("Local vision error")
            return ImageAnalysisResult(
                summary=f"[本地模型错误] {type(exc).__name__}: {exc}",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message=f"{type(exc).__name__}: {exc}",
                ),
            )

    # -------------------------------------------------------------------------
    # transcribe_audio
    # -------------------------------------------------------------------------
    def transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResult:
        t0 = time.perf_counter()

        if self.mode == "local_placeholder":
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return AudioTranscriptionResult(
                transcript="[错误] local_placeholder 模式不支持语音识别，请配置 local_http 或使用 mock backend。",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="local_placeholder does not support audio transcription",
                ),
            )

        if not self.asr_model:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return AudioTranscriptionResult(
                transcript="[错误] PATIENT_AGENT_LOCAL_ASR_MODEL 未配置",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="PATIENT_AGENT_LOCAL_ASR_MODEL not configured",
                ),
            )

        audio_path = Path(request.audio_path)
        if not audio_path.exists():
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return AudioTranscriptionResult(
                transcript=f"[错误] 音频文件不存在: {request.audio_path}",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message=f"Audio file not found: {request.audio_path}",
                ),
            )

        try:
            client = self._require_http()

            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model=self.asr_model,
                    file=f,
                    language=request.language,
                    prompt=request.prompt,
                )

            latency_ms = int((time.perf_counter() - t0) * 1000)
            transcript_text = response.text or ""
            raw = response.model_dump() if hasattr(response, "model_dump") else None

            return AudioTranscriptionResult(
                transcript=transcript_text,
                language=getattr(response, "language", None),
                confidence=None,
                metadata=self._make_metadata(model=self.asr_model, latency_ms=latency_ms),
                raw_response=raw,
            )

        except AttributeError:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return AudioTranscriptionResult(
                transcript="[错误] 本地服务不支持音频转写功能",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="audio.transcriptions not supported by local endpoint",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.exception("Local ASR error")
            return AudioTranscriptionResult(
                transcript=f"[本地模型错误] {type(exc).__name__}: {exc}",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message=f"{type(exc).__name__}: {exc}",
                ),
            )

    # -------------------------------------------------------------------------
    # health_check
    # -------------------------------------------------------------------------
    def health_check(self) -> dict:
        if self.mode == "local_placeholder":
            return {
                "ok": True,
                "backend": "local_placeholder",
                "note": "占位模式，尚未接入真实本地模型。后续可接 Lemonade Server / ROCm GPU / Ryzen AI NPU / ONNX Runtime。",
            }
        try:
            result = self.chat(
                ChatRequest(messages=[ChatMessage(role="user", content="ok")])
            )
            return {
                "ok": result.metadata.error_message is None,
                "backend": self.mode,
                "model": self.chat_model,
                "device": self.device,
                "latency_ms": result.metadata.latency_ms,
                "error": result.metadata.error_message,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": self.mode, "device": self.device, "error": str(exc)}
