"""API 模型接入 — 调用远程 OpenAI-compatible 服务。

环境变量：
    PATIENT_AGENT_API_PROVIDER   provider 标识，默认 "openai_compatible"
    PATIENT_AGENT_API_BASE_URL   API base URL，例如 https://api.siliconflow.cn/v1/
    PATIENT_AGENT_API_KEY        API 密钥
    PATIENT_AGENT_API_CHAT_MODEL 文本对话模型
    PATIENT_AGENT_API_VISION_MODEL 多模态图像理解模型
    PATIENT_AGENT_API_ASR_MODEL  语音识别模型
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


# -----------------------------------------------------------------------------
# Vision prompt for structured JSON output
# -----------------------------------------------------------------------------
_VISION_JSON_PROMPT = (
    "你是一个医疗辅助 AI。请仔细分析这张图片，并严格以 JSON 格式回复：\n"
    "{\n"
    '  "summary": "图片内容的整体描述（中文，1-3句话）",\n'
    '  "modality_guess": "图片类型推测，例如：皮肤照片/X光片/CT/MRI/超声/报告/其他",\n'
    '  "visible_findings": ["发现1", "发现2"],\n'
    '  "abnormal_findings": ["异常发现1"],\n'
    '  "red_flags": ["需要关注的红旗信号1"],\n'
    '  "suggested_departments": ["科室1", "科室2"],\n'
    '  "limitations": ["此分析的局限性说明"],\n'
    '  "confidence": "high/medium/low"\n'
    "}\n"
    "请仅输出 JSON，不要输出其他内容。"
)


class ApiModelRuntimeClient(ModelRuntimeClient):
    """通过 OpenAI-compatible API 调用远程模型。"""

    def __init__(self) -> None:
        self.provider = os.getenv("PATIENT_AGENT_API_PROVIDER", "openai_compatible")
        self.base_url = os.getenv("PATIENT_AGENT_API_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("PATIENT_AGENT_API_KEY", "")
        self.chat_model = os.getenv("PATIENT_AGENT_API_CHAT_MODEL", "")
        self.vision_model = os.getenv("PATIENT_AGENT_API_VISION_MODEL", "")
        self.asr_model = os.getenv("PATIENT_AGENT_API_ASR_MODEL", "")
        self._client: Any = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
                timeout=float(os.getenv("PATIENT_AGENT_API_TIMEOUT", "60")),
                max_retries=int(os.getenv("PATIENT_AGENT_API_MAX_RETRIES", "2")),
            )
        return self._client

    def _make_metadata(
        self,
        model: str | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
        device: str = "remote_api",
    ) -> ModelCallMetadata:
        return ModelCallMetadata(
            provider=self.provider,
            backend="api",
            model=model or self.chat_model,
            device=device,
            latency_ms=latency_ms,
            error_message=error_message,
        )

    # -------------------------------------------------------------------------
    # chat
    # -------------------------------------------------------------------------
    def chat(self, request: ChatRequest) -> ChatResult:
        t0 = time.perf_counter()

        if not self.chat_model:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ChatResult(
                text="[错误] PATIENT_AGENT_API_CHAT_MODEL 未配置",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="PATIENT_AGENT_API_CHAT_MODEL not configured",
                ),
            )

        try:
            client = self._get_client()

            # Build message list
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
            logger.exception("API chat error")
            return ChatResult(
                text=f"[API 错误] {type(exc).__name__}: {exc}",
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

        if not self.vision_model:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ImageAnalysisResult(
                summary="[错误] PATIENT_AGENT_API_VISION_MODEL 未配置",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="PATIENT_AGENT_API_VISION_MODEL not configured",
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
            client = self._get_client()

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            mime = request.mime_type or "image/jpeg"
            prompt = request.prompt or _VISION_JSON_PROMPT

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
                max_tokens=int(os.getenv("PATIENT_AGENT_API_VISION_MAX_TOKENS", "1024")),
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            content = response.choices[0].message.content or ""
            raw = response.model_dump() if hasattr(response, "model_dump") else None

            # Try to parse JSON
            parsed: dict | None = None
            parse_error: str | None = None
            try:
                json_str = content.strip()
                if json_str.startswith("```"):
                    json_str = json_str.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                parsed = json.loads(json_str)
            except (json.JSONDecodeError, IndexError, AttributeError) as exc:
                parse_error = f"JSON parse failed: {exc}"
                logger.warning(f"Vision JSON parse error: {exc}\nRaw content: {content[:200]}")

            if parsed:
                return ImageAnalysisResult(
                    summary=parsed.get("summary", content),
                    modality_guess=parsed.get("modality_guess"),
                    visible_findings=parsed.get("visible_findings", []),
                    abnormal_findings=parsed.get("abnormal_findings", []),
                    red_flags=parsed.get("red_flags", []),
                    suggested_departments=parsed.get("suggested_departments", []),
                    limitations=parsed.get("limitations", []),
                    confidence=parsed.get("confidence"),
                    metadata=self._make_metadata(model=self.vision_model, latency_ms=latency_ms),
                    raw_response=raw,
                )
            else:
                # JSON 解析失败时，把原始文本放入 summary
                return ImageAnalysisResult(
                    summary=content,
                    modality_guess=None,
                    visible_findings=[],
                    abnormal_findings=[],
                    red_flags=[],
                    suggested_departments=[],
                    limitations=[parse_error or "JSON 解析失败，未返回结构化结果"],
                    confidence=None,
                    metadata=self._make_metadata(
                        model=self.vision_model,
                        latency_ms=latency_ms,
                        error_message=parse_error,
                    ),
                    raw_response=raw,
                )

        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.exception("API vision error")
            return ImageAnalysisResult(
                summary=f"[API 错误] {type(exc).__name__}: {exc}",
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

        if not self.asr_model:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return AudioTranscriptionResult(
                transcript="[错误] PATIENT_AGENT_API_ASR_MODEL 未配置",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="PATIENT_AGENT_API_ASR_MODEL not configured",
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
            client = self._get_client()

            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model=self.asr_model,
                    file=f,
                    language=request.language,
                    prompt=request.prompt,
                )

            latency_ms = int((time.perf_counter() - t0) * 1000)

            # OpenAI transcription response
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
            # SDK 不支持 audio.transcriptions
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.warning("API ASR not supported by this endpoint")
            return AudioTranscriptionResult(
                transcript="[错误] 当前 API 不支持音频转写功能",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message="audio.transcriptions not supported by this API endpoint",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.exception("API ASR error")
            return AudioTranscriptionResult(
                transcript=f"[API 错误] {type(exc).__name__}: {exc}",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message=f"{type(exc).__name__}: {exc}",
                ),
            )

    # -------------------------------------------------------------------------
    # health_check
    # -------------------------------------------------------------------------
    def health_check(self) -> dict:
        if not self.chat_model:
            return {"ok": False, "reason": "PATIENT_AGENT_API_CHAT_MODEL not configured"}
        try:
            result = self.chat(ChatRequest(messages=[ChatMessage(role="user", content="ok")]))
            return {
                "ok": result.metadata.error_message is None,
                "backend": "api",
                "model": self.chat_model,
                "latency_ms": result.metadata.latency_ms,
                "error": result.metadata.error_message,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": "api", "error": str(exc)}
