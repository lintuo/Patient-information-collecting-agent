"""本地 Qwen3-ASR 语音识别 — 使用 qwen-asr 包直接加载本地模型。

环境变量：
    PATIENT_AGENT_ASR_BACKEND=local_asr
    PATIENT_AGENT_LOCAL_ASR_MODEL   本地模型路径（可以是 HF repo-id 或本地目录）
    PATIENT_AGENT_LOCAL_ASR_DTYPE   数据类型（bfloat16 / float16 / float32），默认 bfloat16
    PATIENT_AGENT_LOCAL_ASR_DEVICE  设备（cuda:0 / cpu），默认 cuda:0
    PATIENT_AGENT_LOCAL_ASR_MAX_NEW_TOKENS  最大生成 token 数，默认 512
"""

from __future__ import annotations

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


def _get_dtype() -> Any:
    import torch

    dtype_str = os.getenv("PATIENT_AGENT_LOCAL_ASR_DTYPE", "bfloat16").lower()
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


class TransformersASRClient(ModelRuntimeClient):
    """直接使用 qwen-asr 加载本地 Qwen3-ASR 模型进行语音识别。

    仅实现 transcribe_audio，chat / analyze_image 返回错误。
    如需全能力，请将 model_runtime backend 设为 mock/api，
    而将 PATIENT_AGENT_ASR_BACKEND=local_asr 用于语音识别。
    """

    def __init__(self) -> None:
        self.model_path = os.getenv(
            "PATIENT_AGENT_LOCAL_ASR_MODEL",
            "/home/amd-5e046r4/Project/models/Qwen3-ASR-1.7B",
        )
        self.device = os.getenv("PATIENT_AGENT_LOCAL_ASR_DEVICE", "cuda:0")
        self.max_new_tokens = int(os.getenv("PATIENT_AGENT_LOCAL_ASR_MAX_NEW_TOKENS", "512"))
        self._model: Any = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _get_model(self) -> Any:
        if self._model is None:
            import torch
            from qwen_asr import Qwen3ASRModel

            dtype = _get_dtype()
            logger.info(
                f"[ASR] Loading Qwen3-ASR model from {self.model_path} "
                f"dtype={dtype} device={self.device}"
            )
            self._model = Qwen3ASRModel.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=self.device,
                max_new_tokens=self.max_new_tokens,
            )
            logger.info("[ASR] Model loaded successfully")
        return self._model

    def _make_metadata(
        self,
        latency_ms: int | None = None,
        error_message: str | None = None,
    ) -> ModelCallMetadata:
        return ModelCallMetadata(
            provider="qwen_asr",
            backend="local_asr",
            model=self.model_path,
            device=self.device,
            latency_ms=latency_ms,
            error_message=error_message,
        )

    # -------------------------------------------------------------------------
    # ModelRuntimeClient interface — ASR
    # -------------------------------------------------------------------------
    def transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResult:
        t0 = time.perf_counter()

        audio_path = Path(request.audio_path)
        if not audio_path.exists():
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return AudioTranscriptionResult(
                transcript=f"[错误] 音频文件不存在: {request.audio_path}",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message=f"File not found: {request.audio_path}",
                ),
            )

        try:
            model = self._get_model()

            # qwen-asr transcribe 支持：str path / URL / base64 / (np.ndarray, sr)
            results = model.transcribe(
                audio=str(audio_path),
                language=request.language,  # None = 自动语言检测
            )

            result = results[0] if isinstance(results, list) else results

            latency_ms = int((time.perf_counter() - t0) * 1000)
            transcript_text = getattr(result, "text", "") or ""
            language = getattr(result, "language", None)

            # 置信度字段在不同版本中可能叫 different names
            confidence: float | None = None
            for attr in ("confidence", "score", "avg_logprob"):
                val = getattr(result, attr, None)
                if val is not None:
                    confidence = float(val)
                    break

            logger.info(
                f"[ASR] done path={audio_path.name} lang={language} "
                f"text_len={len(transcript_text)} latency_ms={latency_ms}"
            )

            return AudioTranscriptionResult(
                transcript=transcript_text,
                language=language,
                confidence=confidence,
                metadata=self._make_metadata(latency_ms=latency_ms),
            )

        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.exception("Qwen3-ASR transcription error")
            return AudioTranscriptionResult(
                transcript=f"[ASR 错误] {type(exc).__name__}: {exc}",
                metadata=self._make_metadata(
                    latency_ms=latency_ms,
                    error_message=f"{type(exc).__name__}: {exc}",
                ),
            )

    # -------------------------------------------------------------------------
    # ModelRuntimeClient interface — not implemented
    # -------------------------------------------------------------------------
    def chat(self, request: ChatRequest) -> ChatResult:
        t0 = time.perf_counter()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return ChatResult(
            text="[错误] TransformersASRClient 不支持 chat，请使用 api 或 mock backend。",
            metadata=self._make_metadata(
                latency_ms=latency_ms,
                error_message="TransformersASRClient does not support chat",
            ),
        )

    def analyze_image(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        t0 = time.perf_counter()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return ImageAnalysisResult(
            summary="[错误] TransformersASRClient 不支持图像分析，请使用 api 或 mock backend。",
            metadata=self._make_metadata(
                latency_ms=latency_ms,
                error_message="TransformersASRClient does not support image analysis",
            ),
        )

    def health_check(self) -> dict:
        try:
            self._get_model()
            return {
                "ok": True,
                "backend": "local_asr",
                "model": self.model_path,
                "device": self.device,
            }
        except Exception as exc:
            return {"ok": False, "backend": "local_asr", "error": str(exc)}
