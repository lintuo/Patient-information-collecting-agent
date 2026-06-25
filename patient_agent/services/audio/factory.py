"""语音识别（ASR）服务 — 基于统一 model_runtime 层。

当前实现：
    - 优先使用 PATIENT_AGENT_ASR_BACKEND 选择后端
    - api 后端：调用远程 OpenAI-compatible ASR API
    - local_http 后端：调用本地 ASR 服务（如 Whisper 等）
    - mock 后端：返回固定模拟转写结果
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from patient_agent.services.model_runtime.base import ModelRuntimeClient

logger = logging.getLogger(__name__)
load_dotenv()

# Supported audio MIME types
SUPPORTED_AUDIO_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/mp4",
    "audio/x-m4a",
}

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".webm", ".mp4", ".m4a"}


class AudioService:
    """语音识别服务包装器，提供更友好的 API。"""

    def __init__(self) -> None:
        self._client: "ModelRuntimeClient | None" = None

    @property
    def client(self) -> "ModelRuntimeClient":
        if self._client is None:
            from patient_agent.services.model_runtime import get_audio_client

            self._client = get_audio_client()
        return self._client

    def transcribe(
        self,
        audio_path: str,
        prompt: str | None = None,
        language: str | None = None,
    ):
        """转写音频文件。"""
        from patient_agent.services.model_runtime.schemas import (
            AudioTranscriptionRequest,
        )

        req = AudioTranscriptionRequest(
            audio_path=audio_path,
            prompt=prompt,
            language=language,
        )
        return self.client.transcribe_audio(req)

    def validate_audio_file(self, audio_path: str) -> tuple[bool, str]:
        """验证音频文件是否合法。"""
        path = Path(audio_path)
        if not path.exists():
            return False, f"文件不存在: {audio_path}"

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False, f"不支持的音频格式: {ext}，支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"

        return True, "ok"

    def health_check(self) -> dict:
        """健康检查。"""
        return self.client.health_check()


# =============================================================================
# Singleton
# =============================================================================
_audio_service: AudioService | None = None


def get_audio_service() -> AudioService:
    global _audio_service
    if _audio_service is None:
        _audio_service = AudioService()
    return _audio_service


def rebuild_audio_service() -> AudioService:
    global _audio_service
    _audio_service = AudioService()
    return _audio_service
