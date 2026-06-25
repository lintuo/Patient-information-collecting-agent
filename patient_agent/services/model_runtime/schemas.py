"""统一模型接入层 schemas — 与后端实现无关，可被 FastAPI 序列化。

所有 list 字段使用 default_factory=list 避免可变默认值问题。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# 元数据
# =============================================================================
class ModelCallMetadata(BaseModel):
    """每次模型调用的执行元数据"""

    provider: str = "unknown"
    backend: str = "unknown"
    model: str | None = None
    device: str | None = None
    latency_ms: int | None = None
    tokens_per_second: float | None = None
    error_message: str | None = None


# =============================================================================
# Chat
# =============================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class ChatResult(BaseModel):
    text: str
    metadata: ModelCallMetadata = Field(default_factory=ModelCallMetadata)
    raw_response: dict | None = None


# =============================================================================
# 图像理解 (VLM)
# =============================================================================
class ImageAnalysisRequest(BaseModel):
    image_path: str
    prompt: str | None = None
    mime_type: str | None = None


class ImageAnalysisResult(BaseModel):
    summary: str = ""
    modality_guess: str | None = None
    visible_findings: list[str] = Field(default_factory=list)
    abnormal_findings: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    suggested_departments: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: str | None = None
    metadata: ModelCallMetadata = Field(default_factory=ModelCallMetadata)
    raw_response: dict | None = None


# =============================================================================
# 语音识别 (ASR)
# =============================================================================
class AudioTranscriptionRequest(BaseModel):
    audio_path: str
    prompt: str | None = None
    language: str | None = None


class AudioTranscriptionResult(BaseModel):
    transcript: str = ""
    language: str | None = None
    confidence: float | None = None
    metadata: ModelCallMetadata = Field(default_factory=ModelCallMetadata)
    raw_response: dict | None = None
