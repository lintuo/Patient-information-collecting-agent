"""统一模型接入层抽象接口 — 文本对话 / 图像理解 / 语音识别."""

from abc import ABC, abstractmethod

from patient_agent.services.model_runtime.schemas import (
    AudioTranscriptionRequest,
    AudioTranscriptionResult,
    ChatRequest,
    ChatResult,
    ImageAnalysisRequest,
    ImageAnalysisResult,
)


class ModelRuntimeClient(ABC):
    """统一模型运行时抽象基类。

    所有后端（API / 本地 / Mock）都必须实现此接口。
    如果某个后端不支持某项能力，应返回结构化错误或在 metadata 中记录，
    而非让系统崩溃。Mock 后端必须三项全支持。
    """

    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResult:
        """文本对话"""
        raise NotImplementedError

    @abstractmethod
    def analyze_image(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        """多模态图像理解"""
        raise NotImplementedError

    @abstractmethod
    def transcribe_audio(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResult:
        """语音识别"""
        raise NotImplementedError

    def health_check(self) -> dict:
        """健康检查，返回后端基本信息。子类可覆盖。"""
        return {"ok": True, "backend": self.__class__.__name__}
