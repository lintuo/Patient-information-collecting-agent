"""统一模型运行时服务层 — model_runtime 命名空间."""

from patient_agent.services.model_runtime.api_client import ApiModelRuntimeClient
from patient_agent.services.model_runtime.base import ModelRuntimeClient
from patient_agent.services.model_runtime.factory import (
    BACKEND_API,
    BACKEND_LOCAL_HTTP,
    BACKEND_LOCAL_PLACEHOLDER,
    BACKEND_MOCK,
    clear_clients,
    get_audio_client,
    get_chat_client,
    get_model_runtime_client,
    get_runtime_config,
    get_vision_client,
)
from patient_agent.services.model_runtime.mock_client import MockModelRuntimeClient
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

__all__ = [
    # Base
    "ModelRuntimeClient",
    # Schemas
    "ModelCallMetadata",
    "ChatMessage",
    "ChatRequest",
    "ChatResult",
    "ImageAnalysisRequest",
    "ImageAnalysisResult",
    "AudioTranscriptionRequest",
    "AudioTranscriptionResult",
    # Clients
    "ApiModelRuntimeClient",
    "MockModelRuntimeClient",
    # Factory
    "get_model_runtime_client",
    "get_chat_client",
    "get_vision_client",
    "get_audio_client",
    "get_runtime_config",
    "clear_clients",
    # Backend constants
    "BACKEND_MOCK",
    "BACKEND_API",
    "BACKEND_LOCAL_HTTP",
    "BACKEND_LOCAL_PLACEHOLDER",
]
