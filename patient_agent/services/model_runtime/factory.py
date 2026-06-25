"""统一模型运行时工厂 — 根据环境变量选择 backend 并返回 client 实例。

支持的后端：
    mock            — MockModelRuntimeClient，无需外部依赖
    api             — ApiModelRuntimeClient，调用远程 OpenAI-compatible API
    local_http      — LocalModelRuntimeClient，调用本地 OpenAI-compatible 服务
    local_asr       — TransformersASRClient，直接加载本地 Qwen3-ASR 模型
    local_placeholder — LocalModelRuntimeClient（占位模式）
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from patient_agent.services.model_runtime.base import ModelRuntimeClient

logger = logging.getLogger(__name__)
load_dotenv()

# -----------------------------------------------------------------------------
# Backend name constants
# -----------------------------------------------------------------------------
BACKEND_MOCK = "mock"
BACKEND_API = "api"
BACKEND_LOCAL_HTTP = "local_http"
BACKEND_LOCAL_ASR = "local_asr"
BACKEND_LOCAL_PLACEHOLDER = "local_placeholder"

ALL_BACKENDS = {
    BACKEND_MOCK,
    BACKEND_API,
    BACKEND_LOCAL_HTTP,
    BACKEND_LOCAL_ASR,
    BACKEND_LOCAL_PLACEHOLDER,
}

# -----------------------------------------------------------------------------
# Client instances (singleton per backend type)
# -----------------------------------------------------------------------------
_clients: dict[str, "ModelRuntimeClient"] = {}


def _build_client(backend: str) -> "ModelRuntimeClient":
    """根据 backend 字符串构建对应 client 实例。"""
    if backend == BACKEND_MOCK:
        from patient_agent.services.model_runtime.mock_client import MockModelRuntimeClient

        return MockModelRuntimeClient()

    if backend == BACKEND_API:
        from patient_agent.services.model_runtime.api_client import ApiModelRuntimeClient

        return ApiModelRuntimeClient()

    if backend == BACKEND_LOCAL_ASR:
        from patient_agent.services.model_runtime.transformers_asr import TransformersASRClient

        return TransformersASRClient()

    if backend in (BACKEND_LOCAL_HTTP, BACKEND_LOCAL_PLACEHOLDER):
        from patient_agent.services.model_runtime.local_client import LocalModelRuntimeClient

        return LocalModelRuntimeClient()

    # Fallback to mock
    logger.warning(f"Unknown backend '{backend}', falling back to mock")
    from patient_agent.services.model_runtime.mock_client import MockModelRuntimeClient

    return MockModelRuntimeClient()


def _resolve_backend(
    env_key: str,
    fallback: str | None = None,
    default: str = BACKEND_MOCK,
) -> str:
    """从环境变量解析 backend，优先级：env_key > fallback > default。"""
    val = os.getenv(env_key, "")
    if val in ALL_BACKENDS:
        return val
    if fallback:
        fb = os.getenv(fallback, "")
        if fb in ALL_BACKENDS:
            return fb
    return default


# -----------------------------------------------------------------------------
# Public factory functions
# -----------------------------------------------------------------------------

def get_model_runtime_client(backend: str | None = None) -> "ModelRuntimeClient":
    """获取统一模型运行时 client。"""
    if backend is None:
        backend = _resolve_backend("PATIENT_AGENT_MODEL_BACKEND")

    key = f"default:{backend}"
    if key not in _clients:
        _clients[key] = _build_client(backend)
        logger.info(f"ModelRuntimeClient created: backend={backend}")
    return _clients[key]


def get_chat_client(backend: str | None = None) -> "ModelRuntimeClient":
    """获取用于文本对话的 client（可与 vision/audio 共用同一 client）。"""
    if backend is None:
        backend = _resolve_backend(
            "PATIENT_AGENT_CHAT_BACKEND",
            fallback="PATIENT_AGENT_MODEL_BACKEND",
        )

    key = f"chat:{backend}"
    if key not in _clients:
        _clients[key] = _build_client(backend)
        logger.info(f"ChatClient created: backend={backend}")
    return _clients[key]


def get_vision_client(backend: str | None = None) -> "ModelRuntimeClient":
    """获取用于图像理解的 client。"""
    if backend is None:
        backend = _resolve_backend(
            "PATIENT_AGENT_VISION_BACKEND",
            fallback="PATIENT_AGENT_MODEL_BACKEND",
        )

    key = f"vision:{backend}"
    if key not in _clients:
        _clients[key] = _build_client(backend)
        logger.info(f"VisionClient created: backend={backend}")
    return _clients[key]


def get_audio_client(backend: str | None = None) -> "ModelRuntimeClient":
    """获取用于语音识别的 client。

    推荐配置（ASR 独立后端，不影响 chat/vision）：
        PATIENT_AGENT_ASR_BACKEND=local_asr
        PATIENT_AGENT_LOCAL_ASR_MODEL=/path/to/Qwen3-ASR-1.7B
    """
    if backend is None:
        backend = _resolve_backend(
            "PATIENT_AGENT_ASR_BACKEND",
            fallback="PATIENT_AGENT_MODEL_BACKEND",
        )

    key = f"audio:{backend}"
    if key not in _clients:
        _clients[key] = _build_client(backend)
        logger.info(f"AudioClient created: backend={backend}")
    return _clients[key]


def get_runtime_config() -> dict:
    """返回当前运行时配置（不含 API key），用于调试。"""
    return {
        "model_backend": os.getenv("PATIENT_AGENT_MODEL_BACKEND", BACKEND_MOCK),
        "chat_backend": os.getenv("PATIENT_AGENT_CHAT_BACKEND", ""),
        "vision_backend": os.getenv("PATIENT_AGENT_VISION_BACKEND", ""),
        "asr_backend": os.getenv("PATIENT_AGENT_ASR_BACKEND", ""),
        "api_base_url": os.getenv("PATIENT_AGENT_API_BASE_URL", ""),
        "local_asr_model": os.getenv("PATIENT_AGENT_LOCAL_ASR_MODEL", ""),
        "local_asr_device": os.getenv("PATIENT_AGENT_LOCAL_ASR_DEVICE", "cuda:0"),
    }


def clear_clients() -> None:
    """清除所有缓存的 client 实例（用于测试或配置切换）。"""
    _clients.clear()
    logger.info("ModelRuntimeClient cache cleared")
