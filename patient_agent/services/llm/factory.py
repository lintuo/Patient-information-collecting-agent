import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import api_key


def get_chat_model() -> Any | None:
    load_dotenv()
    # 兼容两套变量命名：
    #   1. legacy: PATIENT_AGENT_MODEL_PROVIDER / PATIENT_AGENT_MODEL / PATIENT_AGENT_BASE_URL
    #   2. 当前 .env 中实际定义的: PATIENT_AGENT_MODEL_BACKEND → api/mock，
    #      PATIENT_AGENT_API_BASE_URL, PATIENT_AGENT_API_KEY, PATIENT_AGENT_API_CHAT_MODEL
    provider = os.getenv("PATIENT_AGENT_MODEL_PROVIDER")
    model = os.getenv("PATIENT_AGENT_MODEL")
    api_key = os.getenv("PATIENT_AGENT_API_KEY")
    base_url = os.getenv("PATIENT_AGENT_BASE_URL")

    # 如果 legacy 变量不存在，尝试从当前 .env 的实际命名读取
    if provider is None or provider == "":
        backend = os.getenv("PATIENT_AGENT_MODEL_BACKEND", "mock")
        if backend == "api":
            provider = "openai_compatible"
        else:
            provider = backend

    if model is None or model == "":
        model = os.getenv("PATIENT_AGENT_API_CHAT_MODEL", "mock")

    if base_url is None or base_url == "":
        base_url = os.getenv("PATIENT_AGENT_API_BASE_URL")

    if provider == "mock" or model == "mock":
        return None

    if not api_key:
        raise RuntimeError("PATIENT_AGENT_API_KEY is required when using a real model API")

    temperature = float(os.getenv("PATIENT_AGENT_TEMPERATURE", "0"))
    timeout = float(os.getenv("PATIENT_AGENT_TIMEOUT", "60"))
    max_retries = int(os.getenv("PATIENT_AGENT_MAX_RETRIES", "2"))

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }

    # OpenAI-compatible local or third-party endpoint, e.g. SiliconFlow, vLLM, Ollama-compatible proxy.
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)
