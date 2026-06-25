import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

def get_embedding_model() -> OpenAIEmbeddings:
    return get_embeddings()

def get_embeddings() -> OpenAIEmbeddings:
    provider = os.getenv("PATIENT_AGENT_EMBEDDING_PROVIDER", "openai")
    model = os.getenv("PATIENT_AGENT_EMBEDDING_MODEL")
    api_key = os.getenv("PATIENT_AGENT_EMBEDDING_API_KEY")
    base_url = os.getenv("PATIENT_AGENT_EMBEDDING_BASE_URL")

    if provider != "openai":
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            "For DashScope / Qwen / Bailian OpenAI-compatible API, use provider=openai."
        )

    if not model:
        raise ValueError("PATIENT_AGENT_EMBEDDING_MODEL is not set")

    if not api_key:
        raise ValueError("PATIENT_AGENT_EMBEDDING_API_KEY is not set")

    if not base_url:
        raise ValueError("PATIENT_AGENT_EMBEDDING_BASE_URL is not set")

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,

        # 关键：避免 langchain_openai 把文本先转换成 token id 再发给百炼。
        # 百炼 OpenAI 兼容 embedding 接口更适合接收 str 或 list[str]。
        check_embedding_ctx_length=False,
    )