"""多模态视觉分析服务

支持三种后端，按优先级自动选择：
1. SiliconFlow API  — OpenAI 兼容格式（qwen3.5-omni 等）
2. Ollama 本地模型  — 兼容 OpenAI 接口的 Ollama server
3. transformers     — 直接用 transformers 库加载本地 VLM（Qwen3.5-4B / AMD ROCm GPU）

使用环境变量切换后端，详见 .env 示例。
"""
import base64
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


# =============================================================================
# 配置结构
# =============================================================================
@dataclass
class VisionConfig:
    """多模态服务配置"""
    provider: str = "api"                      # "api" | "ollama" | "transformers"
    model: str = "qwen3.5-omni-plus-2026-03-15"

    # --- API 后端 ---
    api_key: str | None = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
    timeout: int = 120
    max_retries: int = 2

    # --- Ollama 后端 ---
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llava:7b"

    # --- transformers 本地后端 ---
    hf_model: str = "/home/amd-5e046r4/Project/models/Qwen3.5-4B"
    device: str = "cuda"                       # "cuda" | "cpu"（运行时动态转为 hip:0）
    thinking: bool = False                      # 是否启用 Qwen3.5 思维模式，默认关闭

    # --- 分析提示词 ---
    medical_prompt: str = (
        "你是一个医疗辅助 AI。请仔细分析这张图片。\n"
        "1. 判断这张图片是否为医疗相关图片（检查报告、影像片子、皮肤照片、伤口照片等）。\n"
        "2. 如果是医疗图片，请用结构化方式描述你看到的关键信息：\n"
        "   - 图片类型（检查报告/影像/CT/MRI/皮肤/伤口/其他）\n"
        "   - 主要发现（报告中的异常指标、影像中的可疑病灶、皮肤病变描述等）\n"
        "   - 任何需要关注的异常项\n"
        "3. 如果不是医疗图片，请明确告知并提醒用户上传医疗相关图片。\n"
        "4. 不要做诊断，仅描述客观可见内容。\n"
        "请用中文回复。"
    )


# =============================================================================
# 结果结构
# =============================================================================
@dataclass
class VisionResult:
    job_id: str
    file_id: str
    success: bool
    is_medical: bool
    finding: str
    error: str | None = None
    raw_response: Any = None


# =============================================================================
# 后端抽象
# =============================================================================
class VisionBackend(ABC):
    @abstractmethod
    def analyze(self, image_path: str, prompt: str) -> VisionResult:
        raise NotImplementedError


# =============================================================================
# 后端 1：SiliconFlow / DashScope / OpenAI 兼容 API
# =============================================================================
class ApiVisionBackend(VisionBackend):
    def __init__(self, config: VisionConfig):
        self.config = config
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def analyze(self, image_path: str, prompt: str) -> VisionResult:
        from openai import APIError

        client = self._get_client()
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            is_medical = self._detect_medical(content)
            return VisionResult(
                job_id="", file_id="", success=True,
                is_medical=is_medical, finding=content, raw_response=response
            )
        except APIError as e:
            logger.error(f"API vision error: {e}")
            return VisionResult(
                job_id="", file_id="", success=False,
                is_medical=False, finding="", error=str(e)
            )

    @staticmethod
    def _detect_medical(text: str) -> bool:
        non_medical_keywords = ["不是医疗图片", "非医疗", "无关医疗", "不是医学"]
        return not any(kw in text for kw in non_medical_keywords)

    def _detect_medical_safe(self, text: str) -> bool:
        if not text:
            return False
        return self._detect_medical(text)


# =============================================================================
# 后端 2：Ollama（OpenAI 兼容格式）
# =============================================================================
class OllamaVisionBackend(VisionBackend):
    def __init__(self, config: VisionConfig):
        self.config = config
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.config.ollama_base_url,
                api_key="ollama",
            )
        return self._client

    def analyze(self, image_path: str, prompt: str) -> VisionResult:
        from openai import APIError

        client = self._get_client()
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        try:
            response = client.chat.completions.create(
                model=self.config.ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            is_medical = self._detect_medical_safe(content)
            return VisionResult(
                job_id="", file_id="", success=True,
                is_medical=is_medical, finding=content
            )
        except APIError as e:
            logger.error(f"Ollama vision error: {e}")
            return VisionResult(
                job_id="", file_id="", success=False,
                is_medical=False, finding="", error=str(e)
            )


# =============================================================================
# 后端 3：transformers 本地 VLM（Qwen3.5-4B / AMD ROCm GPU）
# =============================================================================
class TransformersVisionBackend(VisionBackend):
    """直接使用 transformers 库加载本地 Qwen3.5-4B 进行视觉理解。

    支持 AMD ROCm GPU（torch 会自动识别 /dev/kfd，无需手动指定 hip:0）。
    首次调用时模型从本地路径加载到 GPU，后续调用复用同一实例。
    """

    def __init__(self, config: VisionConfig):
        self.config = config
        self._processor: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        """懒加载：首次调用时将模型和 processor 加载到 GPU。"""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda:0" else torch.float32

        logger.info(
            f"[Vision] Loading Qwen3.5-4B from {self.config.hf_model} "
            f"device={device} dtype={dtype}"
        )

        self._processor = AutoProcessor.from_pretrained(self.config.hf_model)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.config.hf_model,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda:0" else None,
        )
        if device != "cuda:0":
            self._model = self._model.to(device)

        logger.info(f"[Vision] Model loaded: {self.config.hf_model}")

    def analyze(self, image_path: str, prompt: str) -> VisionResult:
        try:
            self._ensure_loaded()
            import torch
            from PIL import Image as PILImage

            raw_image = PILImage.open(image_path).convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": prompt},
                ],
            }]

            # Qwen3.5-4B 使用 apply_chat_template 构建输入，再用 processor 处理
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Qwen3_5Processor.__call__ 返回 dict，不再是 (image_dict, video_dict) 元组
            inputs = self._processor(
                text=[text],
                images=[raw_image],
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            content = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            is_medical = self._detect_medical_safe(content)
            logger.info(f"[Vision] done path={Path(image_path).name} len={len(content)}")
            return VisionResult(
                job_id="", file_id="", success=True,
                is_medical=is_medical, finding=content
            )

        except Exception as e:
            logger.exception("[Vision] Transformers backend error")
            return VisionResult(
                job_id="", file_id="", success=False,
                is_medical=False, finding="", error=f"{type(e).__name__}: {e}"
            )


# =============================================================================
# VisionService 主入口
# =============================================================================
class VisionService:
    """统一的多模态视觉分析服务，按配置自动选择后端"""

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or self._build_config()
        self._backend: VisionBackend | None = None

    def _build_config(self) -> VisionConfig:
        provider = os.getenv("PATIENT_AGENT_VISION_PROVIDER", "api").lower()

        if provider == "transformers":
            return VisionConfig(
                provider="transformers",
                hf_model=os.getenv(
                    "PATIENT_AGENT_VISION_HF_MODEL",
                    "/home/amd-5e046r4/Project/models/Qwen3.5-4B",
                ),
                device="cuda",
                thinking=os.getenv("PATIENT_AGENT_VISION_THINKING", "false").lower() != "true",
            )
        elif provider == "ollama":
            return VisionConfig(
                provider="ollama",
                ollama_base_url=os.getenv("PATIENT_AGENT_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                ollama_model=os.getenv("PATIENT_AGENT_OLLAMA_MODEL", "llava:7b"),
            )
        else:  # api (default)
            return VisionConfig(
                provider="api",
                model=os.getenv("PATIENT_AGENT_VISION_MODEL", "qwen3.5-omni-plus-2026-03-15"),
                api_key=os.getenv("PATIENT_AGENT_VISION_API_KEY"),
                base_url=os.getenv("PATIENT_AGENT_VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/"),
                timeout=int(os.getenv("PATIENT_AGENT_VISION_TIMEOUT", "120")),
                max_retries=int(os.getenv("PATIENT_AGENT_VISION_MAX_RETRIES", "2")),
            )

    @property
    def backend(self) -> VisionBackend:
        if self._backend is None:
            match self.config.provider:
                case "transformers":
                    self._backend = TransformersVisionBackend(self.config)
                case "ollama":
                    self._backend = OllamaVisionBackend(self.config)
                case _:
                    self._backend = ApiVisionBackend(self.config)
            logger.info(f"VisionService backend: {self.config.provider} / {self.config.model}")
        return self._backend

    def analyze(self, image_path: str, job_id: str = "", file_id: str = "") -> VisionResult:
        """分析单张图片，返回 VisionResult"""
        image_file = Path(image_path)
        if not image_file.exists():
            return VisionResult(
                job_id=job_id, file_id=file_id, success=False,
                is_medical=False, finding="", error=f"File not found: {image_path}"
            )
        result = self.backend.analyze(image_path, self.config.medical_prompt)
        result.job_id = job_id
        result.file_id = file_id
        return result

    def analyze_multiple(self, image_jobs: list[dict]) -> list[VisionResult]:
        """批量分析多张图片"""
        return [self.analyze(job["image_path"], job["job_id"], job["file_id"])
                for job in image_jobs if job.get("status") in ("pending", "running")]


# =============================================================================
# 单例
# =============================================================================
_vision_service: VisionService | None = None


def get_vision_service() -> VisionService:
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service


def rebuild_vision_service() -> VisionService:
    global _vision_service
    _vision_service = VisionService()
    return _vision_service
