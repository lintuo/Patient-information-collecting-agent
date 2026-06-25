# 病患信息 Agent 系统 — 技术文档

> 本文档面向工程师与技术决策者，详细说明系统的架构设计、硬件利用、技术选型及创新点。

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈总览](#2-技术栈总览)
3. [系统架构](#3-系统架构)
4. [模型与硬件利用](#4-模型与硬件利用)
5. [AMD AI MAX+ 平台利用](#5-amd-ai-max-平台利用)
6. [核心设计创新](#6-核心设计创新)
7. [关键模块详解](#7-关键模块详解)
8. [部署与配置](#8-部署与配置)

---

## 1. 项目概述

病患信息 Agent 系统是一个**多模态医疗分诊对话系统**，通过自然语言对话收集患者信息，结合图像分析和语音转写，最终给出科室分诊建议和风险评估报告。

### 1.1 核心能力

| 能力 | 说明 |
|------|------|
| 智能问诊 | LangGraph 驱动的多轮对话，自动追问缺失字段，提取结构化信息 |
| 红旗检测 | 规则 + LLM 双层识别胸痛、呼吸困难、意识障碍等高风险症状 |
| 图像分析 | 医疗影像（CT/MRI/X光）理解，支持本地 VLM 或 API 后端 |
| 语音转写 | 语音录入自动转文字，支持普通话及多语言 |
| 智能分诊 | RAG 增强的科室推荐，基于患者主诉、病史、影像多维证据 |
| 报告生成 | 结构化 Markdown 报告输出 |

---

## 2. 技术栈总览

### 2.1 框架层

| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 工作流编排 | LangGraph | 1.2.2 | 状态机驱动的对话流程控制 |
| Agent 框架 | DeepAgents | 0.6.6 | 对话 Agent，含工具调用能力 |
| API 框架 | FastAPI | 0.136.3 | RESTful 接口 |
| 前端 UI | Gradio | 6.17.3 | 交互式 Web 界面 |
| 数据验证 | Pydantic | 2.13.4 | 状态 Schema 定义 |

### 2.2 模型层

| 组件 | 技术 | 用途 |
|------|------|------|
| 本地 VLM | Qwen3.5-4B + transformers | 医疗图像理解 |
| 本地 ASR | Qwen3-ASR-1.7B + transformers | 语音转写 |
| 远程对话 | DeepSeek-V4-Flash（SiliconFlow API） | 智能对话 |
| 远程视觉 | qwen3.5-omni-plus-2026-03-15（DashScope API） | 图像分析备选 |
| 向量检索 | text-embedding-v4 + Chroma | 科室 RAG 检索 |

### 2.3 深度学习运行时

| 组件 | 版本 | 用途 |
|------|------|------|
| PyTorch | 2.12.0+rocm7.2 | 核心张量运算 |
| TorchAudio | 2.11.0+rocm7.2 | 音频预处理 |
| TorchVision | 0.27.0+rocm7.2 | 图像预处理 |
| Triton-ROCM | 3.7.0 | AMD GPU 算子优化 |
| Transformers | 5.12.1 | 本地模型加载与推理 |
| Accelerate | 1.12.0 | 模型分布式加载 |

### 2.4 依赖全貌

```
langgraph==1.2.2         langgraph-prebuilt==1.1.0     langgraph-checkpoint==4.1.1
langgraph-sdk==0.3.15    langchain==1.3.2              langchain-openai==1.2.2
langchain-anthropic==1.4.4 langchain-google-genai==4.2.4
deepagents==0.6.6        fastapi==0.136.3              uvicorn==0.48.0
pydantic==2.13.4         gradio==6.17.3               torch==2.12.0+rocm7.2
torchaudio==2.11.0+rocm7.2 torchvision==0.27.0+rocm7.2 triton-rocm==3.7.0
transformers==5.12.1     openai==2.38.0               anthropic==0.105.2
google-genai==2.7.0      jinja2==3.1.6               python-dotenv==1.2.2
pillow==12.2.0           tenacity==9.1.4               tiktoken==0.13.0
pydantic-settings==2.14.1 huggingface-hub==1.20.1     accelerate==1.12.0
librosa==0.11.0         filetype==1.2.0
```

---

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Gradio 前端                              │
│                    http://127.0.0.1:7860                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI 后端                               │
│                   http://127.0.0.1:8000                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              LangGraph 工作流引擎                         │  │
│  │                                                          │  │
│  │   START → intake → routing → [conversation] ──────────┐  │  │
│  │                           ↓                           │  │  │
│  │              [audio_transcription]                     │  │  │
│  │                           ↓                           │  │  │
│  │              [department_retrieval]                     │  │  │
│  │                           ↓                           │  │  │
│  │                   [triage]                             │  │  │
│  │                           ↓                           │  │  │
│  │                   [report] → END                       │  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                     │
│         ┌───────────────────┼───────────────────┐                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Model       │    │ Vision       │    │ RAG          │        │
│  │ Runtime     │    │ Service      │    │ (Chroma +    │        │
│  │ Factory     │    │ (三后端)      │    │  Embedding)  │        │
│  └─────────────┘    └──────────────┘    └──────────────┘        │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │ SiliconFlow │    │ Qwen3.5-4B   │    │  Chroma DB   │
  │ API         │    │ (ROCm GPU)   │    │              │
  │ (远程对话/  │    │ Qwen3-ASR-   │    │ 科室知识库    │
  │  远程视觉)  │    │ 1.7B(ROCm)  │    │              │
  └─────────────┘    └──────────────┘    └──────────────┘
```

### 3.2 LangGraph 工作流详解

系统使用 LangGraph 状态机编排完整分诊流程，包含 6 个节点：

#### 节点清单

| 节点 | 触发时机 | 核心逻辑 |
|------|----------|----------|
| `intake` | 每次请求入口 | 加载/保存病例状态，规则提取结构化字段（年龄、性别、症状等） |
| `audio_transcription` | 待转写音频存在时 | 调用 ASR 模型，合并转写文本到对话上下文 |
| `conversation` | intent="turn" 且无音频待处理 | 调用对话 Agent，追问缺失字段 |
| `department_retrieval` | 信息足够或手动触发 | 从 Chroma RAG 检索候选科室 |
| `triage` | 前序节点完成后 | 调用分诊 Agent，生成 `TriageResult` |
| `report` | auto_report=true 或手动触发 | 渲染 Markdown 报告 |

#### 路由决策

```
route_after_intake:
  intent="triage"    → department_retrieval
  intent="report"    → report
  intent="turn" + pending_audio → audio_transcription
  intent="turn" + can_run_triage → department_retrieval
  intent="turn" + else → conversation

route_after_conversation:
  can_run_triage → department_retrieval
  else → END

route_after_triage:
  auto_report=true → report
  else → END
```

### 3.3 数据状态模型

`PatientCaseState` 是贯穿全流程的核心状态 Schema：

```python
class PatientCaseState(BaseModel):
    case_id: str
    status: CaseStatus  # collecting | waiting_image | ready_for_triage | triaged | reported

    # 患者信息
    facts: PatientFacts  # age, sex, chief_complaint, symptoms, duration, severity, ...
    conversation_turns: list[dict]  # 对话历史
    conversation_summary: str

    # 图像
    uploaded_files: list[UploadedFile]
    image_jobs: list[ImageJob]       # pending | running | done | failed
    image_findings: list[str]        # 分析完成的医疗发现

    # 语音
    audio_attachments: list[AudioAttachment]
    audio_transcripts: list[AudioTranscript]

    # 字段追踪
    missing_fields: list[str]       # 分诊前必须填写的字段
    recommended_fields: list[str]   # 有助于提升分诊质量的字段
    red_flags: list[RedFlag]        # chest_pain | breathing_difficulty | ...

    # 分诊结果
    triage_result: TriageResult | None
    report_path: str | None
```

---

## 4. 模型与硬件利用

### 4.1 全部模型一览

#### 本地模型（AMD ROCm GPU）

| 模型 | 路径 | 规模 | 用途 | dtype | 设备 |
|------|------|------|------|-------|------|
| **Qwen3.5-4B** | `/home/amd-5e046r4/Project/models/Qwen3.5-4B` | 4B | 视觉语言模型（VLM） | bfloat16 | `cuda:0`（ROCm 映射为 hip:0） |
| **Qwen3-ASR-1.7B** | `/home/amd-5e046r4/Project/models/Qwen3-ASR-1.7B` | 1.7B | 语音识别 | bfloat16 | `cuda:0`（可配置） |

#### 远程 API 模型

| 模型 | 供应商 | 用途 | 调用接口 |
|------|--------|------|----------|
| **DeepSeek-V4-Flash** | SiliconFlow | 对话生成 | OpenAI-compatible |
| **qwen3.5-omni-plus-2026-03-15** | DashScope | 图像分析（API 模式） | OpenAI-compatible |
| **text-embedding-v4** | DashScope | 向量嵌入（RAG） | OpenAI-compatible |

### 4.2 VLM 推理：Qwen3.5-4B

**实现文件**：`services/vision/factory.py → TransformersVisionBackend`

**模型类**：`Qwen3_5ForConditionalGeneration`（transformers 5.x 新增，2026-02-09 合并）

**推理流程**：

```python
# 懒加载：首次 analyze() 才从磁盘加载模型（约 8GB）
self._processor = AutoProcessor.from_pretrained(hf_model)
self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
    hf_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",      # 自动层间分布到 GPU
)

# 单次推理
inputs = self._processor(text=[prompt], images=[image_path], return_tensors="pt")
inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
generated_ids = self._model.generate(**inputs, max_new_tokens=512)
result = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

**bfloat16 优势**：相比 float16，bfloat16 在 AMD CDNA 架构上具有更好的数值稳定性，且无需手动 loss scaling。

### 4.3 ASR 推理：Qwen3-ASR-1.7B

**实现文件**：`services/model_runtime/transformers_asr.py → TransformersASRClient`

**懒加载**：首次 `transcribe_audio()` 调用时从磁盘加载模型。

**输入处理**：通过 `torchaudio` + `librosa` 完成音频解码、重采样（16kHz）、归一化。

**输出**：转写文本 + 语言检测 + 置信度分数（0~1）。

### 4.4 三后端 Vision Service

```
VisionService
  ├── ApiVisionBackend        → SiliconFlow / DashScope OpenAI-compatible API
  ├── OllamaVisionBackend    → Ollama 本地 server（OpenAI-compatible）
  └── TransformersVisionBackend → Qwen3.5-4B 本地 VLM（ROCm GPU）
```

切换方式：设置 `PATIENT_AGENT_VISION_PROVIDER={api|ollama|transformers}`

---

## 5. AMD AI MAX+ 平台利用

### 5.1 ROCm 7.2 运行时栈

项目使用专为 AMD 显卡构建的 PyTorch ROCm 版本：

```
torch==2.12.0+rocm7.2        # 基础张量运算
torchaudio==2.11.0+rocm7.2  # 音频处理
torchvision==0.27.0+rocm7.2  # 图像处理
triton-rocm==3.7.0           # GPU kernel 优化（FlashAttention 等）
```

ROCm 7.2 支持 AMD gfx1105（Radeon 8060S / AI MAX+ 395）、gfx1103（Ryzen AI）以及 MI300X 等架构。

### 5.2 硬件感知代码模式

系统始终以 CUDA 接口编写代码，ROCm 自动将其映射为 HIP：

```python
# PyTorch CUDA API → ROCm 自动映射为 HIP
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda:0" else torch.float32

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    hf_model,
    torch_dtype=dtype,
    device_map="auto",  # 自动 tensor parallelism / layer placement
)
```

### 5.3 设备兼容性矩阵

| 设备 | 触发条件 | VLM dtype | ASR dtype | device_map |
|------|----------|-----------|-----------|------------|
| AMD gfx1105 (AI MAX+ 395) | `torch.cuda.is_available()` = True | bfloat16 | bfloat16 | auto |
| AMD MI300X | `torch.cuda.is_available()` = True | bfloat16 | bfloat16 | auto |
| CPU fallback | 无 GPU | float32 | float32 | — |

### 5.4 与 AMD AI MAX+ 平台的契合点

| 特性 | AMD AI MAX+ 优势 | 本项目利用方式 |
|------|-----------------|--------------|
| 大显存（最高 256GB HBM） | 可加载更大 VLM | Qwen3.5-4B 4B 参数适合本地部署 |
| ROCm 7.x 生态 | 完整 PyTorch 生态 | transformers、accelerate 等库开箱即用 |
| bfloat16 支持 | CDNA 架构原生支持 | 更高数值稳定性，适合医疗场景 |
| 多 GPU 互联 | Infinity Fabric | `device_map="auto"` 可扩展到多卡 |
| 统一内存 | CPU-GPU 共享 | 图像/音频数据无需显式拷贝 |

---

## 6. 核心设计创新

### 6.1 多后端工厂模式

**两级工厂设计**，实现各能力的独立后端切换：

```
Level 1: model_runtime/factory.py
  ├── chat  → mock / api / local_http
  ├── vision → mock / api / local_http
  └── audio → mock / local_asr

Level 2: vision/factory.py
  └── VisionService
      ├── ApiVisionBackend      (SiliconFlow / DashScope)
      ├── OllamaVisionBackend   (本地 Ollama)
      └── TransformersVisionBackend (本地 Qwen3.5-4B)
```

命名空间隔离：`get_audio_client()` 和 `get_chat_client()` 返回不同缓存实例，互不影响。

### 6.2 懒加载 + 单例缓存

| 组件 | 模式 | 触发时机 |
|------|------|----------|
| 编译后 LangGraph | `@lru_cache(maxsize=1)` | 模块导入时 |
| Vision Service | 全局 `_vision_service` 变量 | 首次 `get_vision_service()` |
| Model Runtime Client | Dict `_clients` 缓存 | 首次 `get_*_client()` |
| Qwen3.5-4B 模型 | `if self._model is None` | 首次 `analyze()` |
| Qwen3-ASR-1.7B 模型 | `if self._model is None` | 首次 `transcribe_audio()` |
| 病例 Repository | 全局 `repo` 实例 | 模块导入时 |

### 6.3 规则引擎降级

对话 Agent 内置规则兜底策略，当 LLM 不可用时自动切换：

```python
def generate_rule_based_response(state: PatientCaseState) -> str:
    if state.red_flags:
        return "您描述的情况包含需要重视的风险信号..." + f"请补充：{missing}。"
    if missing:
        return f"我已记录您的描述。为继续分诊，请补充：{missing}。"
    return "主要信息已基本收集完成，接下来可以进入分诊总结。"
```

### 6.4 多模态证据融合

图像发现和语音转写通过两条路径进入分诊：

```
路径 1：image_findings → RAG 查询 → 科室候选 → triage_node
路径 2：image_findings + audio_transcripts → TriageResult.multimodal_notes
```

`TriageResult` 记录 `used_multimodal_evidence: bool` 和 `multimodal_notes: str`，确保多模态证据可追溯。

### 6.5 异步后台图片处理

图片上传后不阻塞 API 响应：

```python
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision_worker")

def process_pending_images(case_id: str, background: bool = True):
    if background:
        _executor.submit(_run)  # 非阻塞
```

通过 `status=pending → running → done` 三态管理，防止重复处理。

---

## 7. 关键模块详解

### 7.1 Model Runtime Factory

```
services/model_runtime/factory.py
```

提供统一的模型客户端获取接口，支持后端自动解析：

```python
# 优先级：显式 backend > 环境变量 > mock
_backend = _resolve_backend(f"PATIENT_AGENT_{capability}_BACKEND",
                            fallback="PATIENT_AGENT_MODEL_BACKEND",
                            default="mock")
```

支持的 Backend 常量：
- `BACKEND_MOCK` — 开发/测试用，不调用任何模型
- `BACKEND_API` — SiliconFlow / DashScope OpenAI-compatible API
- `BACKEND_LOCAL_HTTP` — Ollama / vLLM 等本地 OpenAI-compatible 服务
- `BACKEND_LOCAL_ASR` — 本地 transformers ASR（Qwen3-ASR-1.7B）
- `BACKEND_LOCAL_PLACEHOLDER` — 占位后端

### 7.2 Vision Service 三后端

```
services/vision/factory.py
```

`VisionService` 是统一的 Vision 入口，`VisionConfig` 从环境变量读取配置：

```python
class VisionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PATIENT_AGENT_VISION_")
    provider: Literal["api", "ollama", "transformers"] = "api"
    hf_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: int = 120
    max_retries: int = 2
```

每个 Backend 实现统一的 `VisionBackend` 接口：

```python
class VisionBackend(Protocol):
    def analyze(self, image_path: str, job_id: str, filename: str) -> VisionResult
    def health_check(self) -> dict: ...
```

### 7.3 红旗检测规则

```
domain/rules.py → detect_red_flags()
```

基于关键词匹配的规则引擎，覆盖 6 大类红旗症状：

```python
RED_FLAG_KEYWORDS = {
    "chest_pain":            ["胸痛", "胸口疼", "心口疼", "心前区"],
    "breathing_difficulty":  ["喘不上气", "呼吸困难", "胸闷", "憋气"],
    "consciousness":         ["意识模糊", "昏迷", "晕厥", "嗜睡"],
    "severe_headache":       ["剧烈头痛", "炸裂样头痛", "雷击样头痛"],
    "allergic_reaction":     ["过敏", "皮疹", "瘙痒", "呼吸困难（过敏）"],
    "bleeding":              ["大出血", "呕血", "咯血", "便血（黑便）"],
}
```

同时扫描 `image_findings` 文本，实现**图像驱动的红旗升级**。

### 7.4 RAG 科室检索

```
services/rag/department_store.py → search_department_candidates()
```

从患者状态构建查询字符串，检索 Chroma 向量库：

```python
def build_department_query(state: PatientCaseState) -> str:
    parts = [
        f"主诉：{facts.chief_complaint or ''}",
        f"症状：{'、'.join(facts.symptoms)}",
        f"持续时间：{facts.duration or ''}",
        f"严重程度：{facts.severity or ''}",
        f"既往史：{'、'.join(facts.medical_history)}",
        f"红旗风险：{'、'.join(state.red_flags)}",
        f"图片分析：{'、'.join(state.image_findings)}",  # ← 图像证据进入 RAG
    ]
    return "\n".join(part for part in parts if part.strip())
```

知识库构建脚本：`scripts/build_department_index.py`，生成 `data/vectorstores/department_chroma/`。

### 7.5 节点错误处理

```
graph/error_handling.py → with_node_error_handling()
```

所有节点被装饰器包裹，单节点失败不导致整个工作流崩溃：

```python
@with_node_error_handling("intake")
def intake_node(state: GraphState) -> GraphState:
    ...

# 失败时：state.errors 追加，state.node_errors 记录详情，继续工作流
```

---

## 8. 部署与配置

### 8.1 快速启动

```bash
cd /home/amd-5e046r4/Project/patient-agent-system
source .venv-rocm/bin/activate
export PYTHONPATH=$PWD/src

# 后端
uvicorn patient_agent.main:app --host 127.0.0.1 --port 8000 --reload

# 前端（新终端）
python src/patient_agent/ui/gradio_app.py
```

### 8.2 模型配置矩阵

| 场景 | Vision Provider | ASR Backend | Chat Backend |
|------|----------------|--------------|--------------|
| 开发（无 GPU） | `api`（默认） | `mock` | `mock` |
| 本地 VLM 生产 | `transformers` | `local_asr` | `api` |
| 全 API 模式 | `api` | `api` | `api` |
| Ollama 模式 | `ollama` | `local_asr` | `local_http` |

### 8.3 关键环境变量

```bash
# Vision
PATIENT_AGENT_VISION_PROVIDER=transformers
PATIENT_AGENT_VISION_HF_MODEL=/home/amd-5e046r4/Project/models/Qwen3.5-4B

# ASR
PATIENT_AGENT_ASR_BACKEND=local_asr
PATIENT_AGENT_LOCAL_ASR_MODEL=/home/amd-5e046r4/Project/models/Qwen3-ASR-1.7B
PATIENT_AGENT_LOCAL_ASR_DEVICE=cuda:0
PATIENT_AGENT_LOCAL_ASR_DTYPE=bfloat16

# LLM
PATIENT_AGENT_MODEL_BACKEND=api
PATIENT_AGENT_API_BASE_URL=https://api.siliconflow.cn/v1/
PATIENT_AGENT_API_KEY=your_key
PATIENT_AGENT_API_CHAT_MODEL=deepseek-ai/DeepSeek-V4-Flash

# RAG
PATIENT_AGENT_EMBEDDING_MODEL=text-embedding-v4
PATIENT_AGENT_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/

# 调试
DEBUG_ENDPOINTS=true
ENV=development
```

---

*文档版本：1.0 | 最后更新：2026-06-19*
