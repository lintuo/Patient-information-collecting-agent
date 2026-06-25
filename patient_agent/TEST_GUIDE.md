# 病患信息 Agent 系统 — 使用指南

本文档说明系统的安装、启动、模型配置和测试方法。

---

## 目录

1. [系统概览](#1-系统概览)
2. [环境准备](#2-环境准备)
3. [项目结构](#3-项目结构)
4. [模型配置](#4-模型配置)
5. [启动服务](#5-启动服务)
6. [API 接口一览](#6-api-接口一览)
7. [测试用例](#7-测试用例)
8. [Gradio 前端操作](#8-gradio-前端操作)
9. [环境变量参考](#9-环境变量参考)
10. [调试与排查](#10-调试与排查)

---

## 1. 系统概览

### 技术栈

| 组件 | 技术 |
|------|------|
| 对话引擎 | LangGraph（状态机） |
| Agent | DeepAgents + tools |
| LLM | OpenAI-compatible API（可切换 mock） |
| 图像理解 | Qwen3.5-4B 本地 VLM / SiliconFlow API / Ollama |
| 语音识别 | Qwen3-ASR-1.7B 本地 ASR |
| RAG | Chroma + OpenAI Embeddings（科室知识库） |
| 后端 | FastAPI |
| 前端 | Gradio |

### 系统架构

```
患者输入（文字 / 图片 / 语音）
        │
        ▼
   intake_node  ← 提取结构化信息，写入 PatientCaseState
        │
        ▼
   [pending audio?] ──→ audio_transcription_node ←── Qwen3-ASR-1.7B
        │
        ▼
   conversation_node ← 对话 Agent（追问 missing_fields，更新 facts）
        │
   [信息足够?] ──no──→ 对话继续
        │
       yes
        │
        ▼
   department_retrieval_node ← RAG 检索候选科室
        │
        ▼
   triage_node ← 分诊 Agent，生成 TriageResult
        │
        ▼
   report_node ← 渲染 Markdown 报告
        │
        ▼
     输出报告
```

### 三种 Vision 后端

| 后端 | 配置 key | 说明 |
|------|----------|------|
| `api`（默认） | `PATIENT_AGENT_VISION_PROVIDER=api` | SiliconFlow/DashScope API |
| `ollama` | `PATIENT_AGENT_VISION_PROVIDER=ollama` | Ollama 本地 server |
| `transformers` | `PATIENT_AGENT_VISION_PROVIDER=transformers` | 本地 Qwen3.5-4B VLM（AMD ROCm） |

---

## 2. 环境准备

### 2.1 硬件要求

- **GPU**：AMD ROCm 兼容显卡（如 RX 7900 XTX、MI300X）
- **内存**：建议 32GB+
- **磁盘**：至少 30GB 用于模型文件

### 2.2 基础环境

```bash
# 克隆项目
cd /home/amd-5e046r4/Project/patient-agent-system

# 激活虚拟环境
source .venv-rocm/bin/activate

# 设置 PYTHONPATH
export PYTHONPATH=$PWD/src
```

### 2.3 初始化数据目录

```bash
mkdir -p data/cases data/reports data/uploads data/vectorstores logs
```

---

## 3. 项目结构

```
patient-agent-system/
├── src/patient_agent/
│   ├── main.py                    # FastAPI 入口
│   ├── agents/                    # Agent 定义
│   │   ├── conversation/          # 对话 Agent
│   │   └── triage/                # 分诊 Agent
│   ├── api/                      # FastAPI 路由
│   ├── graph/                    # LangGraph 工作流
│   │   └── nodes/                # 节点（intake/conversation/triage 等）
│   ├── services/
│   │   ├── model_runtime/         # 统一模型运行时（chat/vision/ASR）
│   │   │   ├── factory.py        # 后端选择工厂
│   │   │   ├── local_client.py   # local_http 后端
│   │   │   ├── mock_client.py    # mock 后端
│   │   │   └── transformers_asr.py # transformers ASR 后端
│   │   ├── vision/
│   │   │   └── factory.py        # Vision 三后端（api/ollama/transformers）
│   │   ├── rag/                  # RAG 服务
│   │   └── audio/               # 音频服务
│   ├── storage/                  # 数据持久化
│   ├── ui/
│   │   └── gradio_app.py        # Gradio 前端
│   └── workers/                  # 后台任务（图片分析）
├── tests/
│   ├── scripts/
│   │   ├── test_local_models.py  # 模型注册测试
│   │   └── run_scenarios.py      # 场景测试
│   └── scenarios/               # JSON 测试场景
├── scripts/
│   └── build_department_index.py  # 构建 RAG 知识库
├── models/                       # 本地模型文件
│   ├── Qwen3.5-4B/             # 视觉语言模型
│   └── Qwen3-ASR-1.7B/         # 语音识别模型
├── data/
│   ├── cases/                   # 病例 JSON
│   ├── uploads/                 # 上传文件
│   ├── vectorstores/            # Chroma 向量库
│   └── triage_knowledge/        # RAG 知识库原始数据
├── .env                         # 环境变量配置
└── TEST_GUIDE.md               # 本文档
```

---

## 4. 模型配置

### 4.1 模型文件位置

```bash
ls /home/amd-5e046r4/Project/models/
# 应包含 Qwen3.5-4B/  和 Qwen3-ASR-1.7B/
```

### 4.2 Vision：Qwen3.5-4B 本地 VLM

**关键依赖**：`transformers >= 5.0`（qwen3_5 支持在 2026-02-09 合并进入 transformers 5.x）

```bash
# 检查版本
.venv-rocm/bin/pip show transformers | grep Version
# transformers >= 5.0 才支持 Qwen3_5ForConditionalGeneration

# 如需升级
.venv-rocm/bin/pip install --upgrade transformers
```

**环境变量**：

```bash
export PATIENT_AGENT_VISION_PROVIDER=transformers
export PATIENT_AGENT_VISION_HF_MODEL=/home/amd-5e046r4/Project/models/Qwen3.5-4B
# device 自动检测：CUDA 可用则用 cuda:0，否则 cpu
# dtype：cuda 用 bfloat16，cpu 用 float32
```

**实现原理**（`services/vision/factory.py → TransformersVisionBackend`）：

```python
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

# 懒加载：首次 analyze() 时触发
self._processor = AutoProcessor.from_pretrained(hf_model)
self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
    hf_model, torch_dtype=torch.bfloat16, device_map="auto"
)

# 推理
inputs = self._processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
generated_ids = self._model.generate(**inputs, max_new_tokens=512)
```

### 4.3 ASR：Qwen3-ASR-1.7B 本地语音识别

**环境变量**：

```bash
export PATIENT_AGENT_ASR_BACKEND=local_asr
export PATIENT_AGENT_LOCAL_ASR_MODEL=/home/amd-5e046r4/Project/models/Qwen3-ASR-1.7B
export PATIENT_AGENT_LOCAL_ASR_DEVICE=cuda:0
export PATIENT_AGENT_LOCAL_ASR_DTYPE=bfloat16
```

**实现原理**（`services/model_runtime/transformers_asr.py → TransformersASRClient`）：加载 Qwen3-ASR-1.7B whisper 模型，处理音频文件，返回转写文本 + 置信度。

### 4.4 Chat：本地 vLLM（可选）

```bash
export PATIENT_AGENT_MODEL_BACKEND=local_http
export PATIENT_AGENT_LOCAL_BASE_URL=http://127.0.0.1:8001/v1
export PATIENT_AGENT_LOCAL_API_KEY=local
export PATIENT_AGENT_LOCAL_CHAT_MODEL=Qwen3-32B-AWQ
```

---

## 5. 启动服务

### 终端 1：FastAPI 后端

```bash
cd /home/amd-5e046r4/Project/patient-agent-system
source .venv-rocm/bin/activate
export PYTHONPATH=$PWD/src

# Vision 用本地 Qwen3.5-4B，ASR 用本地 Qwen3-ASR-1.7B
export PATIENT_AGENT_VISION_PROVIDER=transformers
export PATIENT_AGENT_VISION_HF_MODEL=/home/amd-5e046r4/Project/models/Qwen3.5-4B
export PATIENT_AGENT_ASR_BACKEND=local_asr
export PATIENT_AGENT_LOCAL_ASR_MODEL=/home/amd-5e046r4/Project/models/Qwen3-ASR-1.7B

uvicorn patient_agent.main:app --host 127.0.0.1 --port 8000 --reload
```

验证启动成功：

```bash
curl http://127.0.0.1:8000/
# → {"status":"ok","service":"patient-agent","docs":"/docs"}
```

### 终端 2：Gradio 前端

```bash
cd /home/amd-5e046r4/Project/patient-agent-system
source .venv-rocm/bin/activate
export PYTHONPATH=$PWD/src
export PATIENT_AGENT_API_BASE=http://127.0.0.1:8000

python src/patient_agent/ui/gradio_app.py
```

前端地址：`http://127.0.0.1:7860`

### 远程访问

```bash
# 本地电脑执行
ssh -L 7860:127.0.0.1:7860 -L 8000:127.0.0.1:8000 user@服务器IP
# 然后浏览器打开 http://127.0.0.1:7860
```

### 构建 RAG 知识库（首次）

```bash
python scripts/build_department_index.py
# 生成 data/vectorstores/department_chroma/
```

---

## 6. API 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/cases` | 列出所有病例 |
| GET | `/cases/{case_id}` | 获取病例完整状态 |
| DELETE | `/cases/{case_id}` | 删除病例 |
| POST | `/cases/{case_id}/turn` | 发送文字消息，触发工作流 |
| POST | `/cases/{case_id}/images` | 上传图片 |
| GET | `/cases/{case_id}/images` | 查询图片处理状态和结果 |
| POST | `/cases/{case_id}/images/{job_id}/analyze` | 手动触发单张图片分析 |
| POST | `/cases/{case_id}/audio` | 上传音频 |
| GET | `/cases/{case_id}/audio` | 查询音频转写状态和结果 |
| POST | `/cases/{case_id}/triage` | 手动触发分诊 |
| POST | `/cases/{case_id}/report` | 生成报告 |

### 调试端点（需 `DEBUG_ENDPOINTS=true`）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/debug/model-health` | 检查 LLM 连通性 |
| POST | `/debug/model-chat` | 直接对话测试 |
| POST | `/debug/conversation-agent` | 测试对话 Agent |
| GET | `/debug/departments/search/{case_id}` | 测试 RAG 检索 |
| POST | `/debug/vision-analyze` | 测试 Vision 分析 |
| GET | `/debug/model-runtime/config` | 显示当前模型运行时配置 |

---

## 7. 测试用例

### 7.1 单元测试：模型注册链路

验证 ASR / Vision 后端能正确加载，不依赖实际推理：

```bash
cd /home/amd-5e046r4/Project/patient-agent-system
source .venv-rocm/bin/activate
export PYTHONPATH=$PWD/src

# ASR 注册测试
.venv-rocm/bin/python -m patient_agent.tests.scripts.test_local_models asr

# Vision 注册测试
.venv-rocm/bin/python -m patient_agent.tests.scripts.test_local_models vision

# 全部测试
.venv-rocm/bin/python -m patient_agent.tests.scripts.test_local_models all
```

预期输出：

```
=== ASR Factory 链路测试 ===
  [OK] BACKEND_LOCAL_ASR = 'local_asr'
  [OK] get_audio_client(backend='local_asr') → TransformersASRClient
  [OK] asr_backend config = 'local_asr'
  [OK] local_asr_model  = '/home/amd-5e046r4/Project/models/Qwen3-ASR-1.7B'
✅ ASR 链路注册成功

=== Vision Factory 链路测试 ===
  [OK] VisionService(provider=transformers) → TransformersVisionBackend
  [OK] hf_model = '/home/amd-5e046r4/Project/models/Qwen3.5-4B'
  [OK] get_vision_service() → singleton
✅ Vision 链路注册成功

==================================================
结果: 2 通过, 0 失败
```

> **注意**：`health_check` 报错是因为 `qwen-asr` 包依赖旧版 `transformers==4.57.6`，与升级后的 5.12.1 存在冲突。实际 ASR 推理（`transcribe_audio`）不经过 `qwen-asr`，不受影响。如需消除此警告，卸载 `qwen-asr` 改用 transformers 原生加载 ASR 模型。

### 7.2 集成测试：完整对话流程

**测试一：基础对话 + 红旗检测**

```bash
CASE_ID="test-$(date +%s)"

# 第1轮：主诉
curl -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/turn" \
  -H "Content-Type: application/json" \
  -d '{"text": "我胸口疼，持续两个小时，疼痛7分"}'

# 第2轮：人口学信息
curl -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/turn" \
  -H "Content-Type: application/json" \
  -d '{"text": "58岁，男，有高血压病史，无过敏史"}'

# 查看状态
curl -s "http://127.0.0.1:8000/cases/${CASE_ID}" | python3 -m json.tool
```

预期：`red_flags` 包含 `chest_pain`，`status` 为 `ready_for_triage` 或仍在 `collecting`。

**清理**：`curl -X DELETE "http://127.0.0.1:8000/cases/${CASE_ID}"`

---

**测试二：完整分诊流程**

```bash
CASE_ID="test-full-$(date +%s)"

curl -s -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/turn" \
  -H "Content-Type: application/json" \
  -d '{"text": "我最近喘不上气，胸闷，走路就加重。"}'

curl -s -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/turn" \
  -H "Content-Type: application/json" \
  -d '{"text": "大概三天了，严重程度6分，无胸痛，轻微咳嗽。"}'

curl -s -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/turn" \
  -H "Content-Type: application/json" \
  -d '{"text": "45岁，女，有哮喘病史，用吸入剂控制。"}'

# 手动触发分诊
curl -s -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/triage"

# 生成报告
curl -s -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/report"
```

预期：`triage_result.risk_level` 为 `high` 或 `urgent`，`recommended_departments` 包含 `急诊科` 或 `呼吸内科`。

**清理**：`curl -X DELETE "http://127.0.0.1:8000/cases/${CASE_ID}"`

---

**测试三：图片上传与 Vision 分析**

```bash
CASE_ID="test-img-$(date +%s)"

# 上传图片（auto_analyze=true 自动触发分析）
curl -s -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/images" \
  -F "file=@src/patient_agent/tests/images/test_01——CT.jpg" \
  -F "auto_analyze=true"

# 等待分析完成（本地 Qwen3.5-4B 约需 30-60 秒）
sleep 45

# 查询分析结果
curl -s "http://127.0.0.1:8000/cases/${CASE_ID}/images"
```

预期：`image_jobs` 中 job 状态为 `done`，`finding` 字段包含图片分析描述。

**清理**：`curl -X DELETE "http://127.0.0.1:8000/cases/${CASE_ID}"`

---

**测试四：语音上传与 ASR 转写**

```bash
CASE_ID="test-audio-$(date +%s)"

# 上传音频
curl -s -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/audio" \
  -F "file=@/path/to/audio.wav"

# 触发工作流（LangGraph 自动路由到 audio_transcription_node）
curl -s -X POST "http://127.0.0.1:8000/cases/${CASE_ID}/turn" \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'

# 查询转写结果
curl -s "http://127.0.0.1:8000/cases/${CASE_ID}/audio"
```

预期：`audio_transcripts` 包含转写文本。

**清理**：`curl -X DELETE "http://127.0.0.1:8000/cases/${CASE_ID}"`

---

**测试五：场景自动化测试**

```bash
# 运行所有预设场景
.venv-rocm/bin/python -m patient_agent.tests.scripts.run_scenarios --verbose

# 运行单个场景
.venv-rocm/bin/python -m patient_agent.tests.scripts.run_scenarios \
  --scenario src/patient_agent/tests/scenarios/chest_pain.json --verbose
```

场景文件为 JSON 格式，包含 `turns`（输入序列）、`expected`（预期结果断言）。

---

## 8. Gradio 前端操作

### 界面布局

```
┌─────────────────────────────────────────────────────┐
│  病患信息 Agent 系统                                  │
├──────────────────────────────┬──────────────────────┤
│                              │  📷 图片上传与分析      │
│  对话区域                    │  [上传图片] [上传并分析] │
│  [历史消息]                  │  [刷新状态]            │
│                              │                      │
│  ┌────────────────────────┐ │  状态: pending        │
│  │ 输入框                  │ │  分析结果: ...         │
│  └────────────────────────┘ │                      │
│  [发送]                     ├──────────────────────┤
│                              │  🎤 语音上传与转写     │
│                              │  [上传音频] [上传并转写] │
│                              │                      │
│                              │  转写状态: pending     │
│                              │  转写结果: ...         │
├──────────────────────────────┴──────────────────────┤
│  病例 ID: case-xxxx   [新建/重置] [分诊] [报告]    │
├─────────────────────────────────────────────────────┤
│  JSON 状态: { ... }                                │
└─────────────────────────────────────────────────────┘
```

### 完整操作流程

1. **新建病例**：点击"新建/重置"，创建新的病例 ID
2. **文字对话**：输入症状信息，点击发送，右侧 JSON 实时更新
3. **图片上传**：右侧面板选图片 → "上传并分析" → 等状态变 done → "刷新分析结果"
4. **语音上传**：右侧面板选音频 → "上传并转写" → 等转写完成
5. **生成分诊**：信息足够后，点"分诊"（或对话中自动触发）
6. **生成报告**：点"报告"按钮
7. **查看 JSON**：右侧区域实时显示 `PatientCaseState` 完整状态

---

## 9. 环境变量参考

### Vision 后端选择

```bash
# 三选一：
PATIENT_AGENT_VISION_PROVIDER=api           # 默认，SiliconFlow API
PATIENT_AGENT_VISION_PROVIDER=ollama        # Ollama 本地
PATIENT_AGENT_VISION_PROVIDER=transformers  # 本地 Qwen3.5-4B（AMD ROCm）

# API 模式配置
PATIENT_AGENT_VISION_API_KEY=your_key
PATIENT_AGENT_VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
PATIENT_AGENT_VISION_MODEL=qwen3.5-omni-plus-2026-03-15
PATIENT_AGENT_VISION_TIMEOUT=120

# transformers 本地模式配置
PATIENT_AGENT_VISION_HF_MODEL=/home/amd-5e046r4/Project/models/Qwen3.5-4B

# Ollama 模式配置
PATIENT_AGENT_OLLAMA_BASE_URL=http://localhost:11434/v1
PATIENT_AGENT_OLLAMA_MODEL=llava:7b
```

### ASR 后端

```bash
PATIENT_AGENT_ASR_BACKEND=local_asr
PATIENT_AGENT_LOCAL_ASR_MODEL=/home/amd-5e046r4/Project/models/Qwen3-ASR-1.7B
PATIENT_AGENT_LOCAL_ASR_DEVICE=cuda:0
PATIENT_AGENT_LOCAL_ASR_DTYPE=bfloat16
```

### LLM

```bash
# mock（默认，无需配置）
PATIENT_AGENT_MODEL_BACKEND=mock

# API 模式
PATIENT_AGENT_MODEL_BACKEND=api
PATIENT_AGENT_API_BASE_URL=https://api.siliconflow.cn/v1/
PATIENT_AGENT_API_KEY=your_key
PATIENT_AGENT_API_CHAT_MODEL=deepseek-ai/DeepSeek-V4-Flash

# 本地 vLLM 模式
PATIENT_AGENT_MODEL_BACKEND=local_http
PATIENT_AGENT_LOCAL_BASE_URL=http://127.0.0.1:8001/v1
PATIENT_AGENT_LOCAL_CHAT_MODEL=Qwen3-32B-AWQ
```

### 调试

```bash
DEBUG_ENDPOINTS=true   # 启用 /debug/* 端点
ENV=development
```

---

## 10. 调试与排查

### Vision 后端加载失败

```bash
# 检查 transformers 版本
.venv-rocm/bin/python -c "import transformers; print(transformers.__version__)"
# 必须 >= 5.0

# 检查模型路径
ls /home/amd-5e046r4/Project/models/Qwen3.5-4B/

# 直接测试 Vision 服务
.venv-rocm/bin/python -c "
import os
os.environ['PATIENT_AGENT_VISION_PROVIDER'] = 'transformers'
os.environ['PATIENT_AGENT_VISION_HF_MODEL'] = '/home/amd-5e046r4/Project/models/Qwen3.5-4B'
from patient_agent.services.vision.factory import rebuild_vision_service
svc = rebuild_vision_service()
result = svc.analyze('/path/to/test.jpg')
print('success:', result.success)
print('error:', result.error)
"
```

### ASR health_check 报错

这是 `qwen-asr` 包与 transformers 5.12.1 版本冲突的已知问题，不影响实际 ASR 转写功能。转写走 `TransformersASRClient.transcribe_audio()` 直接调用 transformers，不依赖 `qwen-asr`。

如需消除警告：
```bash
.venv-rocm/bin/pip uninstall qwen-asr -y
```

### 服务无法启动

```bash
# 检查 PYTHONPATH
echo $PYTHONPATH
# 预期: /home/amd-5e046r4/Project/patient-agent-system/src

# 检查 Python 解释器
python -c "import sys; print(sys.executable)"
# 预期: .../.venv-rocm/bin/python

# 检查 transformers 可导入
python -c "from transformers import Qwen3_5ForConditionalGeneration; print('OK')"
```

### 查看实时日志

```bash
tail -f logs/patient-agent.log

# 按模块过滤
grep "vision" logs/patient-agent.log
grep "asr" logs/patient-agent.log
grep "triage" logs/patient-agent.log
```

### 调试端点返回 404

```bash
# 确认 DEBUG_ENDPOINTS=true
grep DEBUG_ENDPOINTS .env
```

---

*最后更新：2026-06-19*
