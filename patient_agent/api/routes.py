from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from patient_agent.api.config import is_debug_enabled
from patient_agent.api.schemas import TurnRequest
from patient_agent.graph import get_compiled_graph
from patient_agent.storage.repository import get_repository

from fastapi import Query

from patient_agent.storage.repository import CaseRepository
from patient_agent.services.rag.department_store import (
    build_department_query,
    search_department_candidates,
)
from patient_agent.workflow.intake import save_uploaded_file, attach_file_to_state
from patient_agent.workers.image_processor import process_pending_images
from patient_agent.domain.rules import refresh_case_status
router = APIRouter()


@router.get("/")
def root():
    return {
        "status": "ok",
        "service": "patient-agent",
        "docs": "/docs",
    }


@router.get("/cases")
def list_cases():
    repo = get_repository()
    return repo.list_case_ids()


@router.get("/cases/{case_id}")
def get_case(case_id: str):
    repo = get_repository()
    return repo.load(case_id)


@router.delete("/cases/{case_id}")
def delete_case(case_id: str):
    repo = get_repository()
    repo.delete(case_id)
    return {"deleted": case_id}


@router.post("/cases/{case_id}/turn")
def submit_turn(case_id: str, req: TurnRequest):
    # P0-2: 使用缓存的 graph 实例，避免重复构建
    graph = get_compiled_graph()

    result = graph.invoke(
        {
            "case_id": case_id,
            "intent": "turn",
            "user_text": req.text,
            "auto_triage": req.auto_triage,
            "auto_report": req.auto_report,
        }
    )

    patient_state = result["patient_state"]

    return {
        "case_id": case_id,
        "status": patient_state.status,
        "assistant_message": result.get("assistant_message"),
        "missing_fields": patient_state.missing_fields,
        "recommended_fields": getattr(patient_state, "recommended_fields", []),
        "red_flags": patient_state.red_flags,
        "triage_result": (
            patient_state.triage_result.model_dump()
            if patient_state.triage_result
            else None
        ),
        "report_path": patient_state.report_path,
        "errors": result.get("errors", []),
        "triage_blockers": result.get("triage_blockers", []),
    }


@router.post("/cases/{case_id}/images")
async def upload_image(
    case_id: str,
    file: UploadFile = File(...),
    auto_analyze: bool = True,
):
    """上传图片到病例。

    图片会保存到 data/uploads/{case_id}/，并自动创建 ImageJob（pending 状态）。
    若 auto_analyze=True，立即在后台异步触发多模态分析。
    """
    import base64

    content = await file.read()
    content_type = file.content_type or "image/jpeg"

    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    uploaded = save_uploaded_file(
        case_id=case_id,
        filename=file.filename or "image",
        content_type=content_type,
        content=content,
    )

    repo = get_repository()
    state = repo.load(case_id)
    state = attach_file_to_state(state, uploaded)
    state = refresh_case_status(state)
    repo.save(state)

    if auto_analyze:
        process_pending_images(case_id, background=True)

    return {
        "case_id": case_id,
        "file_id": uploaded.file_id,
        "content_type": uploaded.content_type,
        "status": "pending",
        "auto_analyze": auto_analyze,
        "message": (
            f"图片已上传，ID={uploaded.file_id}。"
            f"{'正在后台分析中。' if auto_analyze else '可稍后通过 GET /cases/{case_id}/images 轮询结果。'}"
        ),
    }


@router.get("/cases/{case_id}/images")
def get_image_jobs(case_id: str):
    """查询病例所有图片的处理状态"""
    repo = get_repository()
    state = repo.load(case_id)
    return {
        "case_id": case_id,
        "uploaded_files": [
            {
                "file_id": f.file_id,
                "original_name": f.original_name,
                "content_type": f.content_type,
            }
            for f in state.uploaded_files
        ],
        "image_jobs": [
            {
                "job_id": j.job_id,
                "file_id": j.file_id,
                "status": j.status,
                "finding": j.finding,
                "vision_finding": j.vision_finding,
                "error": j.error,
            }
            for j in state.image_jobs
        ],
        "image_findings": state.image_findings,
    }


@router.post("/cases/{case_id}/images/{job_id}/analyze")
def trigger_image_analysis(case_id: str, job_id: str):
    """手动触发某张图片的异步分析"""
    repo = get_repository()
    state = repo.load(case_id)

    job = next((j for j in state.image_jobs if j.job_id == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"ImageJob {job_id} not found")

    process_pending_images(case_id, background=True)
    return {
        "case_id": case_id,
        "job_id": job_id,
        "status": "processing",
        "message": "图片分析已触发，结果将通过 get_image_findings 查询。",
    }


# =============================================================================
# Audio upload
# =============================================================================

@router.post("/cases/{case_id}/audio")
async def upload_audio(
    case_id: str,
    file: UploadFile = File(...),
):
    """上传音频文件到病例。

    音频保存到 data/uploads/{case_id}/audio/，状态初始化为 transcription_status=pending。
    调用后音频进入 PatientCaseState.audio_attachments，随后通过 /cases/{case_id}/turn
    触发 LangGraph 时自动执行 ASR 转写。
    """
    content = await file.read()
    content_type = file.content_type or "audio/wav"

    if not content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are accepted")

    from patient_agent.storage.media import attach_audio_to_state, save_audio_file
    from patient_agent.domain.rules import refresh_case_status

    attachment = save_audio_file(
        case_id=case_id,
        filename=file.filename or "audio",
        content_type=content_type,
        content=content,
    )

    repo = get_repository()
    state = repo.load(case_id)
    state = attach_audio_to_state(state, attachment)
    state = refresh_case_status(state)
    repo.save(state)

    return {
        "case_id": case_id,
        "audio_id": attachment.audio_id,
        "filename": attachment.filename,
        "content_type": attachment.content_type,
        "transcription_status": attachment.transcription_status,
        "message": (
            f"音频已上传，audio_id={attachment.audio_id}。"
            "调用 /cases/{case_id}/turn 时将自动执行语音转写。"
        ),
    }


@router.get("/cases/{case_id}/audio")
def get_audio_jobs(case_id: str):
    """查询病例所有音频的处理状态和转写结果"""
    repo = get_repository()
    state = repo.load(case_id)
    return {
        "case_id": case_id,
        "audio_attachments": [
            {
                "audio_id": a.audio_id,
                "filename": a.filename,
                "content_type": a.content_type,
                "uploaded_at": a.uploaded_at,
                "transcription_status": a.transcription_status,
                "error_message": a.error_message,
            }
            for a in state.audio_attachments
        ],
        "audio_transcripts": [
            {
                "audio_id": t.audio_id,
                "transcript": t.transcript,
                "language": t.language,
                "confidence": t.confidence,
                "model": t.model,
                "backend": t.backend,
                "device": t.device,
                "latency_ms": t.latency_ms,
                "created_at": t.created_at,
                "error_message": t.error_message,
            }
            for t in state.audio_transcripts
        ],
    }


@router.post("/cases/{case_id}/triage")
def run_triage(case_id: str):
    # P0-2: 使用缓存的 graph 实例
    graph = get_compiled_graph()

    result = graph.invoke(
        {
            "case_id": case_id,
            "intent": "triage",
        }
    )

    return result["patient_state"]


@router.post("/cases/{case_id}/report")
def create_report(case_id: str):
    # P0-2: 使用缓存的 graph 实例
    graph = get_compiled_graph()

    result = graph.invoke(
        {
            "case_id": case_id,
            "intent": "report",
        }
    )

    if result.get("errors"):
        return {
            "case_id": case_id,
            "errors": result["errors"],
        }

    return {
        "case_id": case_id,
        "report_path": result.get("report_path"),
    }


@router.get("/debug/model-health")
def debug_model_health():
    """检查 LLM 模型健康状态。
    
    P0-3: 仅在 DEBUG_ENDPOINTS=true 时启用。
    """
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )
    
    from patient_agent.services.llm.factory import get_chat_model

    model = get_chat_model()

    if model is None:
        return {
            "ok": True,
            "mode": "mock",
            "message": "No real model configured. get_chat_model() returned None.",
        }

    try:
        response = model.invoke("请只回复 ok")
        content = getattr(response, "content", str(response))

        return {
            "ok": True,
            "mode": "real",
            "response": content,
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": "real",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

from patient_agent.api.schemas import DebugChatRequest


@router.post("/debug/model-chat")
def debug_model_chat(req: DebugChatRequest):
    """直接与 LLM 对话（仅用于调试）。
    
    P0-3: 仅在 DEBUG_ENDPOINTS=true 时启用。
    """
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )
    
    from patient_agent.services.llm.factory import get_chat_model

    model = get_chat_model()

    if model is None:
        return {
            "ok": True,
            "mode": "mock",
            "response": f"mock response for: {req.prompt}",
        }

    try:
        response = model.invoke(req.prompt)
        content = getattr(response, "content", str(response))

        return {
            "ok": True,
            "mode": "real",
            "response": content,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

@router.post("/debug/conversation-agent")
def debug_conversation_agent(req: DebugChatRequest):
    """测试对话 Agent（仅用于调试）。
    
    P0-3: 仅在 DEBUG_ENDPOINTS=true 时启用。
    """
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )
    
    from patient_agent.agents.conversation.agent import run_conversation_agent
    from patient_agent.domain.rules import refresh_case_status
    from patient_agent.storage.repository import get_repository

    case_id = "debug-agent"
    repo = get_repository()
    state = repo.load(case_id)

    state.facts.chief_complaint = req.prompt
    state = refresh_case_status(state)
    repo.save(state)

    try:
        message = run_conversation_agent(
            case_id=case_id,
            state=state,
            user_text=req.prompt,
        )
        return {
            "ok": True,
            "assistant_message": message,
            "state": repo.load(case_id).model_dump(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

@router.get("/debug/departments/search/{case_id}")
def debug_department_search(case_id: str, k: int = Query(default=5, ge=1, le=10)):
    """调试科室检索功能。

    P0-3: 仅在 DEBUG_ENDPOINTS=true 时启用。
    """
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )

    repo = CaseRepository()
    patient_state = repo.load(case_id)

    query = build_department_query(patient_state)
    candidates = search_department_candidates(patient_state, k=k)

    return {
        "case_id": case_id,
        "query": query,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


@router.post("/debug/vision-analyze")
def debug_vision_analyze(image_path: str = Query(..., description="图片文件绝对路径")):
    """直接测试多模态视觉分析（仅用于调试）。

    P0-3: 仅在 DEBUG_ENDPOINTS=true 时启用。
    """
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )

    from patient_agent.services.vision.factory import get_vision_service

    service = get_vision_service()
    result = service.analyze(image_path)

    return {
        "provider": service.config.provider,
        "model": service.config.model,
        "file_id": result.file_id,
        "success": result.success,
        "is_medical": result.is_medical,
        "finding": result.finding,
        "error": result.error,
    }


# =============================================================================
# Unified model-runtime debug endpoints
# =============================================================================

class _DebugChatRequest(BaseModel):
    text: str
    system_prompt: str | None = None


@router.get("/debug/model-runtime/config")
def debug_model_runtime_config():
    """返回当前 model-runtime 的 backend 配置（不含 API key）。"""
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )

    from patient_agent.services.model_runtime import get_runtime_config

    return get_runtime_config()


@router.post("/debug/model-runtime/chat")
def debug_model_runtime_chat(req: _DebugChatRequest):
    """测试统一 model-runtime 的文本对话能力。"""
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )

    from patient_agent.services.model_runtime import (
        ChatMessage,
        ChatRequest,
        get_chat_client,
    )

    client = get_chat_client()
    result = client.chat(
        ChatRequest(
            messages=[ChatMessage(role="user", content=req.text)],
            system_prompt=req.system_prompt,
        )
    )

    return {
        "text": result.text,
        "metadata": result.metadata.model_dump(),
    }


@router.post("/debug/model-runtime/image")
async def debug_model_runtime_image(
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
):
    """测试统一 model-runtime 的图像理解能力（multipart/form-data）。"""
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )

    import tempfile
    from pathlib import Path

    from patient_agent.services.model_runtime import (
        ImageAnalysisRequest,
        get_vision_client,
    )

    # Save uploaded file to a temp location
    suffix = Path(file.filename or "upload").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        client = get_vision_client()
        result = client.analyze_image(
            ImageAnalysisRequest(
                image_path=tmp_path,
                prompt=prompt,
                mime_type=file.content_type,
            )
        )
        return {
            "summary": result.summary,
            "modality_guess": result.modality_guess,
            "visible_findings": result.visible_findings,
            "abnormal_findings": result.abnormal_findings,
            "red_flags": result.red_flags,
            "suggested_departments": result.suggested_departments,
            "limitations": result.limitations,
            "confidence": result.confidence,
            "metadata": result.metadata.model_dump(),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/debug/model-runtime/audio")
async def debug_model_runtime_audio(
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
    language: str | None = Form(None),
):
    """测试统一 model-runtime 的语音识别能力（multipart/form-data）。"""
    if not is_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug endpoints are disabled in this environment"
        )

    import tempfile
    from pathlib import Path

    from patient_agent.services.model_runtime import (
        AudioTranscriptionRequest,
        get_audio_client,
    )

    # Save uploaded file to a temp location
    suffix = Path(file.filename or "upload").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        client = get_audio_client()
        result = client.transcribe_audio(
            AudioTranscriptionRequest(
                audio_path=tmp_path,
                prompt=prompt,
                language=language,
            )
        )
        return {
            "transcript": result.transcript,
            "language": result.language,
            "confidence": result.confidence,
            "metadata": result.metadata.model_dump(),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
