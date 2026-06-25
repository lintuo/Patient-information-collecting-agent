from patient_agent.domain.patch import PatientStatePatch, apply_patient_patch as apply_patch
from patient_agent.domain.rules import refresh_case_status
from patient_agent.storage.repository import get_repository
from patient_agent.workers.image_processor import process_pending_images


def get_case_state(case_id: str) -> dict:
    """Get the current patient case state by case id."""
    repo = get_repository()
    state = repo.load(case_id)
    return state.model_dump()


def get_missing_fields(case_id: str) -> dict:
    """Get required and recommended missing fields for the case."""
    repo = get_repository()
    state = repo.load(case_id)

    state = refresh_case_status(state)
    repo.save(state)

    return {
        "missing_fields": state.missing_fields,
        "recommended_fields": state.recommended_fields,
        "red_flags": state.red_flags,
        "status": state.status,
    }


def apply_patient_patch(case_id: str, patch: dict) -> dict:
    """Apply structured patient information updates to the case state.

    Use this tool when the patient provides concrete information such as
    chief complaint, symptoms, duration, severity, age, medical history,
    medications, or allergies.
    """
    repo = get_repository()
    state = repo.load(case_id)

    before_missing = list(state.missing_fields)
    before_recommended = list(getattr(state, "recommended_fields", []))

    parsed_patch = PatientStatePatch.model_validate(patch)

    state = apply_patch(state, parsed_patch)
    state = refresh_case_status(state)

    repo.save(state)

    return {
        "case_id": case_id,
        "applied_patch": parsed_patch.model_dump(exclude_none=True),
        "previous_missing_fields": before_missing,
        "current_missing_fields": state.missing_fields,
        "previous_recommended_fields": before_recommended,
        "current_recommended_fields": state.recommended_fields,
        "red_flags": state.red_flags,
        "status": state.status,
        "updated_state": state.model_dump(),
    }


def get_image_findings(case_id: str) -> list[str]:
    """Get image analysis findings already attached to the case."""
    repo = get_repository()
    state = repo.load(case_id)
    return state.image_findings


def analyze_uploaded_images(case_id: str) -> dict:
    """主动触发图片分析。

    适用于患者上传图片后，异步调用多模态大模型对图片进行理解。
    调用后立即返回（后台线程异步处理），可通过 get_image_findings 轮询结果。

    Returns:
        status: "processing" | "no_images" | "error"
        pending_count: 待处理图片数量
        message: 描述信息
    """
    repo = get_repository()
    state = repo.load(case_id)

    pending_jobs = [job for job in state.image_jobs if job.status in ("pending", "running")]

    if not pending_jobs:
        return {
            "status": "no_images",
            "pending_count": 0,
            "message": "当前没有待处理的图片，或所有图片已完成分析。",
            "image_findings": state.image_findings,
        }

    try:
        # 异步后台处理，不阻塞 agent
        process_pending_images(case_id, background=True)
        return {
            "status": "processing",
            "pending_count": len(pending_jobs),
            "message": f"已收到 {len(pending_jobs)} 张图片，正在后台分析，请稍后查询结果。",
            "image_findings": state.image_findings,
        }
    except Exception as e:
        return {
            "status": "error",
            "pending_count": len(pending_jobs),
            "message": f"启动图片分析失败：{str(e)}",
            "image_findings": state.image_findings,
        }