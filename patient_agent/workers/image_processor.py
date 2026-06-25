"""异步图片分析处理器

处理流程：
1. intake_node / upload 收到图片 → 创建 ImageJob（status=pending）
2. conversation_agent 通过 analyze_uploaded_images 工具触发处理
3. 本模块在后台线程中调用 VisionService，完成后更新 ImageJob 和 image_findings
4. 若图片非医疗相关，返回提醒，不加入 facts

外部调用：
    from patient_agent.workers.image_processor import process_pending_images
    process_pending_images(case_id)   # 同步调用（后台线程自动处理）
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from patient_agent.domain.rules import refresh_case_status
from patient_agent.services.vision.factory import get_vision_service, VisionResult
from patient_agent.storage.repository import get_repository

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision_worker")


def _process_single_image(case_id: str, job: dict) -> VisionResult:
    """在后台线程中执行单张图片分析"""
    vision = get_vision_service()
    result = vision.analyze(job["image_path"], job["job_id"], job["file_id"])
    return case_id, job["job_id"], result


def process_pending_images(case_id: str, background: bool = True):
    """处理 case 所有 pending/running 状态的 ImageJob。

    Args:
        case_id: 病例 ID
        background: True=后台线程执行（推荐），False=同步阻塞
    """
    def _run():
        repo = get_repository()
        state = repo.load(case_id)

        pending_jobs = [
            job for job in state.image_jobs
            if job.status in ("pending", "running")
        ]
        if not pending_jobs:
            return

        # 标记为 running，避免重复处理
        for job in pending_jobs:
            job.status = "running"
        repo.save(state)

        for job in pending_jobs:
            try:
                result = get_vision_service().analyze(
                    job.image_path, job.job_id, job.file_id
                )
                job.status = "done"
                job.finding = result.finding

                if result.success and result.is_medical:
                    state.image_findings.append(result.finding)
                elif result.success and not result.is_medical:
                    logger.info(f"Non-medical image skipped for {case_id}: {job.file_id}")
                else:
                    job.error = result.error
                    job.status = "failed"

            except Exception as e:
                logger.exception(f"Image processing failed for job {job.job_id}: {e}")
                job.status = "failed"
                job.error = str(e)

        state = refresh_case_status(state)
        repo.save(state)
        logger.info(f"Image processing completed for case {case_id}")

    if background:
        _executor.submit(_run)
    else:
        _run()


def process_pending_images_and_wait(case_id: str) -> list[VisionResult]:
    """同步等待所有 pending images 处理完毕，返回结果列表（用于测试）"""
    repo = get_repository()
    state = repo.load(case_id)
    pending_jobs = [
        {"job_id": job.job_id, "file_id": job.file_id, "image_path": job.image_path}
        for job in state.image_jobs
        if job.status in ("pending", "running")
    ]
    if not pending_jobs:
        return []

    results = []
    for job_dict in pending_jobs:
        result = get_vision_service().analyze(
            job_dict["image_path"], job_dict["job_id"], job_dict["file_id"]
        )
        results.append(result)

        # 回写 ImageJob
        state = repo.load(case_id)
        for job in state.image_jobs:
            if job.job_id == job_dict["job_id"]:
                job.status = "done"
                job.finding = result.finding
                if result.success and result.is_medical:
                    state.image_findings.append(result.finding)
                elif not result.success:
                    job.status = "failed"
                    job.error = result.error
                break
        state = refresh_case_status(state)
        repo.save(state)

    return results


if __name__ == "__main__":
    import sys
    case_id = sys.argv[1] if len(sys.argv) > 1 else "demo"
    print(f"Processing images for case: {case_id}")
    results = process_pending_images_and_wait(case_id)
    for r in results:
        print(f"[{r.file_id}] medical={r.is_medical} success={r.success}")
        print(f"  finding: {r.finding[:200]}")
        if r.error:
            print(f"  error: {r.error}")
