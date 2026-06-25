"""音频转写节点 — 将 pending 音频转为文本后接入对话流程。

流程：
1. 从 PatientCaseState 找到所有 transcription_status == "pending" 的音频
2. 调用 get_audio_client().transcribe_audio() 逐个转写
3. 成功时：追加 AudioTranscript，记录 latency_ms，标记 status=done
4. 失败时：标记 status=failed，记录 error_message
5. 将转写文本合并到 GraphState.user_text，格式：[语音转写] <transcript>
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from patient_agent.domain.state import AudioTranscript, PatientCaseState
from patient_agent.graph.error_handling import with_node_error_handling
from patient_agent.graph.nodes.helpers import load_and_refresh_state, save_and_refresh_state
from patient_agent.graph.state import GraphState
from patient_agent.services.model_runtime import get_audio_client
from patient_agent.services.model_runtime.schemas import AudioTranscriptionRequest

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@with_node_error_handling("audio_transcription")
def audio_transcription_node(state: GraphState) -> GraphState:
    case_id = state["case_id"]
    patient_state = load_and_refresh_state(case_id)

    pending_attachments = [
        a for a in patient_state.audio_attachments
        if a.transcription_status == "pending"
    ]

    if not pending_attachments:
        # 没有待处理音频，直接跳过
        return {
            **state,
            "patient_state": patient_state,
        }

    logger.info(
        f"[AUDIO_TRANSCRIPTION] case_id={case_id}, "
        f"pending_count={len(pending_attachments)}"
    )

    audio_client = get_audio_client()
    transcribed_parts: list[str] = []

    for attachment in pending_attachments:
        try:
            result = audio_client.transcribe_audio(
                AudioTranscriptionRequest(
                    audio_path=attachment.path,
                    language=None,
                    prompt=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Audio transcription failed for {attachment.audio_id}")
            attachment.transcription_status = "failed"
            attachment.error_message = f"{type(exc).__name__}: {exc}"
            continue

        if result.metadata.error_message:
            logger.warning(
                f"Audio transcription error for {attachment.audio_id}: "
                f"{result.metadata.error_message}"
            )
            attachment.transcription_status = "failed"
            attachment.error_message = result.metadata.error_message
            continue

        # 成功
        attachment.transcription_status = "done"
        attachment.error_message = None

        patient_state.audio_transcripts.append(
            AudioTranscript(
                audio_id=attachment.audio_id,
                transcript=result.transcript,
                language=result.language,
                confidence=result.confidence,
                model=result.metadata.model,
                backend=result.metadata.backend,
                device=result.metadata.device,
                latency_ms=result.metadata.latency_ms,
                created_at=_now_iso(),
            )
        )

        transcribed_parts.append(result.transcript)
        logger.info(
            f"[AUDIO_TRANSCRIPTION] done audio_id={attachment.audio_id}, "
            f"latency_ms={result.metadata.latency_ms}"
        )

    patient_state = save_and_refresh_state(patient_state)

    # 合并转写文本到 user_text
    existing_text = state.get("user_text", "").strip()

    if transcribed_parts:
        asr_text = "\n".join(transcribed_parts)
        if existing_text:
            merged_text = f"{existing_text}\n\n[语音转写] {asr_text}"
        else:
            merged_text = f"[语音转写] {asr_text}"
    else:
        merged_text = existing_text

    logger.info(
        f"[AUDIO_TRANSCRIPTION] case_id={case_id}, "
        f"transcribed={len(transcribed_parts)}, "
        f"user_text_len={len(merged_text)}"
    )

    return {
        **state,
        "patient_state": patient_state,
        "user_text": merged_text,
    }
