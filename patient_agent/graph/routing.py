from patient_agent.domain.rules import can_run_triage
from patient_agent.graph.state import GraphState


def _has_pending_audio(state: GraphState) -> bool:
    patient_state = state.get("patient_state")
    if patient_state is None:
        return False
    return any(
        a.transcription_status == "pending"
        for a in getattr(patient_state, "audio_attachments", [])
    )


def route_after_intake(state: GraphState) -> str:
    intent = state["intent"]

    if intent == "triage":
        return "department_retrieval"

    if intent == "report":
        return "report"

    if intent == "turn":
        patient_state = state["patient_state"]
        recommended_fields_asked = getattr(patient_state, "recommended_fields_asked", False)

        # 优先处理待转写音频
        if _has_pending_audio(state):
            return "audio_transcription"

        if state.get("auto_triage", False) and can_run_triage(patient_state):
            if not recommended_fields_asked and patient_state.recommended_fields:
                return "audio_transcription" if _has_pending_audio(state) else "conversation"
            return "department_retrieval"

        return "conversation"

    return "conversation"


def route_after_audio_transcription(state: GraphState) -> str:
    """音频转写完成后，进入对话节点（转写文本已合并到 user_text）。"""
    return "conversation"


def route_after_conversation(state: GraphState) -> str:
    patient_state = state["patient_state"]
    recommended_fields_asked = getattr(patient_state, "recommended_fields_asked", False)

    if state.get("auto_triage", False) and can_run_triage(patient_state):
        if not recommended_fields_asked and patient_state.recommended_fields:
            return "department_retrieval"
        return "department_retrieval"

    return "finish"


def route_after_triage(state: GraphState) -> str:
    if state.get("auto_report", False):
        return "report"

    return "finish"
