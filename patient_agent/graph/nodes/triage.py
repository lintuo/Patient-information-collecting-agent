import logging

from patient_agent.agents.triage.agent import run_triage_agent
from patient_agent.domain.rules import get_triage_blockers
from patient_agent.graph.error_handling import with_node_error_handling
from patient_agent.graph.nodes.helpers import load_and_refresh_state, save_and_refresh_state
from patient_agent.graph.state import GraphState

logger = logging.getLogger(__name__)


@with_node_error_handling("triage")
def triage_node(state: GraphState) -> GraphState:
    case_id = state["case_id"]

    # P1-3: 使用统一的加载和刷新函数
    patient_state = load_and_refresh_state(case_id)

    blockers = get_triage_blockers(patient_state)

    if blockers:
        return {
            **state,
            "patient_state": patient_state,
            "triage_blockers": blockers,
            "assistant_message": (
                "目前信息还不足以生成分诊建议。"
                f"请先补充：{', '.join(patient_state.missing_fields)}。"
            ),
            "errors": blockers,
        }

    department_candidates = state.get("department_candidates", [])
    logger.info(f"[TRIAGE] case_id={case_id}, received department_candidates: {len(department_candidates)}")

    triage_result = run_triage_agent(
        patient_state,
        department_candidates=department_candidates,
    )

    patient_state.triage_result = triage_result

    # P1-3: 使用统一的保存和刷新函数
    patient_state = save_and_refresh_state(patient_state)

    return {
        **state,
        "patient_state": patient_state,
        "assistant_message": triage_result.summary,
    }