from typing import Literal, NotRequired, TypedDict

from patient_agent.domain.state import PatientCaseState

GraphIntent = Literal["turn", "triage", "report"]


class GraphState(TypedDict):
    case_id: str
    intent: GraphIntent

    patient_state: NotRequired[PatientCaseState]
    user_text: NotRequired[str]

    auto_triage: NotRequired[bool]
    auto_report: NotRequired[bool]

    department_candidates: NotRequired[list[dict]]

    assistant_message: NotRequired[str]
    report_path: NotRequired[str]
    errors: NotRequired[list[str]]
    node_errors: NotRequired[dict[str, dict]]  # P1-2: 各节点错误详情
    triage_blockers: NotRequired[list[str]]
    recommended_fields_asked: NotRequired[bool]  # 已追问过 recommended_fields