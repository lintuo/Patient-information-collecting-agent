import logging

from patient_agent.graph.error_handling import with_node_error_handling
from patient_agent.graph.nodes.helpers import load_and_refresh_state, save_and_refresh_state
from patient_agent.graph.state import GraphState
from patient_agent.reports.renderer import render_report

logger = logging.getLogger(__name__)


@with_node_error_handling("report")
def report_node(state: GraphState) -> GraphState:
    case_id = state["case_id"]

    # P1-3: 使用统一的加载和刷新函数
    patient_state = load_and_refresh_state(case_id)

    if patient_state.triage_result is None:
        return {
            **state,
            "patient_state": patient_state,
            "assistant_message": "尚未生成分诊结果，不能生成报告。请先完成分诊。",
            "errors": ["triage_result is required before rendering report"],
        }

    report_path = render_report(patient_state)
    logger.info(f"[REPORT] case_id={case_id}, report_path={report_path}")

    patient_state.report_path = report_path

    # P1-3: 使用统一的保存和刷新函数
    patient_state = save_and_refresh_state(patient_state)

    return {
        **state,
        "patient_state": patient_state,
        "report_path": report_path,
        "assistant_message": f"报告已生成：{report_path}",
    }