import logging

from patient_agent.graph.error_handling import with_node_error_handling
from patient_agent.graph.nodes.helpers import load_and_refresh_state
from patient_agent.graph.state import GraphState
from patient_agent.services.rag.department_store import search_department_candidates

logger = logging.getLogger(__name__)


@with_node_error_handling("department_retrieval")
def department_retrieval_node(state: GraphState) -> GraphState:
    case_id = state["case_id"]

    # P1-3: 使用统一的加载和刷新函数
    patient_state = load_and_refresh_state(case_id)

    candidates = search_department_candidates(patient_state, k=5)
    logger.info(f"[RAG] case_id={case_id}, found {len(candidates)} department candidates")

    return {
        **state,
        "patient_state": patient_state,
        "department_candidates": candidates,
    }