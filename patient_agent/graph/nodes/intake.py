from patient_agent.domain.patch import apply_patient_patch
from patient_agent.domain.rules import rough_extract_patch_from_text
from patient_agent.graph.error_handling import with_node_error_handling
from patient_agent.graph.nodes.helpers import save_and_refresh_state
from patient_agent.graph.state import GraphState


@with_node_error_handling("intake")
def intake_node(state: GraphState) -> GraphState:
    case_id = state["case_id"]
    user_text = state.get("user_text", "").strip()

    # P1-3: 使用统一的加载和刷新函数
    from patient_agent.graph.nodes.helpers import load_and_refresh_state

    patient_state = load_and_refresh_state(case_id)

    if user_text:
        patient_state.conversation_turns.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        patch = rough_extract_patch_from_text(user_text)
        patient_state = apply_patient_patch(patient_state, patch)

    # P1-3: 使用统一的保存和刷新函数
    patient_state = save_and_refresh_state(patient_state)

    return {
        **state,
        "patient_state": patient_state,
    }