from patient_agent.agents.conversation.agent import run_conversation_agent
from patient_agent.graph.error_handling import with_node_error_handling
from patient_agent.graph.nodes.helpers import load_and_refresh_state, save_and_refresh_state
from patient_agent.graph.state import GraphState


@with_node_error_handling("conversation")
def conversation_node(state: GraphState) -> GraphState:
    case_id = state["case_id"]
    user_text = state.get("user_text", "")

    patient_state = load_and_refresh_state(case_id)

    # 记录追问前 recommended_fields 是否非空
    recommended_was_populated = bool(patient_state.recommended_fields)

    assistant_message = run_conversation_agent(
        case_id=case_id,
        state=patient_state,
        user_text=user_text,
    )

    # Agent 可能已通过工具修改了状态，需要重新加载
    patient_state = load_and_refresh_state(case_id)

    patient_state.conversation_turns.append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )

    # 如果 missing_fields 已填完，且 recommended_fields 在本轮前有内容，
    # 说明 LLM 刚刚追问过 recommended_fields，将标记设为 True
    if recommended_was_populated and not patient_state.recommended_fields:
        patient_state.recommended_fields_asked = True

    patient_state = save_and_refresh_state(patient_state)

    return {
        **state,
        "patient_state": patient_state,
        "assistant_message": assistant_message,
    }
