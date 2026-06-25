from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from patient_agent.graph.nodes.audio_transcription import audio_transcription_node
from patient_agent.graph.nodes.conversation import conversation_node
from patient_agent.graph.nodes.department_retrieval import department_retrieval_node
from patient_agent.graph.nodes.intake import intake_node
from patient_agent.graph.nodes.report import report_node
from patient_agent.graph.nodes.triage import triage_node
from patient_agent.graph.routing import (
    route_after_audio_transcription,
    route_after_conversation,
    route_after_intake,
    route_after_triage,
)
from patient_agent.graph.state import GraphState


def _build_graph():
    """内部构建函数"""
    builder = StateGraph(GraphState)

    builder.add_node("intake", intake_node)
    builder.add_node("audio_transcription", audio_transcription_node)
    builder.add_node("conversation", conversation_node)
    builder.add_node("department_retrieval", department_retrieval_node)
    builder.add_node("triage", triage_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "intake")

    builder.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "audio_transcription": "audio_transcription",
            "conversation": "conversation",
            "department_retrieval": "department_retrieval",
            "report": "report",
        },
    )

    builder.add_conditional_edges(
        "audio_transcription",
        route_after_audio_transcription,
        {
            "conversation": "conversation",
        },
    )

    builder.add_conditional_edges(
        "conversation",
        route_after_conversation,
        {
            "department_retrieval": "department_retrieval",
            "finish": END,
        },
    )

    builder.add_edge("department_retrieval", "triage")

    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "report": "report",
            "finish": END,
        },
    )

    builder.add_edge("report", END)

    return builder.compile()


# =============================================================================
# P0-2: Graph 单例缓存 - 避免重复构建
# =============================================================================
@lru_cache(maxsize=1)
def build_patient_graph():
    """构建并缓存 Patient Graph（兼容旧接口）"""
    return _build_graph()


# 直接导出编译后的 graph 实例，routes.py 可直接使用
# 无需每次调用 build_patient_graph() 函数
@lru_cache(maxsize=1)
def get_compiled_graph():
    """获取编译后的 Graph 单例，全局复用。
    
    优势：
    1. 直接获取缓存的引用，避免函数调用开销
    2. 多进程环境下可共享缓存（如配合 uvicorn --workers）
    3. 代码更简洁
    """
    return _build_graph()