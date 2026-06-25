from typing import Any
import logging
import os
from openai import APIError, RateLimitError
from deepagents import create_deep_agent

from patient_agent.agents.conversation.prompts import CONVERSATION_SYSTEM_PROMPT
from patient_agent.agents.conversation.tools import conversation_tools
from patient_agent.domain.state import PatientCaseState
from patient_agent.services.llm.factory import get_chat_model

logger = logging.getLogger(__name__)

# =============================================================================
# P0-1: Agent 单例缓存 - 避免每次请求重复创建 Agent 实例
# =============================================================================
_cached_conversation_agent = None


def get_conversation_agent():
    """获取 Conversation Agent 单例，全局复用。
    
    优势：
    1. 避免每次请求都重新创建 Agent 实例
    2. 模型加载、工具绑定、系统提示编译只执行一次
    3. 显著降低延迟和 API 调用开销
    """
    global _cached_conversation_agent
    
    if _cached_conversation_agent is not None:
        return _cached_conversation_agent
    
    model = get_chat_model()
    
    if model is None:
        _cached_conversation_agent = None
        return None
    
    _cached_conversation_agent = create_deep_agent(
        model=model,
        tools=conversation_tools,
        system_prompt=CONVERSATION_SYSTEM_PROMPT,
    )
    
    logger.info("Conversation Agent 实例已创建并缓存")
    return _cached_conversation_agent


def rebuild_conversation_agent():
    """强制重建 Agent 实例（用于模型切换等场景）"""
    global _cached_conversation_agent
    _cached_conversation_agent = None
    return get_conversation_agent()

def extract_last_text(result: Any) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []

    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content:
            return str(content)

        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])

    return "请继续描述您的主要不适、持续时间和严重程度。"


def build_conversation_agent():
    """兼容旧接口，内部调用单例获取方法"""
    return get_conversation_agent()


def run_conversation_agent(
    case_id: str,
    state: PatientCaseState,
    user_text: str,
) -> str:
    # P0-1: 使用单例获取 Agent，而非每次创建新实例
    agent = get_conversation_agent()

    if agent is None:
        missing = ", ".join(state.missing_fields)
        recommended = ", ".join(getattr(state, "recommended_fields", []))

        if state.red_flags:
            safety = (
                "您描述的情况包含需要重视的风险信号。"
                "如果症状持续、加重，或伴随呼吸困难、出汗、恶心、晕厥，请尽快线下就医或急诊评估。"
            )
        else:
            safety = ""

        if missing:
            return f"{safety} 为了继续分诊，请补充：{missing}。"

        if recommended:
            return f"{safety} 主要信息已记录。为了让分诊更准确，请再补充：{recommended}。"

        return f"{safety} 主要信息已基本收集完成，接下来可以进入分诊总结。"

    prompt = f"""
case_id: {case_id}

当前 PatientCaseState:
{state.model_dump_json(indent=2)}

本轮患者输入:
{user_text}

请按顺序执行：
1. 判断患者本轮输入中是否包含可结构化更新的信息。
2. 如果包含，请调用 apply_patient_patch(case_id, patch) 更新状态。
3. 更新后查看 missing_fields、recommended_fields、red_flags 和 status。
4. 如果 red_flags 非空，回复中必须包含线下就医或急诊评估提醒。
5. 如果 missing_fields 非空，优先追问 missing_fields 中最重要的 1 到 2 个。
6. 如果 missing_fields 为空但 recommended_fields 非空，可以选择最重要的 1 到 2 个补充追问。
7. 如果信息已经足够，不要继续机械追问，可以提示将进入分诊总结。
8. 不要做最终诊断，不要直接给最终科室建议。
"""

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        return extract_last_text(result)

    except RateLimitError:
        logger.warning(f"Rate limit hit for case_id={case_id}")
        return "服务暂时繁忙，请稍后重试。"
    
    except APIError as e:
        logger.error(f"LLM API error for case_id={case_id}: {e}")
        return "AI 服务暂时不可用，请稍后重试。"
    
    except Exception as e:
        logger.exception(f"Unexpected error for case_id={case_id}: {e}")
        # 降级到基于规则的回复
        return generate_rule_based_response(state)


def generate_rule_based_response(state: PatientCaseState) -> str:
    """当 Agent 执行失败时，基于规则生成降级响应。
    
    这样即使 LLM 服务不可用，系统仍然可以继续收集信息。
    """
    missing = ", ".join(state.missing_fields) or "必要信息"
    
    if state.red_flags:
        return (
            "您描述的情况包含需要重视的风险信号。"
            "如果症状明显、持续加重或伴随呼吸困难、出汗、意识异常，请尽快线下就医或急诊评估。"
            f"为了继续了解情况，请补充：{missing}。"
        )
    
    if missing:
        return f"我已记录您的描述。为了继续分诊，请补充：{missing}。"
    
    return "主要信息已基本收集完成，接下来可以进入分诊总结。"