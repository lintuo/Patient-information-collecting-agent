"""Graph 节点统一错误处理模块

P1-2: 为所有 Graph 节点提供统一的错误处理机制。

优势：
1. 统一的错误日志格式
2. 一致的错误状态返回
3. 便于问题追踪和排查
"""
import logging
from functools import wraps
from typing import Callable, TypeVar

from patient_agent.graph.state import GraphState

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=GraphState)


class NodeExecutionError(Exception):
    """节点执行时发生的错误"""

    def __init__(self, node_name: str, original_error: Exception):
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}' failed: {original_error}")


def with_node_error_handling(node_name: str):
    """节点统一错误处理装饰器。

    使用方式：
        @with_node_error_handling("conversation")
        def conversation_node(state: GraphState) -> GraphState:
            # 节点逻辑
            pass

    Args:
        node_name: 节点名称，用于日志标识

    Returns:
        装饰后的节点函数
    """

    def decorator(func: Callable[[GraphState], GraphState]) -> Callable[[GraphState], GraphState]:
        @wraps(func)
        def wrapper(state: GraphState) -> GraphState:
            try:
                return func(state)
            except Exception as e:
                logger.exception(
                    f"Node '{node_name}' execution failed",
                    extra={
                        "node": node_name,
                        "case_id": state.get("case_id"),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )

                # 构造错误状态
                errors = list(state.get("errors") or [])
                errors.append(f"{node_name}:{type(e).__name__}:{str(e)}")

                node_errors = dict(state.get("node_errors") or {})
                node_errors[node_name] = {
                    "error": str(e),
                    "type": type(e).__name__,
                }

                return {
                    **state,
                    "errors": errors,
                    "node_errors": node_errors,
                }

        return wrapper

    return decorator
