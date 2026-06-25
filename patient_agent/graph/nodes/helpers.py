"""Graph 节点辅助函数模块

P1-3: 提供统一的状态加载、保存、刷新函数。
减少各节点中的重复代码。
"""
from patient_agent.domain.rules import refresh_case_status
from patient_agent.graph.state import GraphState, PatientCaseState
from patient_agent.storage.repository import CaseRepository


def get_state_repository() -> CaseRepository:
    """获取状态仓库（使用全局单例）"""
    from patient_agent.storage.repository import get_repository

    return get_repository()


def load_and_refresh_state(case_id: str) -> PatientCaseState:
    """统一加载和刷新状态。

    包含以下步骤：
    1. 从 Repository 加载病例状态
    2. 刷新 missing_fields、recommended_fields、red_flags
    3. 保存回 Repository

    Args:
        case_id: 病例 ID

    Returns:
        刷新后的 PatientCaseState
    """
    repo = get_state_repository()
    state = repo.load(case_id)
    state = refresh_case_status(state)
    repo.save(state)
    return state


def save_and_refresh_state(state: PatientCaseState) -> PatientCaseState:
    """统一保存和刷新状态。

    包含以下步骤：
    1. 刷新状态（missing_fields 等）
    2. 保存到 Repository

    Args:
        state: PatientCaseState 实例

    Returns:
        刷新后的 PatientCaseState
    """
    repo = get_state_repository()
    state = refresh_case_status(state)
    repo.save(state)
    return state


def get_or_create_state(case_id: str, intent: str) -> PatientCaseState:
    """获取或创建病例状态。

    Args:
        case_id: 病例 ID
        intent: 当前意图（用于日志）

    Returns:
        PatientCaseState 实例
    """
    return load_and_refresh_state(case_id)
