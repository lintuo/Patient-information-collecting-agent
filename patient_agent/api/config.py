"""API 配置模块"""
import os


def is_debug_enabled() -> bool:
    """检查是否启用调试端点。
    
    通过环境变量 DEBUG_ENDPOINTS 控制：
    - "true"/"1"/"yes" (不区分大小写): 启用调试端点
    - 其他值或未设置: 禁用调试端点
    
    生产环境务必设置 DEBUG_ENDPOINTS=false 或不设置。
    """
    value = os.getenv("DEBUG_ENDPOINTS", "false").lower()
    return value in ("true", "1", "yes")


def is_production() -> bool:
    """检查是否为生产环境"""
    return os.getenv("ENV", "development").lower() == "production"


def get_env() -> str:
    """获取当前环境"""
    return os.getenv("ENV", "development")
