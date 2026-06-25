"""日志配置模块

P2-1: 为整个应用提供统一的日志配置。
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def get_log_level() -> int:
    """从环境变量获取日志级别"""
    level_str = os.getenv("PATIENT_AGENT_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def get_log_dir() -> Path:
    """获取日志目录，默认使用项目根目录下的 logs 文件夹"""
    log_dir = os.getenv("PATIENT_AGENT_LOG_DIR", "logs")
    path = Path(log_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(
    log_level: Optional[int] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> None:
    """配置应用日志。

    Args:
        log_level: 日志级别，默认从环境变量获取
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
    """
    if log_level is None:
        log_level = get_log_level()

    # 日志格式
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 清除现有处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handlers = []

    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(console_handler)

    # 文件处理器
    if log_to_file:
        log_dir = get_log_dir()
        log_file = log_dir / "patient-agent.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)

    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

    # 降低第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # 应用日志器
    logger = logging.getLogger("patient_agent")
    logger.info(f"日志系统初始化完成，日志级别: {logging.getLevelName(log_level)}")
    if log_to_file:
        logger.info(f"日志文件: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器。

    Args:
        name: 日志器名称，通常使用 __name__

    Returns:
        Logger 实例
    """
    return logging.getLogger(name)


# 默认初始化
setup_logging()
