# 初始化日志配置（在其他模块导入之前）
from patient_agent.logging_config import setup_logging

setup_logging()

from fastapi import FastAPI

from patient_agent.api.routes import router


app = FastAPI(title="Patient Agent System")

app.include_router(router)