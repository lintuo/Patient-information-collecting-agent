import json
import os
from pathlib import Path

from patient_agent.domain.state import PatientCaseState


class CaseRepository:
    """病例数据持久化管理器。

    P1-1 优化：
    - 支持环境变量配置路径
    - 路径自动转为绝对路径
    - 独立的存储目录配置
    """

    def __init__(self, root: str = None):
        """初始化 Repository。

        Args:
            root: 存储目录路径。如果为 None，则从环境变量获取。
                  环境变量优先级：PATIENT_AGENT_DATA_ROOT > "data/cases"
        """
        if root is None:
            # 优先级：环境变量 > 默认值
            root = os.getenv("PATIENT_AGENT_DATA_ROOT", "data/cases")

        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str) -> Path:
        return self.root / f"{case_id}.json"

    def load(self, case_id: str) -> PatientCaseState:
        path = self._path(case_id)
        if not path.exists():
            return PatientCaseState(case_id=case_id)

        data = json.loads(path.read_text(encoding="utf-8"))
        return PatientCaseState.model_validate(data)

    def save(self, state: PatientCaseState) -> None:
        path = self._path(state.case_id)
        path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def delete(self, case_id: str) -> None:
        path = self._path(case_id)
        if path.exists():
            path.unlink()

    def list_case_ids(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.json"))


repo = CaseRepository()


def get_repository() -> CaseRepository:
    return repo