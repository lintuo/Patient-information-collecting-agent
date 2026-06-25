"""领域规则模块

P2-3: 统一管理 RED_FLAG 配置，避免重复定义。
"""
import re

from patient_agent.domain.patch import PatientStatePatch
from patient_agent.domain.state import PatientCaseState


# =============================================================================
# 字段定义
# =============================================================================
REQUIRED_FIELDS = [
    "chief_complaint",
    "duration",
    "severity",
    "age",
    "sex",
]

RECOMMENDED_FIELDS = [
    "name"
    "medical_history",
    "medications",
    "allergies",
]


# =============================================================================
# P2-3: 统一的红旗规则配置
# =============================================================================
RED_FLAG_CONFIG = {
    "chest_pain": {
        "keywords": ["胸痛", "胸口疼", "胸闷", "压迫感", "心前区疼"],
        "safety_message": "胸痛或胸部压迫感需要重视，若持续、加重或伴随出汗、呼吸困难、恶心、晕厥，请尽快急诊评估。",
        "departments": ["急诊科", "心内科"],
    },
    "breathing_difficulty": {
        "keywords": ["呼吸困难", "喘不上气", "气短", "憋气", "说话费劲"],
        "safety_message": "呼吸困难需要尽快线下评估，若症状明显请及时就医或急诊。",
        "departments": ["急诊科", "呼吸内科"],
    },
    "allergic_reaction": {
        "keywords": ["喉咙肿", "嘴唇肿", "呼吸受限", "全身过敏", "严重过敏"],
        "safety_message": "过敏伴喉咙或嘴唇肿胀、呼吸受限时可能进展较快，请及时急诊评估。",
        "departments": ["急诊科", "皮肤科"],
    },
    "severe_headache": {
        "keywords": ["剧烈头痛", "爆炸样头痛", "突然头痛"],
        "safety_message": "突发剧烈头痛需要重视，建议尽快线下评估。",
        "departments": ["急诊科", "神经内科"],
    },
    "consciousness": {
        "keywords": ["意识不清", "昏迷", "晕厥", "突然晕倒"],
        "safety_message": "意识异常需要立即就医，请尽快急诊评估。",
        "departments": ["急诊科", "神经内科"],
    },
    "bleeding": {
        "keywords": ["大出血", "止不住血", "大量出血"],
        "safety_message": "大量出血需要立即止血处理，请尽快急诊或拨打急救电话。",
        "departments": ["急诊科", "外科"],
    },
}

# 向后兼容：保持 RED_FLAG_RULES 不变
RED_FLAG_RULES = RED_FLAG_CONFIG

# 导出统一的关键词映射（供 detect_red_flags 使用）
RED_FLAG_KEYWORDS = {k: v["keywords"] for k, v in RED_FLAG_CONFIG.items()}


# =============================================================================
# 字段计算函数
# =============================================================================
def compute_missing_fields(state: PatientCaseState) -> list[str]:
    """Return fields that must be collected before triage."""
    facts = state.facts
    missing: list[str] = []

    if not facts.chief_complaint:
        missing.append("chief_complaint")

    if not facts.duration:
        missing.append("duration")

    if not facts.severity:
        missing.append("severity")

    if facts.age is None:
        missing.append("age")

    if not facts.sex:
        missing.append("sex")

    return missing


def compute_recommended_fields(state: PatientCaseState) -> list[str]:
    """Return useful-but-not-blocking fields for better triage and reporting."""
    facts = state.facts
    recommended: list[str] = []

    if not facts.name:
        recommended.append("name")

    if not facts.medical_history:
        recommended.append("medical_history")

    if not facts.medications:
        recommended.append("medications")

    if not facts.allergies:
        recommended.append("allergies")

    return recommended


def merge_unique(existing: list[str], new_values: list[str]) -> list[str]:
    """合并列表并去重"""
    merged = list(existing)

    for value in new_values:
        if value and value not in merged:
            merged.append(value)

    return merged


# =============================================================================
# 红旗检测
# =============================================================================
def detect_red_flags(state: PatientCaseState) -> list[str]:
    """P2-3: 使用统一的 RED_FLAG_KEYWORDS 检测红旗症状"""
    facts = state.facts

    text = " ".join(
        [
            facts.chief_complaint or "",
            facts.onset or "",
            facts.duration or "",
            facts.severity or "",
            *facts.symptoms,
            *facts.medical_history,
        ]
    )

    red_flags: list[str] = []

    # 使用统一的关键词映射
    for flag, keywords in RED_FLAG_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            red_flags.append(flag)

    # Also check image_findings for red_flag keywords
    image_text = " ".join(getattr(state, "image_findings", []))
    for flag, keywords in RED_FLAG_KEYWORDS.items():
        if flag not in red_flags and any(keyword in image_text for keyword in keywords):
            red_flags.append(flag)

    return red_flags


def has_pending_images(state: PatientCaseState) -> bool:
    return any(job.status in {"pending", "running"} for job in state.image_jobs)


def refresh_case_status(state: PatientCaseState) -> PatientCaseState:
    state.missing_fields = compute_missing_fields(state)

    if hasattr(state, "recommended_fields"):
        state.recommended_fields = compute_recommended_fields(state)

    rule_flags = detect_red_flags(state)
    state.red_flags = merge_unique(state.red_flags, rule_flags)

    if state.report_path:
        state.status = "reported"
    elif state.triage_result:
        state.status = "triaged"
    elif not state.missing_fields and has_pending_images(state):
        state.status = "waiting_image"
    elif not state.missing_fields:
        state.status = "ready_for_triage"
    else:
        state.status = "collecting"

    return state


# =============================================================================
# 规则提取（降级用）
# =============================================================================
def rough_extract_patch_from_text(text: str) -> PatientStatePatch:
    """Rule-based fallback extraction.

    This is not meant to replace the LLM extraction.
    It only keeps the system usable if tool calling fails or the model API is unavailable.
    """
    patch = PatientStatePatch()

    clean = text.strip()

    complaint_keywords = [
        "疼",
        "痛",
        "咳嗽",
        "发烧",
        "发热",
        "头晕",
        "恶心",
        "呕吐",
        "腹泻",
        "胸闷",
        "呼吸困难",
        "皮疹",
        "红疹",
        "伤口",
    ]

    if clean and any(keyword in clean for keyword in complaint_keywords):
        patch.chief_complaint = clean

    duration_match = re.search(
        r"(\d+\s*(分钟|小时|天|周|月)|半小时|一小时|两小时|两个小时|三小时|一天|两天|三天)",
        text,
    )
    if duration_match:
        patch.duration = duration_match.group(0)

    severity_match = re.search(
        r"(\d+\s*分|轻微|很痛|非常痛|严重|剧烈)",
        text,
    )
    if severity_match:
        patch.severity = severity_match.group(0)

    history = []
    for word in ["高血压", "糖尿病", "冠心病", "哮喘", "慢阻肺"]:
        if word in text:
            history.append(word)
    if history:
        patch.medical_history = history

    medications = []
    for word in ["降压药", "降糖药", "阿司匹林", "硝酸甘油", "布洛芬"]:
        if word in text:
            medications.append(word)
    if medications:
        patch.medications = medications

    allergies = []
    for word in ["青霉素过敏", "头孢过敏", "药物过敏", "食物过敏"]:
        if word in text:
            allergies.append(word)
    if allergies:
        patch.allergies = allergies

    age_match = re.search(r"(\d{1,3})\s*岁", text)
    if age_match:
        patch.age = int(age_match.group(1))

    if "男" in text and not any(word in text for word in ["男女", "男朋友"]):
        patch.sex = "男"
    elif "女" in text:
        patch.sex = "女"

    return patch


# =============================================================================
# 分诊阻断检查
# =============================================================================
def get_triage_blockers(
    state: PatientCaseState,
    require_images_done: bool = True,
) -> list[str]:
    blockers: list[str] = []

    if state.missing_fields:
        blockers.extend([f"missing:{field}" for field in state.missing_fields])

    if require_images_done and has_pending_images(state):
        blockers.append("pending_images")

    return blockers


def can_run_triage(
    state: PatientCaseState,
    require_images_done: bool = True,
) -> bool:
    return not get_triage_blockers(
        state,
        require_images_done=require_images_done,
    )


# =============================================================================
# 辅助函数：获取红旗安全提示
# =============================================================================
def get_red_flag_safety_message(flag: str) -> str:
    """根据红旗类型获取安全提示"""
    if flag in RED_FLAG_CONFIG:
        return RED_FLAG_CONFIG[flag]["safety_message"]
    return "请及时就医评估。"


def get_red_flag_departments(flag: str) -> list[str]:
    """根据红旗类型获取建议科室"""
    if flag in RED_FLAG_CONFIG:
        return RED_FLAG_CONFIG[flag]["departments"]
    return ["急诊科"]