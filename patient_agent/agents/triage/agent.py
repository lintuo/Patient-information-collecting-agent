from typing import Any

from deepagents import create_deep_agent

from patient_agent.agents.triage.prompts import TRIAGE_SYSTEM_PROMPT
from patient_agent.agents.triage.tools import triage_tools
from patient_agent.domain.state import PatientCaseState, TriageResult
from patient_agent.services.llm.factory import get_chat_model


URGENT_FLAGS = {
    "chest_pain",
    "breathing_difficulty",
    "consciousness",
    "bleeding",
    "severe_headache",
    "allergic_reaction",
    "high_fever_child",
    "severe_abdominal_pain",
}


def get_case_value(state: PatientCaseState, field: str, default: Any = None) -> Any:
    facts = getattr(state, "facts", None)

    if facts is not None and hasattr(facts, field):
        value = getattr(facts, field)
        if value not in (None, "", []):
            return value

    value = getattr(state, field, default)
    if value in (None, "", []):
        return default

    return value


def extract_last_text(result: Any) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []

    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content:
            return str(content)

        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])

    return "根据当前信息，建议进一步线下评估。"


def build_triage_agent():
    model = get_chat_model()

    if model is None:
        return None

    return create_deep_agent(
        model=model,
        tools=triage_tools,
        system_prompt=TRIAGE_SYSTEM_PROMPT,
    )


def departments_from_candidates(candidates: list[dict] | None) -> list[str]:
    departments: list[str] = []

    for item in candidates or []:
        dept = item.get("department")
        if dept and dept not in departments:
            departments.append(dept)

    return departments[:3]


def reasons_from_candidates(candidates: list[dict] | None) -> list[str]:
    reasons: list[str] = []

    for item in candidates or []:
        reason = (
            item.get("department_reason")
            or item.get("reason")
            or item.get("content")
            or item.get("suitable_for")
        )
        department = item.get("department")

        if department and reason:
            reasons.append(f"{department}：{reason}")
        elif department:
            reasons.append(f"{department}：由科室知识库检索命中。")

    return reasons[:3]


def normalize_risk_level(value: str | None, fallback: str = "medium") -> str:
    if not value:
        return fallback

    text = str(value).strip().lower()

    mapping = {
        "urgent": "urgent",
        "emergency": "urgent",
        "critical": "urgent",
        "high": "high",
        "medium": "medium",
        "moderate": "medium",
        "low": "low",
        "急诊": "urgent",
        "紧急": "urgent",
        "危急": "urgent",
        "高": "high",
        "高风险": "high",
        "中": "medium",
        "中风险": "medium",
        "低": "low",
        "低风险": "low",
    }

    if text in mapping:
        return mapping[text]

    if "急" in text or "危" in text:
        return "urgent"

    if "高" in text:
        return "high"

    if "中" in text:
        return "medium"

    if "低" in text:
        return "low"

    return fallback


def infer_risk_level(state: PatientCaseState) -> str:
    red_flags = set(get_case_value(state, "red_flags", []) or [])
    severity_text = str(get_case_value(state, "severity", "") or "")

    if red_flags & URGENT_FLAGS:
        return "urgent"

    if any(value in severity_text for value in ["9", "10", "严重", "剧烈"]):
        return "urgent"

    if any(value in severity_text for value in ["7", "8"]):
        return "high"

    if any(value in severity_text for value in ["4", "5", "6"]):
        return "medium"

    return "low"


def classify_risk_and_departments(
    state: PatientCaseState,
) -> tuple[str, list[str], list[str]]:
    red_flags = set(get_case_value(state, "red_flags", []) or [])
    risk_level = infer_risk_level(state)

    if "chest_pain" in red_flags:
        return (
            "urgent",
            ["急诊科", "心内科"],
            ["患者存在胸痛或胸部压迫感，需要优先排除急性心血管风险。"],
        )

    if "breathing_difficulty" in red_flags:
        return (
            "urgent",
            ["急诊科", "呼吸内科"],
            ["患者存在呼吸困难，需要尽快线下评估。"],
        )

    if "severe_headache" in red_flags:
        return (
            "urgent",
            ["急诊科", "神经内科"],
            ["患者存在严重头痛相关红旗风险，需要排除神经系统急症。"],
        )

    if "severe_abdominal_pain" in red_flags:
        return (
            "urgent",
            ["急诊科", "普外科", "消化内科"],
            ["患者存在严重腹痛相关风险，需要排除急腹症。"],
        )

    if "allergic_reaction" in red_flags:
        return (
            "urgent",
            ["急诊科", "皮肤科"],
            ["患者存在严重过敏风险，需要优先评估呼吸道和循环风险。"],
        )

    if red_flags:
        return (
            "high",
            ["急诊科", "全科医学科"],
            ["存在红旗风险信号，需要优先线下评估。"],
        )

    return (
        risk_level,
        ["全科医学科"],
        ["当前信息不足以形成诊断，建议由线下医生进一步评估。"],
    )


def build_candidate_text(department_candidates: list[dict] | None) -> str:
    if not department_candidates:
        return "无。"

    blocks: list[str] = []

    for item in department_candidates:
        blocks.append(
            "\n".join(
                [
                    f"候选科室：{item.get('department', '未知')}",
                    f"优先级：{item.get('priority', '未标注')}",
                    f"检索依据：{item.get('content') or item.get('department_reason') or item.get('reason') or '未提供'}",
                ]
            )
        )

    return "\n\n".join(blocks)


def build_fallback_summary(state: PatientCaseState) -> str:
    chief_complaint = get_case_value(state, "chief_complaint", "未明确")
    duration = get_case_value(state, "duration", "未提供")
    severity = get_case_value(state, "severity", "未提供")
    symptoms = get_case_value(state, "symptoms", [])
    red_flags = get_case_value(state, "red_flags", [])

    symptoms_text = "、".join(symptoms) if symptoms else "未提供"
    red_flags_text = "、".join(red_flags) if red_flags else "无明确红旗风险"

    return (
        f"患者主诉：{chief_complaint}。"
        f"持续时间：{duration}。"
        f"严重程度：{severity}。"
        f"伴随症状：{symptoms_text}。"
        f"红旗风险：{red_flags_text}。"
        "建议结合线下检查进一步评估。"
    )


def build_multimodal_evidence_text(state: PatientCaseState) -> str:
    """Build text representation of image_findings and audio_transcripts for the prompt."""
    parts: list[str] = []

    findings = getattr(state, "image_findings", [])
    if findings:
        parts.append("【图像分析结果】\n" + "\n".join(f"- {f}" for f in findings))
    else:
        parts.append("【图像分析结果】无")

    transcripts = getattr(state, "audio_transcripts", [])
    if transcripts:
        for t in transcripts:
            conf = f"{t.confidence * 100:.0f}%" if t.confidence else "未知"
            parts.append(
                f"【语音转写】语言={t.language or '?'}，置信度={conf}，内容：{t.transcript}"
            )
    else:
        parts.append("【语音转写】无")

    return "\n\n".join(parts)


def run_triage_agent(
    state: PatientCaseState,
    department_candidates: list[dict] | None = None,
) -> TriageResult:
    department_candidates = department_candidates or []

    risk_level = infer_risk_level(state)
    departments = departments_from_candidates(department_candidates)
    reasons = reasons_from_candidates(department_candidates)

    if not departments:
        risk_level, departments, reasons = classify_risk_and_departments(state)

    if not reasons:
        reasons = ["根据患者结构化信息和科室知识库候选结果生成分诊建议。"]

    candidate_text = build_candidate_text(department_candidates)
    multimodal_text = build_multimodal_evidence_text(state)
    agent = build_triage_agent()

    rag_used = bool(department_candidates)
    rag_ids = [item.get("id", "") for item in department_candidates if item.get("id")]

    if agent is None:
        summary = build_fallback_summary(state)
    else:
        prompt = f"""当前 PatientCaseState:
{state.model_dump_json(indent=2)}

RAG 检索到的候选科室知识：
{candidate_text}

多模态辅助证据（图像分析 + 语音转写）：
{multimodal_text}

请基于 PatientCaseState、候选科室知识和多模态证据生成分诊辅助总结。

要求：
1. 不做最终诊断。
2. 推荐科室必须优先从候选科室中选择。
3. 如果 red_flags 非空，优先考虑急诊科或高优先级科室。
4. 如果 image_findings 包含红旗信号（如出血、骨折、肺部阴影），提升风险等级并推荐急诊。
5. 如果 audio_transcripts 存在且 confidence < 0.8，应在总结中注明"语音转写置信度偏低，仅供参考"。
6. 说明推荐原因，并注明使用了哪些多模态证据。
7. 输出中文自然语言总结即可。
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
            summary = extract_last_text(result)
        except Exception as exc:
            summary = build_fallback_summary(state)
            reasons.append(f"分诊模型调用失败，已使用规则和RAG候选兜底：{exc}")

    risk_level = normalize_risk_level(risk_level, fallback="medium")

    return TriageResult(
        summary=summary,
        risk_level=risk_level,
        recommended_departments=departments,
        reasons=reasons,
        safety_notice="本建议仅用于分诊辅助，不能替代医生诊断。如症状加重或出现危险信号，请及时就医。",
        rag_used=rag_used,
        rag_candidate_ids=rag_ids,
        rag_notes="参考了 RAG 检索结果" if rag_used else "未使用 RAG",
        used_multimodal_evidence=bool(
            getattr(state, "image_findings", []) or getattr(state, "audio_transcripts", [])
        ),
        multimodal_notes="使用了图像分析或语音转写作为辅助证据" if (
            getattr(state, "image_findings", []) or getattr(state, "audio_transcripts", [])
        ) else "",
    )