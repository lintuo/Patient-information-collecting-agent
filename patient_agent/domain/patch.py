from pydantic import BaseModel, Field

from patient_agent.domain.state import PatientCaseState


class PatientStatePatch(BaseModel):
    age: int | None = Field(default=None, description="患者年龄")
    sex: str | None = Field(default=None, description="患者性别")

    chief_complaint: str | None = Field(default=None, description="患者主诉")
    symptoms: list[str] | None = Field(default=None, description="症状列表")
    associated_symptoms: list[str] | None = Field(default=None, description="伴随症状")

    onset: str | None = Field(default=None, description="起病时间")
    duration: str | None = Field(default=None, description="症状持续时间")
    severity: str | None = Field(default=None, description="症状严重程度，例如疼痛评分")

    medical_history: list[str] | None = Field(default=None, description="既往史")
    medications: list[str] | None = Field(default=None, description="当前用药")
    allergies: list[str] | None = Field(default=None, description="过敏史")

    red_flags: list[str] | None = None

    pregnancy_related: str | None = Field(default=None, description="妊娠相关信息，如不适用则为空")
    extra_notes: list[str] | None = Field(default=None, description="其他患者明确提供的重要信息")


def append_unique(target: list[str], values: list[str] | None) -> None:
    if not values:
        return

    for value in values:
        clean = value.strip()
        if clean and clean not in target:
            target.append(clean)


def apply_patient_patch(
    state: PatientCaseState,
    patch: PatientStatePatch,
) -> PatientCaseState:
    facts = state.facts

    if patch.age is not None:
        facts.age = patch.age

    if patch.sex:
        facts.sex = patch.sex

    if patch.chief_complaint:
        if not facts.chief_complaint:
            facts.chief_complaint = patch.chief_complaint
        elif patch.chief_complaint not in facts.chief_complaint:
            append_unique(facts.symptoms, [patch.chief_complaint])

    append_unique(facts.symptoms, patch.symptoms)

    if patch.onset:
        facts.onset = patch.onset

    if patch.duration:
        facts.duration = patch.duration

    if patch.severity:
        facts.severity = patch.severity

    append_unique(facts.medical_history, patch.medical_history)
    append_unique(facts.medications, patch.medications)
    append_unique(facts.allergies, patch.allergies)

    append_unique(state.red_flags, patch.red_flags)

    return state