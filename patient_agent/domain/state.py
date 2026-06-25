from typing import Literal

from pydantic import BaseModel, Field


CaseStatus = Literal[
    "collecting",
    "waiting_image",
    "ready_for_triage",
    "triaged",
    "reported",
]

RedFlag = Literal[
    "chest_pain",
    "breathing_difficulty",
    "consciousness",
    "bleeding",
    "severe_headache",
    "allergic_reaction",
    "high_fever_child",
    "severe_abdominal_pain",
]

RiskLevel = Literal["low", "medium", "high", "urgent"]


class UploadedFile(BaseModel):
    file_id: str
    path: str
    content_type: str
    original_name: str | None = None


class Transcript(BaseModel):
    audio_file_id: str
    text: str
    confidence: float | None = None


class AudioAttachment(BaseModel):
    """音频文件元数据，记录一次音频上传的信息。"""
    audio_id: str
    filename: str
    path: str
    content_type: str
    uploaded_at: str = ""  # ISO format datetime
    transcription_status: Literal["pending", "done", "failed"] = "pending"
    error_message: str | None = None


class AudioTranscript(BaseModel):
    """语音识别结果，对应一次音频转写。"""
    audio_id: str
    transcript: str
    language: str | None = None
    confidence: float | None = None
    model: str | None = None
    backend: str | None = None
    device: str | None = None
    latency_ms: int | None = None
    created_at: str = ""  # ISO format datetime
    error_message: str | None = None


class ImageJob(BaseModel):
    job_id: str
    file_id: str
    image_path: str
    status: Literal["pending", "running", "done", "failed"] = "pending"
    finding: str | None = None
    vision_finding: str | None = None  # 多模态大模型图像理解结果
    error: str | None = None


class PatientFacts(BaseModel):
    name: str | None = None
    age: int | None = None
    sex: str | None = None
    chief_complaint: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    onset: str | None = None
    duration: str | None = None
    severity: str | None = None
    medical_history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    summary: str
    risk_level: RiskLevel
    recommended_departments: list[str]
    reasons: list[str]
    safety_notice: str
    rag_used: bool = False
    rag_candidate_ids: list[str] = Field(default_factory=list)
    rag_notes: str = ""
    used_multimodal_evidence: bool = False
    multimodal_notes: str = ""


class PatientCaseState(BaseModel):
    case_id: str
    status: CaseStatus = "collecting"

    facts: PatientFacts = Field(default_factory=PatientFacts)

    conversation_turns: list[dict] = Field(default_factory=list)
    conversation_summary: str = ""

    uploaded_files: list[UploadedFile] = Field(default_factory=list)
    transcripts: list[Transcript] = Field(default_factory=list)

    image_jobs: list[ImageJob] = Field(default_factory=list)
    image_findings: list[str] = Field(default_factory=list)

    audio_attachments: list[AudioAttachment] = Field(default_factory=list)
    audio_transcripts: list[AudioTranscript] = Field(default_factory=list)

    missing_fields: list[str] = Field(default_factory=list)
    recommended_fields: list[str] = Field(default_factory=list)
    recommended_fields_asked: bool = False  # 是否已追问过 recommended_fields
    red_flags: list[RedFlag] = Field(default_factory=list)

    triage_result: TriageResult | None = None
    report_path: str | None = None