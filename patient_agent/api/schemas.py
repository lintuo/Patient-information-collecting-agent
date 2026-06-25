from pydantic import BaseModel


class TurnRequest(BaseModel):
    text: str
    auto_triage: bool = True
    auto_report: bool = False


class TurnResponse(BaseModel):
    case_id: str
    status: str
    assistant_message: str | None = None
    missing_fields: list[str]
    red_flags: list[str]


class DebugChatRequest(BaseModel):
    prompt: str


class ImageUploadRequest(BaseModel):
    """图片上传请求体（用于非 multipart 的 JSON API）"""
    file_id: str
    filename: str
    content_type: str
    content_base64: str  # base64 编码的文件内容