"""文件存储模块 — 支持图片和音频上传。

- 图片：保存到 data/uploads/{case_id}/images/
- 音频：保存到 data/uploads/{case_id}/audio/
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from patient_agent.domain.state import AudioAttachment, UploadedFile


# Supported types
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/mp4",
    "audio/x-m4a",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".webm", ".mp4", ".m4a"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(filename: str) -> str:
    """去掉路径分隔符，防止路径穿越。"""
    return os.path.basename(filename)


def _ext_from_content_type(content_type: str) -> str:
    ct = content_type.lower().strip()
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/webm": ".webm",
        "audio/mp4": ".mp4",
        "audio/x-m4a": ".m4a",
    }
    return mapping.get(ct, ".bin")


# =============================================================================
# Image storage
# =============================================================================

def save_uploaded_file(
    case_id: str,
    filename: str,
    content_type: str,
    content: bytes,
    upload_root: str = "data/uploads",
) -> UploadedFile:
    """保存用户上传的文件（图片或音频），返回 UploadedFile 元数据。

    图片自动创建 ImageJob 条目（由调用方处理）。
    音频返回 AudioAttachment 元数据。
    """
    file_id = str(uuid4())
    safe_name = _safe_filename(filename or f"{file_id}.bin")

    if content_type in IMAGE_TYPES:
        subdir = "images"
    else:
        subdir = "audio"

    case_dir = Path(upload_root) / case_id / subdir
    case_dir.mkdir(parents=True, exist_ok=True)

    path = case_dir / f"{file_id}-{safe_name}"
    path.write_bytes(content)

    return UploadedFile(
        file_id=file_id,
        path=str(path),
        content_type=content_type,
        original_name=safe_name,
    )


# =============================================================================
# Audio storage
# =============================================================================

def save_audio_file(
    case_id: str,
    filename: str,
    content_type: str,
    content: bytes,
    upload_root: str = "data/uploads",
) -> AudioAttachment:
    """保存音频文件到 data/uploads/{case_id}/audio/，返回 AudioAttachment 元数据。

    transcription_status 初始为 "pending"，由 audio_transcription_node 后续更新。
    """
    audio_id = str(uuid4())
    safe_name = _safe_filename(filename or f"{audio_id}.bin")

    # 强制从 content_type 推导扩展名，避免文件名伪造
    ext = _ext_from_content_type(content_type)

    case_dir = Path(upload_root) / case_id / "audio"
    case_dir.mkdir(parents=True, exist_ok=True)

    # 文件名 = audio_id + 扩展名（不含用户输入的后缀，防止路径穿越）
    path = case_dir / f"{audio_id}{ext}"
    path.write_bytes(content)

    return AudioAttachment(
        audio_id=audio_id,
        filename=safe_name,
        path=str(path),
        content_type=content_type,
        uploaded_at=_now_iso(),
        transcription_status="pending",
    )


def get_audio_path(audio_id: str, case_id: str, upload_root: str = "data/uploads") -> Path:
    """根据 audio_id 查找对应的音频文件路径。"""
    audio_dir = Path(upload_root) / case_id / "audio"
    # audio_id 本身是文件名主体（不含扩展名），但我们只存一个文件
    for ext in AUDIO_EXTENSIONS:
        p = audio_dir / f"{audio_id}{ext}"
        if p.exists():
            return p
    # Fallback: 尝试在 audio_dir 下找包含 audio_id 前缀的文件
    for p in audio_dir.iterdir():
        if p.name.startswith(audio_id):
            return p
    raise FileNotFoundError(f"Audio file not found for audio_id={audio_id}")


def attach_audio_to_state(state, attachment: AudioAttachment):
    """将 AudioAttachment 追加到 PatientCaseState.audio_attachments。"""
    state.audio_attachments.append(attachment)
    return state
