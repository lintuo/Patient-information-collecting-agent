# 初始化日志配置
from patient_agent.logging_config import setup_logging

setup_logging()

import os
import uuid

import gradio as gr
import httpx


API_BASE = os.getenv("PATIENT_AGENT_API_BASE", "http://127.0.0.1:8000")


def _request(method: str, path: str, **kwargs):
    url = f"{API_BASE}{path}"

    with httpx.Client(timeout=120) as client:
        response = client.request(method, url, **kwargs)

    response.raise_for_status()
    return response.json()


def new_case_id():
    return f"case-{uuid.uuid4().hex[:8]}"


def reset_case(case_id: str):
    if not case_id:
        case_id = new_case_id()

    try:
        _request("DELETE", f"/cases/{case_id}")
    except Exception:
        pass

    return case_id, [], {}, None, None


def load_case(case_id: str):
    if not case_id:
        return {}

    try:
        return _request("GET", f"/cases/{case_id}")
    except Exception as exc:
        return {
            "error": type(exc).__name__,
            "message": str(exc),
        }


def send_message(message: str, history: list, case_id: str):
    if not case_id:
        case_id = new_case_id()

    history = history or []

    if not message or not message.strip():
        return "", history, load_case(case_id), gr.update(), None

    history = [
        *history,
        {
            "role": "user",
            "content": message,
        },
        {
            "role": "assistant",
            "content": "正在生成回复，请稍候...",
        },
    ]

    try:
        result = _request(
            "POST",
            f"/cases/{case_id}/turn",
            json={"text": message},
        )

        assistant_message = result.get("assistant_message") or "我已记录。"

        history[-1] = {
            "role": "assistant",
            "content": assistant_message,
        }

        case_state = load_case(case_id)

        return "", history, case_state, gr.update(), None

    except Exception as exc:
        history[-1] = {
            "role": "assistant",
            "content": f"请求失败：{type(exc).__name__}: {exc}",
        }

        return "", history, load_case(case_id), gr.update(), None


def upload_and_analyze_image(image_file, case_id: str, history: list):
    """上传图片并触发异步分析"""
    if not case_id:
        case_id = new_case_id()

    history = history or []

    if image_file is None:
        return case_id, history, gr.update(), gr.update(), None

    try:
        with open(image_file, "rb") as f:
            files = {"file": (os.path.basename(image_file), f, "image/jpeg")}
            data = {"auto_analyze": "true"}
            result = _request(
                "POST",
                f"/cases/{case_id}/images",
                files=files,
                data=data,
            )

        msg = result.get("message", "图片上传成功。")
        history = [
            *history,
            {"role": "user", "content": f"[上传图片: {os.path.basename(image_file)}]"},
            {"role": "assistant", "content": msg},
        ]

        case_state = load_case(case_id)

        return case_id, history, case_state, gr.update(), image_file

    except Exception as exc:
        history = [
            *history,
            {"role": "user", "content": f"[上传图片]"},
            {"role": "assistant", "content": f"图片上传失败：{type(exc).__name__}: {exc}"},
        ]
        return case_id, history, load_case(case_id), gr.update(), None


def refresh_image_status(case_id: str):
    """轮询图片分析结果"""
    if not case_id:
        return None, None, None

    try:
        result = _request("GET", f"/cases/{case_id}/images")
        jobs = result.get("image_jobs", [])
        findings = result.get("image_findings", [])

        if not jobs:
            return None, None, None

        pending = [j for j in jobs if j["status"] in ("pending", "running")]
        done = [j for j in jobs if j["status"] == "done"]
        failed = [j for j in jobs if j["status"] == "failed"]
        skipped = [j for j in jobs if j["status"] == "done" and not j.get("finding")]

        summary_parts = []
        if pending:
            summary_parts.append(f"⏳ {len(pending)} 张图片分析中...")
        if done:
            summary_parts.append(f"✅ {len(done)} 张已完成")
        if failed:
            summary_parts.append(f"❌ {len(failed)} 张失败")
        if skipped:
            summary_parts.append(f"⚠️ {len(skipped)} 张被判定为非医疗图片，已跳过")

        summary = "\n".join(summary_parts) if summary_parts else "暂无图片"

        # image_findings: 只放真正有结果的 findings（给左上分析结果区用）
        findings_text = "\n\n".join(f"• {f}" for f in findings) if findings else ""

        # image_status: 放每张图片的详细状态
        job_details = []
        for j in jobs:
            if j["status"] == "failed":
                job_details.append(f"[❌] {j['file_id']}: {j.get('error', '未知错误')}")
            elif j["status"] == "done" and j.get("finding"):
                job_details.append(f"[✅] {j['file_id']}: {j['finding'][:100]}")
            elif j["status"] == "done":
                job_details.append(f"[⚠️] {j['file_id']}: 非医疗图片，已跳过（请上传医疗相关图片）")
            elif j["status"] in ("pending", "running"):
                job_details.append(f"[⏳] {j['file_id']}: 分析中...")

        details_text = "\n".join(job_details) if job_details else "暂无图片"
        return details_text, summary, findings_text

    except Exception:
        return None, None, None


def upload_audio_and_transcribe(audio_file, case_id: str, history: list):
    """上传音频文件，触发 ASR 转写，并将结果接入对话流程。"""
    if not case_id:
        case_id = new_case_id()

    history = history or []

    if audio_file is None:
        return case_id, history, gr.update(), None, None

    filename = os.path.basename(audio_file)
    try:
        with open(audio_file, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            upload_result = _request(
                "POST",
                f"/cases/{case_id}/audio",
                files=files,
            )

        audio_id = upload_result.get("audio_id", "?")
        msg = f"音频已上传（audio_id={audio_id}），正在转写中..."

        history = [
            *history,
            {"role": "user", "content": f"[上传音频: {filename}]"},
            {"role": "assistant", "content": msg},
        ]

        # 触发对话（LangGraph 会自动路由到 audio_transcription_node）
        turn_result = _request(
            "POST",
            f"/cases/{case_id}/turn",
            json={"text": ""},
        )

        # 加载转写结果
        audio_result = _request("GET", f"/cases/{case_id}/audio")
        transcripts = audio_result.get("audio_transcripts", [])

        if transcripts:
            latest = transcripts[-1]
            transcript_text = latest.get("transcript", "")
            if transcript_text:
                history = [
                    *history,
                    {"role": "user", "content": f"[语音转写] {transcript_text}"},
                ]

        assistant_message = turn_result.get("assistant_message") or ""
        if assistant_message:
            history = [
                *history,
                {"role": "assistant", "content": assistant_message},
            ]

        return case_id, history, gr.update(), None, load_case(case_id)

    except Exception as exc:
        history = [
            *history,
            {"role": "user", "content": f"[上传音频: {filename}]"},
            {"role": "assistant", "content": f"音频处理失败：{type(exc).__name__}: {exc}"},
        ]
        return case_id, history, gr.update(), None, load_case(case_id)


def refresh_audio_status(case_id: str):
    """轮询音频转写结果"""
    if not case_id:
        return None, None

    try:
        result = _request("GET", f"/cases/{case_id}/audio")
        attachments = result.get("audio_attachments", [])
        transcripts = result.get("audio_transcripts", [])

        if not attachments:
            return None, None

        pending = [a for a in attachments if a["transcription_status"] == "pending"]
        done = [a for a in attachments if a["transcription_status"] == "done"]
        failed = [a for a in attachments if a["transcription_status"] == "failed"]

        summary_parts = []
        if pending:
            summary_parts.append(f"⏳ {len(pending)} 个音频转写中...")
        if done:
            summary_parts.append(f"✅ {len(done)} 个已完成")
        if failed:
            summary_parts.append(f"❌ {len(failed)} 个失败")

        summary = "\n".join(summary_parts) if summary_parts else "暂无音频"

        transcript_texts = []
        for t in transcripts:
            confidence_str = f"{t['confidence']*100:.0f}%" if t.get("confidence") else "?"
            transcript_texts.append(
                f"• [{t['language'] or '?'}] ({confidence_str}): {t['transcript']}"
            )
        transcripts_display = "\n".join(transcript_texts) if transcript_texts else "暂无转写结果"

        return summary, transcripts_display

    except Exception:
        return None, None


def run_triage(case_id: str, history: list):
    if not case_id:
        return history, {"error": "case_id is required"}

    history = history or []

    try:
        state = _request("POST", f"/cases/{case_id}/triage")
        triage = state.get("triage_result")

        if triage:
            message = (
                f"分诊总结：{triage.get('summary', '')}\n\n"
                f"风险等级：{triage.get('risk_level', '')}\n\n"
                f"建议科室：{', '.join(triage.get('recommended_departments', []))}"
            )
        else:
            message = "已尝试生成分诊结果，但响应中没有 triage_result。"

        history = [
            *history,
            {"role": "user", "content": "[生成分诊]"},
            {"role": "assistant", "content": message},
        ]

        return history, state

    except Exception as exc:
        history = [
            *history,
            {"role": "user", "content": "[生成分诊]"},
            {"role": "assistant", "content": f"分诊失败：{type(exc).__name__}: {exc}"},
        ]

        return history, {"error": str(exc)}


def generate_report(case_id: str, history: list):
    if not case_id:
        return history, {"error": "case_id is required"}

    history = history or []

    try:
        result = _request("POST", f"/cases/{case_id}/report")
        report_path = result.get("report_path")

        if report_path:
            message = f"报告已生成：{report_path}"
        else:
            message = f"报告生成结果：{result}"

        history = [
            *history,
            {"role": "user", "content": "[生成报告]"},
            {"role": "assistant", "content": message},
        ]

        return history, load_case(case_id)

    except Exception as exc:
        history = [
            *history,
            {"role": "user", "content": "[生成报告]"},
            {"role": "assistant", "content": f"报告生成失败：{type(exc).__name__}: {exc}"},
        ]

        return history, {"error": str(exc)}


# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="病患信息 Agent 调试台") as demo:
    gr.Markdown("# 病患信息 Agent 调试台")

    # ── 顶部：病例 ID 操作栏 ──────────────────────────────────
    with gr.Row():
        case_id_input = gr.Textbox(
            label="Case ID",
            value=new_case_id(),
            scale=3,
        )
        reset_btn = gr.Button("新建/重置病例", variant="secondary", scale=1)
        load_btn = gr.Button("刷新状态", variant="secondary", scale=1)

    # ── 左栏：对话 ──────────────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="问诊对话",
                height=480,
            )

            msg = gr.Textbox(
                label="患者输入",
                placeholder="例如：我胸口疼，持续两个小时了，疼痛7分",
                lines=3,
            )

            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                triage_btn = gr.Button("生成分诊")
                report_btn = gr.Button("生成报告")

            case_state = gr.JSON(label="PatientCaseState")

        # ── 右栏：图片上传与分析 ─────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📷 图片上传与分析")

            image_upload = gr.Image(
                label="上传医疗图片",
                type="filepath",
                height=180,
            )

            upload_btn = gr.Button("上传并分析", variant="primary")
            refresh_image_btn = gr.Button("刷新分析结果", variant="secondary")

            image_preview = gr.Image(
                label="已上传图片预览",
                type="filepath",
                height=140,
                interactive=False,
            )

            image_status = gr.Textbox(
                label="图片分析状态",
                lines=3,
                interactive=False,
            )

            image_findings = gr.Textbox(
                label="图片分析结果",
                lines=5,
                interactive=False,
            )

    # ── 音频上传与分析 ────────────────────────────────────────
    gr.Markdown("### 🎤 语音输入（音频上传）")

    with gr.Row():
        audio_upload = gr.Audio(
            label="上传语音（wav/mp3/ogg）",
            type="filepath",
        )
        audio_upload_btn = gr.Button("上传并转写", variant="primary")

    audio_status = gr.Textbox(
        label="音频状态",
        lines=2,
        interactive=False,
    )

    audio_transcripts = gr.Textbox(
        label="语音转写结果",
        lines=4,
        interactive=False,
    )

    refresh_audio_btn = gr.Button("刷新音频状态", variant="secondary")

    # ── 事件绑定 ────────────────────────────────────────────
    send_btn.click(
        fn=send_message,
        inputs=[msg, chatbot, case_id_input],
        outputs=[msg, chatbot, case_state, image_upload, image_preview],
    )

    msg.submit(
        fn=send_message,
        inputs=[msg, chatbot, case_id_input],
        outputs=[msg, chatbot, case_state, image_upload, image_preview],
    )

    reset_btn.click(
        fn=reset_case,
        inputs=[case_id_input],
        outputs=[case_id_input, chatbot, case_state, image_upload, image_preview],
    )

    load_btn.click(
        fn=load_case,
        inputs=[case_id_input],
        outputs=[case_state],
    )

    upload_btn.click(
        fn=upload_and_analyze_image,
        inputs=[image_upload, case_id_input, chatbot],
        outputs=[case_id_input, chatbot, case_state, image_upload, image_preview],
    ).then(
        fn=refresh_image_status,
        inputs=[case_id_input],
        outputs=[image_status, image_findings, image_findings],
    )

    refresh_image_btn.click(
        fn=refresh_image_status,
        inputs=[case_id_input],
        outputs=[image_status, image_findings, image_findings],
    )

    audio_upload_btn.click(
        fn=upload_audio_and_transcribe,
        inputs=[audio_upload, case_id_input, chatbot],
        outputs=[case_id_input, chatbot, audio_upload, audio_status, case_state],
    ).then(
        fn=refresh_audio_status,
        inputs=[case_id_input],
        outputs=[audio_status, audio_transcripts],
    )

    refresh_audio_btn.click(
        fn=refresh_audio_status,
        inputs=[case_id_input],
        outputs=[audio_status, audio_transcripts],
    )

    triage_btn.click(
        fn=run_triage,
        inputs=[case_id_input, chatbot],
        outputs=[chatbot, case_state],
    )

    report_btn.click(
        fn=generate_report,
        inputs=[case_id_input, chatbot],
        outputs=[chatbot, case_state],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
