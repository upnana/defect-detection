import base64
import json
import mimetypes
from pathlib import Path

import httpx

from app.config import settings
from app.defect_types import DEFECT_CLASSES, normalize_defect_ids

_CLASS_LINES = "\n".join(
    f'- "{c["id"]}": {c["label_zh"]} — {c["description"]}' for c in DEFECT_CLASSES
)

SCREW_INSPECT_PROMPT = f"""你是工业螺丝外观缺陷质检专家。请分析螺丝图片，判断质量类别。

允许的缺陷类别 id（violation_types 只能使用这些英文 id，可多选；良品时仅含 good）：
{_CLASS_LINES}

请严格按以下 JSON 返回（只返回 JSON，不要 markdown 或其它文字）：
{{
    "violation_detected": true/false,
    "violation_types": ["good"] 或缺陷 id 列表,
    "confidence": 0.0-1.0,
    "risk_description": "中文简要说明观察到的区域与依据"
}}

规则：
- 无明显缺陷时 violation_detected=false，violation_types=["good"]，confidence 表示对良品的置信度（越高越确定良品）。
- 存在任一缺陷时 violation_detected=true，violation_types 列出所有确认的缺陷 id（不要包含 good）。
- confidence 表示你对当前判断的整体置信度。"""


def _to_model_media_ref(media_url: str, default_mime: str) -> str:
    if not media_url:
        return media_url

    if media_url.startswith(("http://", "https://", "data:")):
        return media_url

    local_path = None
    if media_url.startswith("/uploads/"):
        local_path = Path(settings.upload_dir) / media_url.removeprefix("/uploads/")
    else:
        candidate = Path(media_url)
        if candidate.exists():
            local_path = candidate

    if local_path and local_path.exists():
        mime = mimetypes.guess_type(local_path.name)[0] or default_mime
        encoded = base64.b64encode(local_path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    return media_url


def _build_vlm_prompt(coarse_screen: dict | None) -> str:
    if not coarse_screen or not coarse_screen.get("enabled"):
        return SCREW_INSPECT_PROMPT

    detections = coarse_screen.get("detections") or []
    if not detections:
        return SCREW_INSPECT_PROMPT + "\n\nYOLO 粗筛未给出高置信候选框，请独立完整判断。"

    lines = []
    for d in detections[:8]:
        lines.append(
            f"- 类别 {d.get('class_id')} 置信度 {d.get('confidence')} 框 {d.get('bbox')}"
        )
    extra = "\n".join(lines)
    return (
        SCREW_INSPECT_PROMPT
        + "\n\nYOLO 粗筛候选（请结合整图验证，可修正误检）：\n"
        + extra
    )


async def analyze_screw_image(image_url: str, coarse_screen: dict | None = None) -> dict:
    image_ref = _to_model_media_ref(image_url, "image/jpeg")
    prompt = _build_vlm_prompt(coarse_screen)
    content = [
        {"type": "input_image", "image_url": image_ref},
        {"type": "input_text", "text": prompt},
    ]
    timeout = int(max(5, settings.vlm_timeout_seconds + 2))
    result = await _call_model(content, timeout=timeout)
    result["source"] = "vlm"
    if coarse_screen:
        result["coarse_screen"] = coarse_screen
    return result


async def _call_model(content: list, timeout: int = 30) -> dict:
    provider = (settings.vlm_provider or "ark").lower()
    if provider == "deepseek":
        raw_text = await _call_openai_chat(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
            content,
            timeout,
            provider_label="DeepSeek",
        )
    elif provider in ("qwen", "dashscope", "tongyi"):
        raw_text = await _call_openai_chat(
            settings.qwen_api_key,
            settings.qwen_base_url,
            settings.qwen_model,
            content,
            timeout,
            provider_label="千问",
        )
    else:
        raw_text = await _call_ark(content, timeout)
    return _parse_result(raw_text)


def _ark_blocks_to_openai(content: list) -> list:
    blocks = []
    for part in content:
        ptype = part.get("type")
        if ptype == "input_text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "input_image":
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": part.get("image_url", "")},
                }
            )
    return blocks


async def _call_ark(content: list, timeout: int) -> str:
    if not settings.ark_api_key or not settings.ark_model_id:
        raise RuntimeError("豆包 Ark 未配置：请在 .env 设置 ARK_API_KEY 与 ARK_MODEL_ID")
    url = f"{settings.ark_base_url.rstrip('/')}/responses"
    headers = {
        "Authorization": f"Bearer {settings.ark_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.ark_model_id,
        "input": [{"role": "user", "content": content}],
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = response.text[:500] if response.text else str(e)
            raise RuntimeError(f"VLM 调用失败: {detail}") from e
        data = response.json()
    return _extract_ark_text(data)


async def _call_openai_chat(
    api_key: str,
    base_url: str,
    model: str,
    content: list,
    timeout: int,
    provider_label: str,
) -> str:
    if not api_key:
        raise RuntimeError(f"{provider_label} 未配置 API Key")
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _ark_blocks_to_openai(content)}],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = response.text[:500] if response.text else str(e)
            raise RuntimeError(f"{provider_label} VLM 调用失败: {detail}") from e
        data = response.json()
    return _extract_openai_chat_text(data)


def _extract_ark_text(data: dict) -> str:
    output = data.get("output", [])
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
    return ""


def _extract_openai_chat_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def _parse_result(text: str) -> dict:
    default = {
        "violation_detected": False,
        "violation_types": ["good"],
        "confidence": 0.0,
        "risk_description": "VLM 响应解析失败",
    }
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        result = json.loads(text)
        types = normalize_defect_ids(result.get("violation_types", []))
        if not types:
            types = ["good"]
        defect = bool(result.get("violation_detected", False))
        if "good" in types and len(types) > 1:
            types = [t for t in types if t != "good"]
        if not types:
            types = ["good"]
        if types == ["good"]:
            defect = False
        elif any(t != "good" for t in types):
            defect = True
        return {
            "violation_detected": defect,
            "violation_types": types,
            "confidence": float(result.get("confidence", 0.0)),
            "risk_description": str(result.get("risk_description", "")),
        }
    except (json.JSONDecodeError, ValueError):
        return default
