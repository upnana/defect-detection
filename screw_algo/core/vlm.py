import base64
import json
import mimetypes
from pathlib import Path

import httpx

from screw_algo.core.inspection_schema import validate_inspection_json
from screw_algo.core.vlm_prompt import build_vlm_prompt
from screw_algo.core.vlm_qwen import call_qwen_vl, is_qwen_provider
from screw_algo.core.vlm_visual import annotated_data_url
from screw_algo.defect_types import normalize_defect_ids
from screw_algo.settings import settings


def _resolve_image_path(media_url: str) -> Path | None:
    if media_url.startswith("/uploads/"):
        return None
    p = Path(media_url)
    return p if p.exists() else None


def _media_ref_for_vlm(
    image_url: str,
    coarse_screen: dict | None,
    variety: dict | None,
) -> str:
    path = _resolve_image_path(image_url)
    if path is None:
        return image_url if image_url.startswith(("http://", "https://", "data:")) else image_url

    if settings.vlm_draw_yolo_boxes and coarse_screen and coarse_screen.get("detections"):
        return annotated_data_url(path, coarse_screen, variety)

    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


async def analyze_screw_image(
    image_url: str,
    coarse_screen: dict | None = None,
    variety: dict | None = None,
) -> dict:
    image_ref = _media_ref_for_vlm(image_url, coarse_screen, variety)
    prompt = build_vlm_prompt(coarse_screen, variety)
    content = [
        {"type": "input_image", "image_url": image_ref},
        {"type": "input_text", "text": prompt},
    ]
    timeout = int(max(5, settings.vlm_timeout_seconds + 2))
    provider = (settings.vlm_provider or "qwen").lower()
    result = await _call_model(content, timeout=timeout, provider=provider)
    result["source"] = "vlm"
    result["vlm_provider"] = provider
    result["vlm_model"] = _model_name_for_provider(provider)
    result["vlm_input_annotated"] = bool(settings.vlm_draw_yolo_boxes)
    result["vlm_prompt_chars"] = len(prompt)
    if coarse_screen:
        result["coarse_screen"] = coarse_screen
    if variety:
        result["variety_context"] = {
            "class_id": variety.get("class_id"),
            "label_zh": variety.get("label_zh"),
        }
    return result


def _model_name_for_provider(provider: str) -> str | None:
    if is_qwen_provider(provider):
        return settings.qwen_model
    if provider == "deepseek":
        return settings.deepseek_model
    if provider == "ark":
        return settings.ark_model_id or None
    return None


async def _call_model(content: list, timeout: int = 30, provider: str | None = None) -> dict:
    provider = (provider or settings.vlm_provider or "qwen").lower()
    use_json = settings.vlm_json_schema_validate

    if is_qwen_provider(provider):
        raw_text = await call_qwen_vl(content, timeout=timeout, json_mode=use_json)
    elif provider == "deepseek":
        raw_text = await _call_openai_chat(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
            content,
            timeout,
            provider_label="DeepSeek",
            json_mode=use_json,
            image_first=False,
        )
    else:
        raw_text = await _call_ark(content, timeout)
    return _parse_result(raw_text)


def _ark_blocks_to_openai(content: list, *, image_first: bool = False) -> list:
    images = []
    texts = []
    for part in content:
        ptype = part.get("type")
        if ptype == "input_text":
            texts.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "input_image":
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": part.get("image_url", "")},
                }
            )
    return (images + texts) if image_first else (texts + images)


async def _call_ark(content: list, timeout: int) -> str:
    if not settings.ark_api_key or not settings.ark_model_id:
        raise RuntimeError("豆包 Ark 未配置：请在 screw_algo/.env 设置 ARK_API_KEY 与 ARK_MODEL_ID")
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
    json_mode: bool = False,
    image_first: bool = False,
) -> str:
    if not api_key:
        raise RuntimeError(f"{provider_label} 未配置 API Key")
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "user", "content": _ark_blocks_to_openai(content, image_first=image_first)}
        ],
        "temperature": 0.2,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
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
        "schema_valid": False,
    }
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        raw = json.loads(text)
        if settings.vlm_json_schema_validate:
            out = validate_inspection_json(raw)
            out["schema_valid"] = True
            return out
        result = raw
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
            "schema_valid": False,
        }
    except (json.JSONDecodeError, ValueError):
        return default
