"""Qwen2.5-VL / DashScope 兼容模式多模态调用（课件默认 VLM）。"""

from __future__ import annotations

import httpx

from screw_algo.settings import settings


def _to_openai_vision_content(content: list, *, image_first: bool = True) -> list:
    """Qwen-VL 推荐：先图后文。"""
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
    ordered = images + texts if image_first else texts + images
    return ordered


async def call_qwen_vl(
    content: list,
    *,
    timeout: int,
    json_mode: bool = True,
) -> str:
    api_key = settings.qwen_api_key
    if not api_key:
        raise RuntimeError(
            "Qwen-VL 未配置：请在 screw_algo/.env 设置 QWEN_API_KEY（或 DASHSCOPE_API_KEY）"
        )
    model = settings.qwen_model or "qwen2.5-vl-72b-instruct"
    url = f"{settings.qwen_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": _to_openai_vision_content(content, image_first=True),
            }
        ],
        "temperature": 0.1,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = response.text[:800] if response.text else str(e)
            raise RuntimeError(f"Qwen-VL 调用失败: {detail}") from e
        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def is_qwen_provider(provider: str | None) -> bool:
    p = (provider or settings.vlm_provider or "qwen").lower().strip()
    return p in (
        "qwen",
        "qwen-vl",
        "qwen2.5-vl",
        "qwen-vl-plus",
        "dashscope",
        "tongyi",
        "aliyun",
    )
