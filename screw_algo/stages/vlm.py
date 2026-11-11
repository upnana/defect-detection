from __future__ import annotations

import asyncio
from typing import Any

from screw_algo.core.coarse import coarse_result_to_ai_result
from screw_algo.core.vlm import analyze_screw_image


async def run_vlm_async(
    image_path: str,
    coarse: dict[str, Any],
    timeout_s: float,
    variety: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if coarse.get("skip_vlm"):
        out = coarse_result_to_ai_result(coarse)
        out["stage"] = "coarse_fast_pass"
        return out

    try:
        out = await asyncio.wait_for(
            analyze_screw_image(image_path, coarse, variety),
            timeout=timeout_s,
        )
        out["stage"] = "vlm"
        return out
    except asyncio.TimeoutError:
        out = coarse_result_to_ai_result(coarse)
        out["risk_description"] = f"VLM 超时（>{timeout_s}s），回退粗筛"
        out["source"] = "yolo_fallback_timeout"
        out["stage"] = "vlm_timeout_fallback"
        return out


def run_vlm(
    image_path: str,
    coarse: dict[str, Any],
    timeout_s: float,
    variety: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return asyncio.run(run_vlm_async(image_path, coarse, timeout_s, variety))
