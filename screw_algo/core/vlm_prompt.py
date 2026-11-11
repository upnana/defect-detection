"""VLM 精查 Prompt：课件固定模板（assets）+ 动态品种/粗筛附录。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from screw_algo.defect_types import DEFECT_CLASSES
from screw_algo.settings import PKG_ROOT, resolve_pkg_path, settings

JSON_OUTPUT_SPEC = """仅输出一个 JSON 对象（字段名固定）：
{
  "violation_detected": boolean,
  "violation_types": ["good"] 或缺陷 id 数组,
  "confidence": 0.0-1.0,
  "risk_description": "string"
}"""


def _defect_class_table() -> str:
    return "\n".join(
        f'- "{c["id"]}": {c["label_zh"]} — {c["description"]}' for c in DEFECT_CLASSES
    )


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    path = resolve_pkg_path(settings.vlm_prompt_path)
    if not path.exists():
        path = PKG_ROOT / "assets" / "vlm_screw_inspect_prompt.txt"
    if not path.exists():
        raise FileNotFoundError(f"VLM Prompt 模板不存在: {path}")
    return path.read_text(encoding="utf-8")


def base_inspect_prompt() -> str:
    tpl = _load_prompt_template()
    return tpl.replace("{{DEFECT_CLASS_TABLE}}", _defect_class_table()).replace(
        "{{JSON_OUTPUT_SPEC}}", JSON_OUTPUT_SPEC
    )


def build_vlm_prompt(
    coarse_screen: dict[str, Any] | None,
    variety: dict[str, Any] | None,
) -> str:
    parts = [base_inspect_prompt()]
    if variety and variety.get("enabled"):
        conf = variety.get("confidence")
        conf_s = f"{conf:.4f}" if isinstance(conf, (int, float)) else conf
        parts.append(
            f"\n【附录 · 品种分类】\n"
            f"标签：{variety.get('label_zh')} | class_id：{variety.get('class_id')} | 置信度：{conf_s}"
        )
        if variety.get("general_detection_head") or variety.get("class_id") == settings.variety_pending_class_id:
            parts.append(
                "说明：品种待确认，已启用通用 YOLO 检测头；请勿仅依赖品种标签，以图像与框为准独立判断。"
            )
    if coarse_screen and coarse_screen.get("enabled"):
        detections = coarse_screen.get("detections") or []
        if not detections:
            parts.append("\n【附录 · YOLO 粗筛】\n未给出高置信候选框，请对整图独立完整判断。")
        else:
            lines = []
            for d in detections[:8]:
                obb = d.get("obb_corners")
                box_desc = obb if obb else d.get("bbox")
                lines.append(
                    f"- class_id={d.get('class_id')} confidence={d.get('confidence')} "
                    f"geometry={box_desc} detector={d.get('detector', 'yolo')}"
                )
            parts.append("\n【附录 · YOLO 粗筛候选（与画框图一致）】\n" + "\n".join(lines))
    return "\n".join(parts)


def clear_prompt_template_cache() -> None:
    _load_prompt_template.cache_clear()
