"""按品种 / 品种待确认 选择 YOLO 粗筛权重（课件：按品种缺陷检测 + 通用头）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from screw_algo.settings import PKG_ROOT, resolve_pkg_path, settings

_cache: dict | None = None


def _load_map() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    path = resolve_pkg_path(settings.variety_yolo_models_path)
    if not path.exists():
        path = PKG_ROOT / "assets" / "variety_yolo_models.json"
    _cache = json.loads(path.read_text(encoding="utf-8"))
    return _cache


def resolve_coarse_model_path(variety: dict[str, Any] | None) -> tuple[Path | None, str, bool]:
    """
    返回 (权重路径, 选用原因, 是否通用检测头).
    """
    if not settings.yolo_model_path and not _load_map().get("general_detection"):
        return None, "未配置", False

    data = _load_map()
    pending_id = settings.variety_pending_class_id or "variety_pending"
    use_obb = (settings.yolo_coarse_task or "auto").lower() in ("obb", "auto")
    variety = variety or {}
    class_id = variety.get("class_id")
    low_conf = variety.get("low_confidence") or variety.get("general_detection_head")
    is_pending = class_id == pending_id or low_conf or (
        class_id is None and variety.get("enabled") and variety.get("confidence", 1) < settings.variety_min_confidence
    )

    def _pick(key: str) -> str:
        raw = data.get(key) or data.get("general_detection") or settings.yolo_model_path
        return str(raw)

    if is_pending:
        if use_obb:
            raw = _pick(data.get("variety_pending_obb_model_key") or "obb_general_detection")
            p = resolve_pkg_path(raw)
            if p.exists():
                return p, "品种待确认 → 通用 OBB 检测头", True
        raw = _pick(data.get("variety_pending_model_key") or "general_detection")
        p = resolve_pkg_path(raw)
        return (p if p.exists() else None), "品种待确认 → 通用检测头", True

    if class_id and class_id in (data.get("by_class_id") or {}):
        raw = data["by_class_id"][class_id]
        p = resolve_pkg_path(raw)
        if p.exists():
            return p, f"按品种 {class_id} 选用专用权重", False

    if use_obb:
        raw = _pick("obb_general_detection")
        p = resolve_pkg_path(raw)
        if p.exists():
            return p, "默认 OBB 粗筛", False

    raw = _pick("default_by_variety") or settings.yolo_model_path
    p = resolve_pkg_path(raw)
    return (p if p.exists() else None), "默认粗筛权重", False
