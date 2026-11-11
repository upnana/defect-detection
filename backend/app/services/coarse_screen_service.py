"""YOLOv8 粗筛：双阈值召回 / 触发 VLM 精查。未配置权重时跳过粗筛。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import settings
from app.defect_types import DEFECT_ONLY_IDS, normalize_defect_ids

_yolo_model = None


def _resolve_image_path(image_url: str) -> Path | None:
    if not image_url:
        return None
    if image_url.startswith("/uploads/"):
        path = Path(settings.upload_dir) / image_url.removeprefix("/uploads/")
        return path if path.exists() else None
    candidate = Path(image_url)
    return candidate if candidate.exists() else None


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not settings.yolo_model_path:
        return None
    raw = Path(settings.yolo_model_path)
    backend_root = Path(__file__).resolve().parent.parent.parent
    model_path = raw if raw.is_absolute() else (backend_root / raw)
    if not model_path.exists():
        return None
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("已配置 YOLO_MODEL_PATH 但未安装 ultralytics，请 pip install ultralytics") from e
    _yolo_model = YOLO(str(model_path))
    return _yolo_model


def coarse_screen(image_url: str) -> dict[str, Any]:
    """
    返回:
      enabled: 是否执行了 YOLO
      skip_vlm: 粗筛判定为明显良品，可跳过 VLM
      trigger_vlm: 是否应进入 VLM 精查
      detections: 候选框列表
      primary_class: 最高置信类别 id
      max_confidence: 最高置信度
    """
    base = {
        "enabled": False,
        "skip_vlm": False,
        "trigger_vlm": True,
        "detections": [],
        "primary_class": None,
        "max_confidence": 0.0,
        "message": "未配置 YOLO 权重，跳过粗筛",
    }

    model = _load_yolo()
    if model is None:
        return base

    image_path = _resolve_image_path(image_url)
    if image_path is None:
        base["message"] = "无法解析本地图片路径，跳过粗筛"
        return base

    recall_th = settings.yolo_recall_threshold
    vlm_th = settings.yolo_vlm_trigger_threshold

    results = model.predict(source=str(image_path), verbose=False)
    detections: list[dict] = []
    max_conf = 0.0
    primary_class = None

    names = getattr(model, "names", {}) or {}

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name not in DEFECT_ONLY_IDS and cls_name != "good":
                cls_name = str(cls_name).strip()

            if conf < recall_th:
                continue

            xyxy = boxes.xyxy[i].tolist()
            detections.append(
                {
                    "class_id": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in xyxy],
                }
            )
            if conf > max_conf:
                max_conf = conf
                primary_class = cls_name

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    skip_vlm = max_conf > 0 and max_conf < vlm_th and primary_class == "good"
    trigger_vlm = not skip_vlm and (max_conf >= vlm_th or any(d["class_id"] in DEFECT_ONLY_IDS for d in detections))

    return {
        "enabled": True,
        "skip_vlm": skip_vlm,
        "trigger_vlm": trigger_vlm,
        "detections": detections,
        "primary_class": primary_class,
        "max_confidence": round(max_conf, 4),
        "message": "YOLO 粗筛完成",
        "defect_candidates": normalize_defect_ids([d["class_id"] for d in detections if d["class_id"] in DEFECT_ONLY_IDS]),
    }


def coarse_result_to_ai_result(screen: dict) -> dict:
    """粗筛快速放行时构造与 VLM 兼容的分析结果。"""
    if screen.get("primary_class") == "good" and screen.get("skip_vlm"):
        return {
            "violation_detected": False,
            "violation_types": ["good"],
            "confidence": max(0.05, 1.0 - float(screen.get("max_confidence", 0))),
            "risk_description": "YOLO 粗筛：未见可疑缺陷，毫秒级良品放行",
            "source": "yolo_fast_pass",
            "coarse_screen": screen,
        }
    defects = screen.get("defect_candidates") or []
    return {
        "violation_detected": bool(defects),
        "violation_types": defects or (["good"] if screen.get("primary_class") == "good" else []),
        "confidence": float(screen.get("max_confidence", 0)),
        "risk_description": "YOLO 粗筛候选（待 VLM 确认或超时回退）",
        "source": "yolo_coarse",
        "coarse_screen": screen,
    }
