"""YOLOv8 粗筛：OBB/检测、按品种选权重、双阈值。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from screw_algo.core.yolo_routing import resolve_coarse_model_path
from screw_algo.defect_types import DEFECT_ONLY_IDS, normalize_defect_ids
from screw_algo.settings import resolve_pkg_path, settings

_yolo_models: dict[str, Any] = {}


def _resolve_image_path(image_url: str) -> Path | None:
    if not image_url:
        return None
    candidate = Path(image_url)
    return candidate if candidate.exists() else None


def _load_yolo(model_path: Path):
    key = str(model_path.resolve())
    if key in _yolo_models:
        return _yolo_models[key]
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    _yolo_models[key] = model
    return model


def _parse_detections(result, names: dict, recall_th: float, task: str) -> list[dict]:
    detections: list[dict] = []
    used_obb = False

    obb = getattr(result, "obb", None)
    if task in ("obb", "auto") and obb is not None and len(obb) > 0:
        used_obb = True
        for i in range(len(obb)):
            conf = float(obb.conf[i].item())
            if conf < recall_th:
                continue
            cls_id = int(obb.cls[i].item())
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name not in DEFECT_ONLY_IDS and cls_name != "good":
                cls_name = str(cls_name).strip()
            corners = obb.xyxyxyxy[i].cpu().numpy().reshape(-1).tolist()
            xyxy = obb.xyxy[i].tolist() if hasattr(obb, "xyxy") else None
            if xyxy is None and len(corners) >= 8:
                xs = corners[0::2]
                ys = corners[1::2]
                xyxy = [min(xs), min(ys), max(xs), max(ys)]
            detections.append(
                {
                    "class_id": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in (xyxy or corners[:4])],
                    "obb_corners": [round(x, 2) for x in corners],
                    "detector": "yolov8_obb",
                }
            )

    if not used_obb:
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                if conf < recall_th:
                    continue
                cls_id = int(boxes.cls[i].item())
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name not in DEFECT_ONLY_IDS and cls_name != "good":
                    cls_name = str(cls_name).strip()
                xyxy = boxes.xyxy[i].tolist()
                detections.append(
                    {
                        "class_id": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(x, 2) for x in xyxy],
                        "detector": "yolov8_detect",
                    }
                )

    return detections


def coarse_screen(image_url: str, variety: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "enabled": False,
        "skip_vlm": False,
        "trigger_vlm": True,
        "detections": [],
        "primary_class": None,
        "max_confidence": 0.0,
        "message": "未配置 YOLO 权重，跳过粗筛",
        "yolo_task": settings.yolo_coarse_task,
    }

    model_path, route_reason, general_head = resolve_coarse_model_path(variety)
    if model_path is None:
        fallback = resolve_pkg_path(settings.yolo_model_path)
        if fallback.exists():
            model_path = fallback
            route_reason = "回退 YOLO_MODEL_PATH"
        else:
            return base

    model = _load_yolo(model_path)
    image_path = _resolve_image_path(image_url)
    if image_path is None:
        base["message"] = "无法解析本地图片路径，跳过粗筛"
        return base

    recall_th = settings.yolo_recall_threshold
    vlm_th = settings.yolo_vlm_trigger_threshold
    task = (settings.yolo_coarse_task or "auto").lower()

    results = model.predict(source=str(image_path), verbose=False)
    detections: list[dict] = []
    max_conf = 0.0
    primary_class = None
    names = getattr(model, "names", {}) or {}

    for result in results:
        detections.extend(_parse_detections(result, names, recall_th, task))

    for d in detections:
        conf = d["confidence"]
        if conf > max_conf:
            max_conf = conf
            primary_class = d["class_id"]

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    skip_vlm = max_conf > 0 and max_conf < vlm_th and primary_class == "good"
    trigger_vlm = not skip_vlm and (max_conf >= vlm_th or any(d["class_id"] in DEFECT_ONLY_IDS for d in detections))

    det_mode = detections[0]["detector"] if detections else ("yolov8_obb" if task == "obb" else "yolov8_detect")

    return {
        "enabled": True,
        "skip_vlm": skip_vlm,
        "trigger_vlm": trigger_vlm,
        "detections": detections,
        "primary_class": primary_class,
        "max_confidence": round(max_conf, 4),
        "message": "YOLO 粗筛完成",
        "defect_candidates": normalize_defect_ids(
            [d["class_id"] for d in detections if d["class_id"] in DEFECT_ONLY_IDS]
        ),
        "model_weights": str(model_path),
        "route_reason": route_reason,
        "general_detection_head": general_head,
        "detector_mode": det_mode,
    }


def coarse_result_to_ai_result(screen: dict) -> dict:
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
