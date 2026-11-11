"""路径 1：YOLOv8-Pose 2D 关键点 + 3D 模型角点 + solvePnP。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import settings
from app.services.pose_common import pack_pose_result, resolve_backend_path, solve_pnp

_yolo_pose_model = None


def _load_yolo_pose():
    global _yolo_pose_model
    if _yolo_pose_model is not None:
        return _yolo_pose_model
    if not settings.yolo_pose_model_path:
        return None
    model_path = resolve_backend_path(settings.yolo_pose_model_path)
    if not model_path.exists():
        return None
    from ultralytics import YOLO

    _yolo_pose_model = YOLO(str(model_path))
    return _yolo_pose_model


def is_configured() -> bool:
    if not settings.yolo_pose_model_path:
        return False
    return resolve_backend_path(settings.yolo_pose_model_path).exists()


def estimate_yolo_pnp(
    image_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    names_3d: list[str],
    pts_3d: np.ndarray,
    model_meta: dict,
) -> dict:
    model = _load_yolo_pose()
    if model is None:
        raise RuntimeError("未配置 YOLOv8-Pose 权重（YOLO_POSE_MODEL_PATH）")

    index_map = model_meta.get("yolo_keypoint_index_map", list(range(len(names_3d))))
    min_kpt_conf = float(getattr(settings, "yolo_pose_min_kpt_conf", 0.25) or 0.25)

    results = model.predict(source=str(image_path), verbose=False)
    best = None
    best_conf = -1.0

    for res in results:
        kpts = getattr(res, "keypoints", None)
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            continue
        for ki in range(len(kpts.xy)):
            conf = float(kpts.conf[ki].mean()) if kpts.conf is not None else 0.5
            if conf > best_conf:
                best_conf = conf
                best = kpts.xy[ki].cpu().numpy()

    if best is None or best_conf < min_kpt_conf:
        raise RuntimeError("YOLOv8-Pose 未检测到足够置信度的关键点")

    n = min(len(index_map), len(pts_3d), best.shape[0])
    idx = index_map[:n]
    image_pts = best[idx, :2]
    object_pts = pts_3d[:n]

    rvec, tvec, err = solve_pnp(object_pts, image_pts, K, dist)
    kp_list = [
        {
            "name": names_3d[i] if i < len(names_3d) else f"kp{i}",
            "u": round(float(image_pts[i, 0]), 1),
            "v": round(float(image_pts[i, 1]), 1),
        }
        for i in range(n)
    ]
    return pack_pose_result(
        "yolo_pnp",
        rvec,
        tvec,
        err,
        kp_list,
        {
            "path": "yolov8_pose_pnp",
            "yolo_pose_confidence": round(best_conf, 3),
            "model_weights": str(resolve_backend_path(settings.yolo_pose_model_path)),
        },
    )
