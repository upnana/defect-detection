"""6D 位姿估计 — 双路径 + 可选轮廓回退。"""

from __future__ import annotations

from typing import Any

from screw_algo.core import pose_contour, pose_tinydetr, pose_yolo
from screw_algo.core.pose_common import cv2, load_camera_matrix, load_screw_model_3d, resolve_image_path
from screw_algo.settings import settings


def estimate_pose_6d(image_url: str) -> dict[str, Any]:
    if not settings.pose_enabled:
        return {
            "success": False,
            "method": "disabled",
            "message": "位姿估计已在配置中关闭（POSE_ENABLED=0）",
        }

    image_path = resolve_image_path(image_url)
    if image_path is None:
        return {"success": False, "method": "none", "message": "无法定位图像文件"}

    cv = cv2()
    img = cv.imread(str(image_path))
    if img is None:
        return {"success": False, "method": "none", "message": "图像读取失败"}

    names_3d, pts_3d, model_meta = load_screw_model_3d()
    K, dist = load_camera_matrix(img.shape)

    mode = (settings.pose_mode or "auto").lower()
    errors: list[str] = []

    def _try_yolo() -> dict | None:
        if not pose_yolo.is_configured():
            if mode == "yolo_pnp":
                raise RuntimeError("未配置 YOLO_POSE_MODEL_PATH")
            return None
        try:
            return pose_yolo.estimate_yolo_pnp(image_path, K, dist, names_3d, pts_3d, model_meta)
        except Exception as e:
            errors.append(f"路径1 YOLOv8-Pose+PnP: {e}")
            if mode == "yolo_pnp":
                raise
            return None

    def _try_tinydetr() -> dict | None:
        if not pose_tinydetr.is_configured():
            if mode == "tinydetr_pose":
                raise RuntimeError("未配置 TINYDETR_POSE_MODEL_PATH")
            return None
        try:
            return pose_tinydetr.estimate_tinydetr_pose(image_path, K, dist, names_3d, pts_3d)
        except Exception as e:
            errors.append(f"路径2 TinyDETR-Pose: {e}")
            if mode == "tinydetr_pose":
                raise
            return None

    if mode in ("auto", "yolo_pnp", "yolo"):
        result = _try_yolo()
        if result is not None:
            return result

    if mode in ("auto", "tinydetr_pose", "tinydetr"):
        result = _try_tinydetr()
        if result is not None:
            return result

    if mode in ("contour_pnp", "contour", "simulation") or settings.pose_contour_fallback:
        try:
            return pose_contour.estimate_contour_pnp(image_path, K, dist, names_3d, pts_3d)
        except Exception as e:
            errors.append(f"轮廓回退: {e}")

    return {
        "success": False,
        "method": mode,
        "message": "6D 位姿失败：请配置路径1（YOLO_POSE_MODEL_PATH）或路径2（TINYDETR_POSE_MODEL_PATH）",
        "errors": errors,
    }
