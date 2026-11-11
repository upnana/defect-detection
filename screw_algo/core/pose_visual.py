"""6D 位姿叠图：3D 框线 + RGB 坐标轴 + 关键点（课件可视化）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screw_algo.core.pose_common import cv2 as _cv2_mod, load_camera_matrix, load_screw_model_3d

# 3D AABB 12 条棱
_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _aabb_corners(points_mm: np.ndarray, margin: float = 1.5) -> np.ndarray:
    mn = points_mm.min(axis=0) - margin
    mx = points_mm.max(axis=0) + margin
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )


def _camera_rvec_tvec(pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cam = pose.get("camera_frame") or {}
    rvec = np.asarray(cam.get("rotation_vec"), dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(cam.get("translation_mm"), dtype=np.float64).reshape(3, 1)
    return rvec, tvec


def _project_points(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    cv = _cv2_mod()
    pts = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv.projectPoints(pts, rvec, tvec, K, dist)
    return projected.reshape(-1, 2)


def _draw_line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], th: int = 2) -> None:
    h, w = img.shape[:2]
    if max(p1[0], p2[0]) < 0 or max(p1[1], p2[1]) < 0 or min(p1[0], p2[0]) > w or min(p1[1], p2[1]) > h:
        return
    cv2.line(img, p1, p2, color, th, cv2.LINE_AA)


def draw_pose_6d_overlay(
    image_path: Path,
    pose_result: dict[str, Any],
    *,
    axis_length_mm: float | None = None,
) -> bytes:
    if not pose_result.get("success"):
        msg = pose_result.get("message") or "位姿估计失败"
        raise RuntimeError(msg)

    path = Path(image_path)
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"无法读取图像: {path}")

    _, pts_3d, _ = load_screw_model_3d()
    K, dist = load_camera_matrix(img.shape)
    rvec, tvec = _camera_rvec_tvec(pose_result)

    span = float(np.linalg.norm(pts_3d.max(axis=0) - pts_3d.min(axis=0)))
    axis_len = axis_length_mm if axis_length_mm is not None else max(10.0, span * 0.65)

    box_corners = _aabb_corners(pts_3d, margin=1.2)
    box_2d = _project_points(box_corners, rvec, tvec, K, dist).astype(np.int32)
    for i, j in _BOX_EDGES:
        _draw_line(img, tuple(box_2d[i]), tuple(box_2d[j]), (0, 255, 0), 2)

    head_2d = _project_points(pts_3d[:4], rvec, tvec, K, dist).astype(np.int32)
    cv2.polylines(img, [head_2d.reshape(-1, 1, 2)], True, (0, 220, 255), 2, cv2.LINE_AA)

    axes_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float64,
    )
    axes_2d = _project_points(axes_3d, rvec, tvec, K, dist).astype(np.int32)
    origin = tuple(axes_2d[0])
    _draw_line(img, origin, tuple(axes_2d[1]), (0, 0, 255), 2)
    _draw_line(img, origin, tuple(axes_2d[2]), (0, 255, 0), 2)
    _draw_line(img, origin, tuple(axes_2d[3]), (255, 0, 0), 2)
    cv2.putText(img, "X", tuple(axes_2d[1] + (4, -4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Y", tuple(axes_2d[2] + (4, -4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Z", tuple(axes_2d[3] + (4, -4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    for kp in pose_result.get("keypoints_2d") or []:
        u, v = int(kp.get("u", 0)), int(kp.get("v", 0))
        cv2.circle(img, (u, v), 4, (255, 255, 0), -1, cv2.LINE_AA)

    method = pose_result.get("method") or "pose"
    reproj = pose_result.get("reprojection_error_px")
    cam = pose_result.get("camera_frame") or {}
    euler = cam.get("rotation_deg") or [0, 0, 0]
    t_mm = cam.get("translation_mm") or [0, 0, 0]
    line1 = f"6D [{method}] reproj={reproj}px" if reproj is not None else f"6D [{method}]"
    line2 = f"t=({t_mm[0]:.1f},{t_mm[1]:.1f},{t_mm[2]:.1f})mm  r=({euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f})deg"
    cv2.putText(img, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, line2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, line2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()
