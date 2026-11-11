"""6D 位姿公共：相机内参、3D 模型、PnP、结果打包。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from screw_algo.core import hand_eye
from screw_algo.settings import PKG_ROOT, resolve_pkg_path, settings

ASSETS_DIR = PKG_ROOT / "assets"


def resolve_image_path(image_url: str) -> Path | None:
    if not image_url:
        return None
    p = Path(image_url)
    return p if p.exists() else None


def cv2():
    import cv2

    return cv2


def load_camera_matrix(image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape[:2]
    if settings.camera_intrinsics_json:
        raw = settings.camera_intrinsics_json.strip()
        data = json.loads(Path(raw).read_text(encoding="utf-8")) if Path(raw).exists() else json.loads(raw)
        K = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
        return K, dist
    if settings.camera_intrinsics_path:
        path = resolve_pkg_path(settings.camera_intrinsics_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            K = np.array(data["camera_matrix"], dtype=np.float64)
            dist = np.array(data.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
            return K, dist

    f = max(w, h) * 0.9
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def load_screw_model_3d() -> tuple[list[str], np.ndarray, dict]:
    path = (
        resolve_pkg_path(settings.screw_model_3d_path)
        if settings.screw_model_3d_path
        else ASSETS_DIR / "screw_keypoints_3d.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data.get("keypoint_names", [])
    pts = np.array(data["points_mm"], dtype=np.float64)
    return names, pts, data


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    cv = cv2()
    n = len(object_points)
    if n >= 6:
        flags = cv.SOLVEPNP_ITERATIVE
    elif n == 4:
        flags = cv.SOLVEPNP_IPPE
    else:
        flags = cv.SOLVEPNP_EPNP

    ok, rvec, tvec = cv.solvePnP(
        object_points.astype(np.float64),
        image_points.astype(np.float64),
        K,
        dist,
        flags=flags,
    )
    if not ok:
        raise RuntimeError("solvePnP 失败")

    projected, _ = cv.projectPoints(object_points, rvec, tvec, K, dist)
    err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))
    return rvec, tvec, err


def reprojection_error_vs_points(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    cv = cv2()
    projected, _ = cv.projectPoints(object_points.astype(np.float64), rvec, tvec, K, dist)
    return float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points.astype(np.float64), axis=1)))


def rot6d_to_rvec(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[:3].astype(np.float64)
    a2 = rot6d[3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    cv = cv2()
    rvec, _ = cv.Rodrigues(R)
    return rvec


def pack_pose_result(
    method: str,
    rvec: np.ndarray,
    tvec: np.ndarray,
    reproj_err: float,
    keypoints_2d: list[dict] | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    cv = cv2()
    t_mm = tvec.reshape(3).astype(np.float64)
    robot = hand_eye.transform_pose_to_robot(t_mm, rvec)
    R, _ = cv.Rodrigues(rvec)
    conf = max(0.05, min(0.99, 1.0 - reproj_err / 20.0))

    rvec_robot = np.array(robot["rotation_vec"], dtype=np.float64).reshape(3, 1)
    t_robot = np.array(robot["translation_mm"], dtype=np.float64)
    pose_6d_camera = hand_eye.build_pose_6d(t_mm, rvec, "camera")
    pose_6d_robot = hand_eye.build_pose_6d(t_robot, rvec_robot, "robot_base")

    result: dict[str, Any] = {
        "success": True,
        "method": method,
        "confidence": round(conf, 3),
        "reprojection_error_px": round(reproj_err, 3),
        "pose_6d": pose_6d_robot,
        "pose_6d_camera": pose_6d_camera,
        "pose_6d_robot": pose_6d_robot,
        "camera_frame": {
            "translation_mm": [round(float(x), 3) for x in t_mm],
            "rotation_vec": [round(float(x), 4) for x in rvec.flatten()],
            "rotation_deg": hand_eye.rvec_to_euler_deg(rvec),
            "rotation_matrix": R.round(4).tolist(),
            "pose_6d": pose_6d_camera,
        },
        "robot_frame": {**robot, "pose_6d": pose_6d_robot},
        "keypoints_2d": keypoints_2d or [],
    }
    if extra:
        result.update(extra)
    return result
