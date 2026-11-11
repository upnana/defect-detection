"""手眼标定：相机坐标系 → 机器人基坐标系 4×4 齐次变换。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import settings

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def _resolve_config_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    candidate = BACKEND_DIR / path_str
    return candidate if candidate.exists() else p


def _parse_matrix(raw: str | list) -> np.ndarray:
    if isinstance(raw, str):
        text = raw.strip()
        if Path(text).exists():
            data = json.loads(Path(text).read_text(encoding="utf-8"))
        else:
            data = json.loads(text)
    else:
        data = raw

    if isinstance(data, dict) and "matrix" in data:
        data = data["matrix"]
    arr = np.array(data, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError("手眼矩阵必须为 4×4 或 16 个元素")


def load_hand_eye_matrix() -> np.ndarray:
    if settings.hand_eye_matrix_json:
        return _parse_matrix(settings.hand_eye_matrix_json)
    if settings.hand_eye_matrix_path:
        path = _resolve_config_path(settings.hand_eye_matrix_path)
        if path.exists():
            return _parse_matrix(path.read_text(encoding="utf-8"))
    return np.eye(4, dtype=np.float64)


def transform_pose_to_robot(tvec_mm: np.ndarray, rvec: np.ndarray) -> dict:
    """将 OpenCV 相机系位姿 (rvec,tvec) 变换到机器人基坐标系。"""
    R_obj_cam, _ = _import_cv2().Rodrigues(rvec)
    t = np.asarray(tvec_mm, dtype=np.float64).reshape(3)

    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R_obj_cam
    T_cam_obj[:3, 3] = t

    T_hand_eye = load_hand_eye_matrix()
    T_robot_obj = T_hand_eye @ T_cam_obj

    R = T_robot_obj[:3, :3]
    t_robot = T_robot_obj[:3, 3]
    rvec_robot, _ = _import_cv2().Rodrigues(R)

    return {
        "translation_mm": [round(float(x), 3) for x in t_robot],
        "rotation_vec": [round(float(x), 4) for x in rvec_robot.flatten()],
        "rotation_deg": rvec_to_euler_deg(rvec_robot),
        "transform_4x4": T_robot_obj.round(4).tolist(),
    }


def rotation_matrix_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))


def rvec_to_euler_deg(rvec: np.ndarray) -> list[float]:
    R, _ = _import_cv2().Rodrigues(np.asarray(rvec, dtype=np.float64))
    rx, ry, rz = rotation_matrix_to_euler_deg(R)
    return [round(rx, 2), round(ry, 2), round(rz, 2)]


def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Hamilton 四元数，标量在前 (w, x, y, z)。"""
    m = np.asarray(R, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def build_pose_6d(
    translation_mm: np.ndarray | list[float],
    rvec: np.ndarray,
    frame: str,
) -> dict:
    """
    标准 6D：平移 (X,Y,Z) mm + 旋转（欧拉角 / yaw-pitch-roll / 四元数）。
    欧拉为绕固定轴 X→Y→Z 的 rx, ry, rz（度）；yaw/pitch/roll 对应 rz, ry, rx。
    """
    t = np.asarray(translation_mm, dtype=np.float64).reshape(3)
    R, _ = _import_cv2().Rodrigues(np.asarray(rvec, dtype=np.float64))
    rx, ry, rz = rotation_matrix_to_euler_deg(R)
    qw, qx, qy, qz = rotation_matrix_to_quaternion_wxyz(R)

    return {
        "frame": frame,
        "translation_mm": {
            "x": round(float(t[0]), 3),
            "y": round(float(t[1]), 3),
            "z": round(float(t[2]), 3),
            "unit": "mm",
        },
        "rotation": {
            "euler_xyz_deg": {
                "rx": round(rx, 2),
                "ry": round(ry, 2),
                "rz": round(rz, 2),
                "unit": "deg",
                "convention": "intrinsic XYZ (OpenCV Rodrigues → R → Euler)",
            },
            "yaw_pitch_roll_deg": {
                "yaw": round(rz, 2),
                "pitch": round(ry, 2),
                "roll": round(rx, 2),
                "unit": "deg",
            },
            "quaternion_wxyz": {
                "w": round(qw, 5),
                "x": round(qx, 5),
                "y": round(qy, 5),
                "z": round(qz, 5),
            },
        },
        "vector_6d": [
            round(float(t[0]), 3),
            round(float(t[1]), 3),
            round(float(t[2]), 3),
            round(rx, 2),
            round(ry, 2),
            round(rz, 2),
        ],
        "vector_6d_desc": "tx_mm, ty_mm, tz_mm, rx_deg, ry_deg, rz_deg",
    }


_cv2 = None


def _import_cv2():
    global _cv2
    if _cv2 is None:
        import cv2

        _cv2 = cv2
    return _cv2
