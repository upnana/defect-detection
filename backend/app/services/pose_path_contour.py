"""开发回退：轮廓 minAreaRect + PnP（非路线图主路径，POSE_CONTOUR_FALLBACK=1）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.pose_common import cv2, pack_pose_result, solve_pnp


def contour_corners_2d(image_path: Path) -> np.ndarray:
    cv = cv2()
    img = cv.imread(str(image_path))
    if img is None:
        raise RuntimeError("无法读取图像")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("轮廓检测失败")
    cnt = max(contours, key=cv.contourArea)
    if cv.contourArea(cnt) < 100:
        raise RuntimeError("轮廓过小")
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    return np.array(box[:4], dtype=np.float64)


def estimate_contour_pnp(
    image_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    names_3d: list[str],
    pts_3d: np.ndarray,
) -> dict:
    cv = cv2()
    img = cv.imread(str(image_path))
    if img is None:
        raise RuntimeError("无法读取图像")

    image_pts = contour_corners_2d(image_path)
    object_pts = pts_3d[:4]
    rvec, tvec, err = solve_pnp(object_pts, image_pts, K, dist)

    cnt_area = 0.0
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt_area = float(cv.contourArea(max(contours, key=cv.contourArea)))

    kp_list = [
        {"name": names_3d[i] if i < len(names_3d) else f"kp{i}", "u": round(float(image_pts[i, 0]), 1), "v": round(float(image_pts[i, 1]), 1)}
        for i in range(4)
    ]
    return pack_pose_result(
        "contour_pnp",
        rvec,
        tvec,
        err,
        kp_list,
        {"path": "contour_fallback", "contour_area_px": round(cnt_area, 1)},
    )
