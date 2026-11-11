"""VLM 输入：将 YOLO 粗筛框绘制到图像（课件：画框后送 Qwen2.5-VL）。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def draw_coarse_detections(
    image_path: Path,
    coarse_screen: dict[str, Any] | None,
    variety: dict[str, Any] | None = None,
) -> bytes:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    detections = (coarse_screen or {}).get("detections") or []
    pending = (variety or {}).get("class_id") == "variety_pending" or (variety or {}).get(
        "general_detection_head"
    )
    header = "GENERAL" if pending else "YOLO"
    for d in detections[:12]:
        conf = d.get("confidence", 0)
        label = f"{header}:{d.get('class_id')} {conf:.2f}"
        obb = d.get("obb_corners")
        if obb and len(obb) >= 8:
            pts = np.array(obb, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(img, [pts], True, (0, 165, 255), 2)
            x, y = int(pts[0, 0]), int(pts[0, 1]) - 4
        else:
            box = d.get("bbox") or []
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x, y = x1, max(0, y1 - 6)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    if variety and variety.get("label_zh"):
        cv2.putText(
            img,
            f"Variety: {variety.get('label_zh')} ({variety.get('class_id')})",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def image_path_to_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    enc = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{enc}"


def annotated_data_url(
    image_path: Path,
    coarse_screen: dict[str, Any] | None,
    variety: dict[str, Any] | None = None,
) -> str:
    jpeg = draw_coarse_detections(image_path, coarse_screen, variety)
    enc = base64.b64encode(jpeg).decode("utf-8")
    return f"data:image/jpeg;base64,{enc}"
