"""路径 2：TinyDETR-Pose 端到端回归 6D（相机系 t + rot6d）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from screw_algo.core.pose_common import (
    cv2,
    pack_pose_result,
    reprojection_error_vs_points,
    rot6d_to_rvec,
)
from screw_algo.core.pose_contour import contour_corners_2d
from screw_algo.ml.tinydetr_pose_net import TinyDETRPoseNet
from screw_algo.settings import resolve_pkg_path, settings

_model = None
_norm: dict | None = None
_device = "cpu"


def _load_checkpoint():
    global _model, _norm, _device
    if _model is not None:
        return _model, _norm

    if not settings.tinydetr_pose_model_path:
        return None, None

    ckpt_path = resolve_pkg_path(settings.tinydetr_pose_model_path)
    if not ckpt_path.exists():
        return None, None

    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    net = TinyDETRPoseNet(out_dim=int(ckpt.get("out_dim", 9)))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.to(_device)
    _norm = ckpt.get("norm") or {}
    _model = net
    return _model, _norm


def is_configured() -> bool:
    if not settings.tinydetr_pose_model_path:
        return False
    return resolve_pkg_path(settings.tinydetr_pose_model_path).exists()


def _preprocess(image_path: Path) -> tuple[np.ndarray, "torch.Tensor"]:
    import torch
    from torchvision import transforms

    cv = cv2()
    img = cv.imread(str(image_path))
    if img is None:
        raise RuntimeError("无法读取图像")
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    size = int(settings.tinydetr_pose_imgsz or 224)
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = tf(rgb).unsqueeze(0)
    return rgb, tensor


def estimate_tinydetr_pose(
    image_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    names_3d: list[str],
    pts_3d: np.ndarray,
) -> dict:
    import torch

    net, norm = _load_checkpoint()
    if net is None:
        raise RuntimeError("未配置 TinyDETR-Pose 权重（TINYDETR_POSE_MODEL_PATH）")

    _, tensor = _preprocess(image_path)
    tensor = tensor.to(_device)

    with torch.no_grad():
        out = net(tensor).cpu().numpy().reshape(-1)

    t_mean = np.array(norm.get("t_mean", [0.0, 0.0, 200.0]), dtype=np.float64)
    t_std = np.array(norm.get("t_std", [50.0, 50.0, 50.0]), dtype=np.float64)
    t_norm = out[:3]
    rot6d = out[3:9]
    t_mm = t_norm * t_std + t_mean
    rvec = rot6d_to_rvec(rot6d)
    tvec = t_mm.reshape(3, 1)

    reproj_err = float(norm.get("default_reproj_px", 8.0))
    if settings.tinydetr_pose_eval_reproj:
        try:
            image_pts = contour_corners_2d(image_path)
            reproj_err = reprojection_error_vs_points(pts_3d[:4], image_pts, rvec, tvec, K, dist)
        except Exception:
            pass

    nn_conf = float(out[9]) if len(out) > 9 else max(0.05, min(0.99, 1.0 - reproj_err / 25.0))

    return pack_pose_result(
        "tinydetr_pose",
        rvec,
        tvec,
        reproj_err,
        extra={
            "path": "tinydetr_pose_e2e",
            "model_weights": str(resolve_pkg_path(settings.tinydetr_pose_model_path)),
            "network_confidence": round(nn_conf, 3),
            "translation_norm": [round(float(x), 4) for x in t_norm],
        },
    )
