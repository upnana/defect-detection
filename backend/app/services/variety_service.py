"""品种识别：MobileNet / EfficientNet + 分拣策略映射。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.defect_types import DEFECT_CLASSES, ID_TO_ZH
from app.ml.variety_net import build_variety_classifier
from app.services.pose_common import resolve_backend_path

APP_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = APP_DIR / "assets"

_model = None
_meta: dict[str, Any] | None = None
_device = "cpu"
_strategies: dict | None = None


def _resolve_image_path(image_url: str) -> Path | None:
    if not image_url:
        return None
    if image_url.startswith("/uploads/"):
        path = Path(settings.upload_dir) / image_url.removeprefix("/uploads/")
        return path if path.exists() else None
    p = Path(image_url)
    return p if p.exists() else None


def _load_strategies() -> dict:
    global _strategies
    if _strategies is not None:
        return _strategies
    path = (
        resolve_backend_path(settings.variety_strategies_path)
        if settings.variety_strategies_path
        else ASSETS_DIR / "variety_sort_strategies.json"
    )
    _strategies = json.loads(path.read_text(encoding="utf-8"))
    return _strategies


def sorting_strategy_for(class_id: str) -> dict[str, Any]:
    data = _load_strategies()
    base = dict(data.get("default") or {})
    override = (data.get("by_class_id") or {}).get(class_id)
    if override:
        base.update(override)
    base["class_id"] = class_id
    return base


def is_configured() -> bool:
    if not settings.variety_enabled:
        return False
    if not settings.variety_model_path:
        return False
    return resolve_backend_path(settings.variety_model_path).exists()


def _load_model():
    global _model, _meta, _device
    if _model is not None:
        return _model, _meta

    ckpt_path = resolve_backend_path(settings.variety_model_path)
    if not ckpt_path.exists():
        return None, None

    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    backbone = ckpt.get("backbone") or settings.variety_backbone or "mobilenet_v3_small"
    class_names: list[str] = list(ckpt.get("class_names") or [])
    if not class_names:
        class_names = [c["id"] for c in DEFECT_CLASSES]

    net = build_variety_classifier(backbone, len(class_names))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.to(_device)

    _meta = {
        "backbone": backbone,
        "class_names": class_names,
        "imgsz": int(ckpt.get("imgsz") or settings.variety_imgsz or 224),
        "weights": str(ckpt_path),
    }
    _model = net
    return _model, _meta


def _preprocess(image_path: Path, imgsz: int) -> "torch.Tensor":
    import torch
    from torchvision import transforms

    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("无法读取图像")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tf(rgb).unsqueeze(0)


def classify_variety(image_url: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "enabled": False,
        "class_id": None,
        "label_zh": None,
        "confidence": 0.0,
        "top_k": [],
        "sorting_strategy": sorting_strategy_for("good"),
        "message": "品种分类未启用或未配置权重",
    }

    if not settings.variety_enabled:
        base["message"] = "品种分类已在配置中关闭（VARIETY_ENABLED=0）"
        return base

    image_path = _resolve_image_path(image_url)
    if image_path is None:
        base["message"] = "无法解析图片路径"
        return base

    model, meta = _load_model()
    if model is None or meta is None:
        base["message"] = "未找到品种分类权重（VARIETY_MODEL_PATH）"
        return base

    import torch

    tensor = _preprocess(image_path, meta["imgsz"]).to(_device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)

    class_names: list[str] = meta["class_names"]
    order = np.argsort(-probs)
    top_k = []
    for idx in order[: min(3, len(class_names))]:
        cid = class_names[int(idx)]
        top_k.append(
            {
                "class_id": cid,
                "label_zh": ID_TO_ZH.get(cid, cid),
                "confidence": round(float(probs[idx]), 4),
            }
        )

    best_idx = int(order[0])
    best_id = class_names[best_idx]
    best_conf = float(probs[best_idx])
    min_conf = float(settings.variety_min_confidence or 0.35)

    if best_conf < min_conf:
        return {
            "enabled": True,
            "backbone": meta["backbone"],
            "class_id": None,
            "label_zh": None,
            "confidence": round(best_conf, 4),
            "top_k": top_k,
            "sorting_strategy": _load_strategies().get("default"),
            "low_confidence": True,
            "message": f"品种置信度低于阈值 {min_conf}",
            "model_weights": meta["weights"],
        }

    return {
        "enabled": True,
        "backbone": meta["backbone"],
        "class_id": best_id,
        "label_zh": ID_TO_ZH.get(best_id, best_id),
        "confidence": round(best_conf, 4),
        "top_k": top_k,
        "sorting_strategy": sorting_strategy_for(best_id),
        "message": "品种分类完成",
        "model_weights": meta["weights"],
    }
