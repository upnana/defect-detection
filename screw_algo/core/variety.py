"""品种识别：MobileNet / EfficientNet 图像分类 + 分拣策略映射。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from screw_algo.defect_types import DEFECT_CLASSES, ID_TO_ZH
from screw_algo.ml.variety_net import build_variety_classifier
from screw_algo.settings import PKG_ROOT, resolve_pkg_path, settings

_model = None
_meta: dict[str, Any] | None = None
_device = "cpu"
_strategies: dict | None = None


def _load_strategies() -> dict:
    global _strategies
    if _strategies is not None:
        return _strategies
    path = resolve_pkg_path(settings.variety_strategies_path)
    if not path.exists():
        path = PKG_ROOT / "assets" / "variety_sort_strategies.json"
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
    return resolve_pkg_path(settings.variety_model_path).exists()


def _load_model():
    global _model, _meta, _device
    if _model is not None:
        return _model, _meta

    ckpt_path = resolve_pkg_path(settings.variety_model_path)
    if not ckpt_path.exists():
        return None, None

    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    backbone = ckpt.get("backbone") or settings.variety_backbone or BACKBONE_MOBILENET
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


def classify_variety(image_path: str | Path) -> dict[str, Any]:
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

    model, meta = _load_model()
    if model is None or meta is None:
        base["message"] = "未找到品种分类权重（VARIETY_MODEL_PATH）"
        return base

    import torch

    path = Path(image_path)
    if not path.is_file():
        base["message"] = "图像路径无效"
        return base

    tensor = _preprocess(path, meta["imgsz"]).to(_device)
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

    pending_id = settings.variety_pending_class_id or "variety_pending"
    if best_conf < min_conf:
        return {
            "enabled": True,
            "backbone": meta["backbone"],
            "class_id": pending_id,
            "label_zh": "品种待确认",
            "confidence": round(best_conf, 4),
            "top_k": top_k,
            "sorting_strategy": sorting_strategy_for(pending_id),
            "low_confidence": True,
            "general_detection_head": True,
            "message": f"品种置信度低于阈值 {min_conf}，切换通用检测头",
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
