"""
算法流水线（离线、无 Web、无 backend 依赖）:

  输入图像
    → [0] MobileNet / EfficientNet 品种分类 → 品类标签、分拣策略
    → [1] YOLOv8-OBB/检测 粗筛（按品种选权重，可选跳过 VLM）
    → [2] VLM 精查（可 --no-vlm 关闭）
    → [3] 规则分拣 pass | block | review
    → [4] 6D 位姿（路径1 YOLO-Pose+PnP → 路径2 TinyDETR → 可选轮廓回退）
    → 汇总 JSON
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from screw_algo.config import ROOT, AlgoConfig
from screw_algo.core.coarse import coarse_result_to_ai_result
from screw_algo.stages import coarse, pose, rules, variety, vlm


def _resolve_image(image_path: str | Path) -> str:
    p = Path(image_path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return str(p)


@dataclass
class PipelineResult:
    image: str
    config_snapshot: dict[str, Any]
    timing_ms: dict[str, float] = field(default_factory=dict)
    variety: dict[str, Any] = field(default_factory=dict)
    coarse: dict[str, Any] = field(default_factory=dict)
    inspection: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    pose: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "config": self.config_snapshot,
            "timing_ms": self.timing_ms,
            "variety_classification": self.variety,
            "coarse_screen": self.coarse,
            "inspection": self.inspection,
            "decision": self.decision,
            "sorting_strategy": self.variety.get("sorting_strategy"),
            "pose_6d": self.pose.get("pose_6d"),
            "pose": self.pose,
        }


def run_pipeline(
    image_path: str | Path,
    *,
    use_vlm: bool = True,
    vlm_timeout_s: float | None = None,
) -> PipelineResult:
    image_path = _resolve_image(image_path)

    cfg = AlgoConfig.load()
    snap = {
        "package_root": str(cfg.package_root),
        "variety_classifier": str(cfg.variety_model) if cfg.variety_model else None,
        "yolo_coarse": str(cfg.yolo_coarse) if cfg.yolo_coarse else None,
        "yolo_pose": str(cfg.yolo_pose) if cfg.yolo_pose else None,
        "tinydetr_pose": str(cfg.tinydetr_pose) if cfg.tinydetr_pose else None,
        "vlm_provider": cfg.vlm_provider,
        "pose_mode": cfg.pose_mode,
    }
    result = PipelineResult(image=image_path, config_snapshot=snap)
    timeout = vlm_timeout_s if vlm_timeout_s is not None else cfg.vlm_timeout_s

    t_var = time.perf_counter()
    result.variety = variety.run_variety(image_path)
    result.timing_ms["variety"] = (time.perf_counter() - t_var) * 1000

    t0 = time.perf_counter()
    result.coarse = coarse.run_coarse(image_path, result.variety)
    result.timing_ms["coarse"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    if use_vlm:
        result.inspection = vlm.run_vlm(image_path, result.coarse, timeout, result.variety)
    else:
        result.inspection = coarse_result_to_ai_result(result.coarse)
        result.inspection["stage"] = "coarse_only_no_vlm"
    result.inspection["variety_hint"] = {
        "class_id": result.variety.get("class_id"),
        "confidence": result.variety.get("confidence"),
        "sorting_strategy": result.variety.get("sorting_strategy"),
    }
    result.timing_ms["inspection"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    result.decision = rules.decide_inspection(result.inspection)
    result.decision["variety_class_id"] = result.variety.get("class_id")
    result.timing_ms["decision"] = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    result.pose = pose.run_pose_6d(image_path)
    result.timing_ms["pose"] = (time.perf_counter() - t3) * 1000
    result.timing_ms["total"] = sum(
        v for k, v in result.timing_ms.items() if k != "total"
    )

    return result
