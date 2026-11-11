from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from screw_algo.settings import PROJECT_ROOT, PKG_ROOT, resolve_pkg_path, settings


@dataclass
class AlgoConfig:
    package_root: Path
    variety_model: Path | None
    yolo_coarse: Path | None
    yolo_pose: Path | None
    tinydetr_pose: Path | None
    vlm_provider: str
    pose_mode: str
    vlm_timeout_s: float

    @classmethod
    def load(cls) -> "AlgoConfig":
        def _p(raw: str) -> Path | None:
            if not raw:
                return None
            p = resolve_pkg_path(raw)
            return p if p.exists() else None

        return cls(
            package_root=PKG_ROOT,
            variety_model=_p(settings.variety_model_path) if settings.variety_enabled else None,
            yolo_coarse=_p(settings.yolo_model_path),
            yolo_pose=_p(settings.yolo_pose_model_path),
            tinydetr_pose=_p(settings.tinydetr_pose_model_path),
            vlm_provider=settings.vlm_provider or "deepseek",
            pose_mode=settings.pose_mode or "auto",
            vlm_timeout_s=float(settings.vlm_timeout_seconds or 30),
        )


ROOT = PROJECT_ROOT
