"""Standalone config: screw_algo/.env + defaults under screw_algo/models & assets."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent


def resolve_pkg_path(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    candidate = PKG_ROOT / raw
    return candidate if candidate.exists() else raw


class Settings(BaseSettings):
    vlm_provider: str = "qwen"

    ark_api_key: str = ""
    ark_model_id: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    qwen_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
    )
    qwen_model: str = "qwen2.5-vl-72b-instruct"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    yolo_model_path: str = "models/screw_yolov8.pt"
    yolo_coarse_task: str = "auto"
    yolo_obb_model_path: str = "models/screw_yolov8_obb.pt"
    yolo_recall_threshold: float = 0.15
    yolo_vlm_trigger_threshold: float = 0.5
    vlm_timeout_seconds: float = 3.0
    vlm_draw_yolo_boxes: bool = True
    vlm_json_schema_validate: bool = True
    vlm_prompt_path: str = "assets/vlm_screw_inspect_prompt.txt"
    variety_yolo_models_path: str = "assets/variety_yolo_models.json"
    variety_pending_class_id: str = "variety_pending"

    pose_enabled: bool = True
    pose_mode: str = "auto"
    pose_contour_fallback: bool = True
    yolo_pose_model_path: str = "models/screw_yolov8_pose.pt"
    yolo_pose_min_kpt_conf: float = 0.25
    tinydetr_pose_model_path: str = "models/screw_tinydetr_pose.pt"
    tinydetr_pose_imgsz: int = 224
    tinydetr_pose_eval_reproj: bool = True

    camera_intrinsics_path: str = "assets/default_camera_intrinsics.json"
    camera_intrinsics_json: str = ""
    screw_model_3d_path: str = "assets/screw_keypoints_3d.json"
    hand_eye_matrix_path: str = "assets/hand_eye_identity.json"
    hand_eye_matrix_json: str = ""

    variety_enabled: bool = True
    variety_model_path: str = "models/screw_variety_mobilenet.pt"
    variety_backbone: str = "mobilenet_v3_small"
    variety_imgsz: int = 224
    variety_min_confidence: float = 0.35
    variety_strategies_path: str = "assets/variety_sort_strategies.json"

    class Config:
        env_file = str(PKG_ROOT / ".env")
        extra = "ignore"


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
settings = Settings()
