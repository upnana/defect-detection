from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.defect_types import DATASET_ID, DATASET_URL, DEFECT_CLASSES

router = APIRouter()

ASSETS = Path(__file__).resolve().parent.parent / "assets"


@router.get("/defect-classes")
def get_defect_classes():
    return {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "classes": DEFECT_CLASSES,
    }


@router.get("/pose-config")
def get_pose_config():
    return {
        "pose_enabled": settings.pose_enabled,
        "pose_mode": settings.pose_mode,
        "yolo_pose_model_configured": bool(settings.yolo_pose_model_path),
        "camera_intrinsics_path": settings.camera_intrinsics_path,
        "hand_eye_matrix_path": settings.hand_eye_matrix_path,
        "screw_model_3d_path": settings.screw_model_3d_path or str(ASSETS / "screw_keypoints_3d.json"),
        "pipeline": [
            "Path A: YOLOv8-Pose 2D 关键点 + 螺丝 3D 模型 + solvePnP",
            "Path B: 轮廓 minAreaRect 回退（auto 模式下无 Pose 权重时使用）",
            "手眼标定: T_robot = T_hand_eye × T_cam_obj",
        ],
    }
