from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # VLM 精查：ark | deepseek | qwen（千问/DashScope OpenAI 兼容）
    vlm_provider: str = "ark"

    ark_api_key: str = ""
    ark_model_id: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"

    qwen_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
    )
    qwen_model: str = "qwen-vl-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 其他非敏感配置可以保留默认值
    database_url: str = "sqlite:///./screw_inspect.db"
    upload_dir: str = "uploads"
    faiss_data_dir: str = "faiss_data"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # 是否禁用 FAISS，由环境变量 DISABLE_FAISS 控制
    disable_faiss: bool = False

    # YOLOv8 粗筛（可选）
    yolo_model_path: str = ""
    yolo_recall_threshold: float = 0.15
    yolo_vlm_trigger_threshold: float = 0.5
    vlm_timeout_seconds: float = 3.0

    # 6D 位姿估计（路径1 YOLO-Pose+PnP / 路径2 TinyDETR-Pose）
    pose_enabled: bool = True
    pose_mode: str = "auto"  # auto | yolo_pnp | tinydetr_pose | contour_pnp
    pose_contour_fallback: bool = False
    yolo_pose_model_path: str = ""
    yolo_pose_min_kpt_conf: float = 0.25
    tinydetr_pose_model_path: str = ""
    tinydetr_pose_imgsz: int = 224
    tinydetr_pose_eval_reproj: bool = True
    camera_intrinsics_path: str = "app/assets/default_camera_intrinsics.json"
    camera_intrinsics_json: str = ""
    screw_model_3d_path: str = ""
    hand_eye_matrix_path: str = "app/assets/hand_eye_identity.json"
    hand_eye_matrix_json: str = ""

    variety_enabled: bool = True
    variety_model_path: str = "models/screw_variety_mobilenet.pt"
    variety_backbone: str = "mobilenet_v3_small"
    variety_imgsz: int = 224
    variety_min_confidence: float = 0.35
    variety_strategies_path: str = "app/assets/variety_sort_strategies.json"

    class Config:
        env_file = ".env"


settings = Settings()
