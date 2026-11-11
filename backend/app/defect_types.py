"""螺丝缺陷类别 — 对齐 Hugging Face shrish23/screw-dataset"""

DATASET_ID = "shrish23/screw-dataset"
DATASET_URL = "https://huggingface.co/datasets/shrish23/screw-dataset"

DEFECT_CLASSES: list[dict] = [
    {
        "id": "good",
        "label_zh": "良好",
        "description": "外观完整，无篡改、划痕或螺纹加工缺陷",
        "is_defect": False,
    },
    {
        "id": "manipulated_front",
        "label_zh": "正面被篡改",
        "description": "螺丝头部正面纹理异常、装配痕迹或人为改动",
        "is_defect": True,
    },
    {
        "id": "scratch_head",
        "label_zh": "头部划痕",
        "description": "螺丝头部表面可见线性或点状划伤",
        "is_defect": True,
    },
    {
        "id": "scratch_neck",
        "label_zh": "颈部划痕",
        "description": "螺杆颈部柱面划痕或磕碰",
        "is_defect": True,
    },
    {
        "id": "thread_side",
        "label_zh": "侧面螺纹缺陷",
        "description": "侧面牙型不完整、乱纹或缺损",
        "is_defect": True,
    },
    {
        "id": "thread_top",
        "label_zh": "顶部螺纹缺陷",
        "description": "顶部螺纹区域缺损或加工异常",
        "is_defect": True,
    },
]

DEFECT_CLASS_IDS = {c["id"] for c in DEFECT_CLASSES}
DEFECT_ONLY_IDS = {c["id"] for c in DEFECT_CLASSES if c["is_defect"]}

ID_TO_ZH = {c["id"]: c["label_zh"] for c in DEFECT_CLASSES}


def normalize_defect_ids(raw: list) -> list[str]:
    out: list[str] = []
    for item in raw or []:
        key = str(item).strip()
        if key in DEFECT_CLASS_IDS and key not in out:
            out.append(key)
    return out
