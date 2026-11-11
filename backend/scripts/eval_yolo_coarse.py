"""在 val 集上评估 YOLO 粗筛（类别准确率 + 粗筛逻辑统计）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.coarse_screen_service import coarse_screen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAL = PROJECT_ROOT / "data" / "yolo-screw" / "images" / "val"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=str, default=str(DEFAULT_VAL))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not settings.yolo_model_path:
        print("请先在 .env 设置 YOLO_MODEL_PATH")
        sys.exit(1)

    val_dir = Path(args.val_dir)
    images = sorted(val_dir.glob("*.png"))
    if args.limit > 0:
        images = images[: args.limit]

    # 文件名前缀为 class_name_
    correct = 0
    total = 0
    skip_vlm = 0
    trigger_vlm = 0

    for img in images:
        name = img.stem
        gt = name.split("_")[0] if "_" in name else None
        if not gt:
            continue
        screen = coarse_screen(str(img.resolve()))
        total += 1
        pred = screen.get("primary_class")
        if pred == gt:
            correct += 1
        if screen.get("skip_vlm"):
            skip_vlm += 1
        if screen.get("trigger_vlm"):
            trigger_vlm += 1

    if total == 0:
        print("无验证样本")
        return
    print(f"YOLO 权重: {settings.yolo_model_path}")
    print(f"Val 样本: {total}")
    print(f"主类准确率(与文件名类别一致): {correct/total*100:.1f}%")
    print(f"skip_vlm(良品快速放行): {skip_vlm} ({skip_vlm/total*100:.1f}%)")
    print(f"trigger_vlm: {trigger_vlm} ({trigger_vlm/total*100:.1f}%)")


if __name__ == "__main__":
    main()
