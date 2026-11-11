"""
训练 YOLOv8-OBB 粗筛权重 → screw_algo/models/screw_yolov8_obb.pt

  python screw_algo/scripts/convert_screw_masks_to_yolo_obb.py
  python screw_algo/scripts/train_screw_yolo_obb.py --epochs 40
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PKG = _ROOT / "screw_algo"
DEFAULT_DATA = _ROOT / "data" / "yolo-screw-obb" / "screw-obb.yaml"
DEFAULT_EXPORT = PKG / "models" / "screw_yolov8_obb.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--export", default=str(DEFAULT_EXPORT))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--weights", default="yolov8n-obb.pt")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=8,
        workers=0,
        project=str(PKG / "runs" / "obb"),
        name="yolov8n-obb-screw",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    DEFAULT_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.export)
    print(f"OBB 权重已导出: {args.export}")


if __name__ == "__main__":
    main()
