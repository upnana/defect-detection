"""训练路径 1 权重：YOLOv8-Pose → backend/models/screw_yolov8_pose.pt"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT / "data" / "yolo-screw-pose"
DEFAULT_EXPORT = BACKEND / "models" / "screw_yolov8_pose.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA / "screw-pose.yaml"))
    parser.add_argument("--export", default=str(DEFAULT_EXPORT))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", default="yolov8n-pose.pt")
    args = parser.parse_args()

    device = args.device
    try:
        import torch

        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
            args.batch = min(args.batch, 4)
    except ImportError:
        device = "cpu"

    from ultralytics import YOLO

    w = Path(args.weights)
    if not w.is_absolute() and (BACKEND / w).exists():
        args.weights = str(BACKEND / w)
    elif not w.exists() and (PROJECT / args.weights).exists():
        args.weights = str(PROJECT / args.weights)

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        project=str(BACKEND / "runs" / "pose"),
        name="yolov8n-pose-screw",
        exist_ok=True,
        patience=20,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    DEFAULT_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.export)
    print(f"路径1 权重已导出: {args.export}")


if __name__ == "__main__":
    main()
