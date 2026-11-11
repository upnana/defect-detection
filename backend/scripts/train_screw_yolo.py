"""
微调 YOLOv8-seg 粗筛模型，导出 backend/models/screw_yolov8.pt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "yolo-screw"
DEFAULT_EXPORT = BACKEND_ROOT / "models" / "screw_yolov8.pt"


def _prepare_seg_yaml(data_dir: Path) -> Path:
    """Ultralytics segment 需要 labels 与 images 同级目录结构 labels/train。"""
    seg_root = data_dir / "seg_layout"
    if seg_root.exists():
        shutil.rmtree(seg_root)
    for split in ("train", "val", "test"):
        imgs = data_dir / "images" / split
        lbls = data_dir / "labels_seg" / split
        if not imgs.is_dir():
            continue
        (seg_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (seg_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in imgs.glob("*"):
            shutil.copy2(img, seg_root / "images" / split / img.name)
            lbl = lbls / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, seg_root / "labels" / split / lbl.name)

    names = sorted(
        [
            "good",
            "manipulated_front",
            "scratch_head",
            "scratch_neck",
            "thread_side",
            "thread_top",
        ]
    )
    yaml_path = seg_root / "screw-seg.yaml"
    yaml_path.write_text(
        f"path: {seg_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def train(
    data_dir: Path,
    export_path: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    base_weights: str,
):
    from ultralytics import YOLO

    yaml_path = _prepare_seg_yaml(data_dir)
    print(f"训练配置: {yaml_path}")
    model = YOLO(base_weights)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        project=str(BACKEND_ROOT / "runs" / "screw"),
        name="yolov8n-seg-screw",
        exist_ok=True,
        patience=15,
        save=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        best = Path(results.save_dir) / "weights" / "last.pt"
    if not best.exists():
        raise RuntimeError("训练完成但未找到 best.pt")

    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export_path)
    print(f"已导出粗筛权重: {export_path}")
    return export_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--export", type=str, default=str(DEFAULT_EXPORT))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0", help="0=cuda, cpu=cpu")
    parser.add_argument("--weights", type=str, default="yolov8n-seg.pt")
    args = parser.parse_args()

    device = args.device
    try:
        import torch

        if device != "cpu" and not torch.cuda.is_available():
            print("未检测到 CUDA，改用 CPU 训练（较慢）")
            device = "cpu"
            if args.batch > 4:
                args.batch = 4
    except ImportError:
        device = "cpu"

    train(
        Path(args.data),
        Path(args.export),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        base_weights=args.weights,
    )


if __name__ == "__main__":
    main()
