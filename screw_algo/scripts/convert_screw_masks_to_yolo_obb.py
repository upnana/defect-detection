"""
从 screw-dataset mask 生成 YOLO-OBB 标签（minAreaRect 四角，归一化 xyxyxyxy）。
输出 data/yolo-screw-obb/ 供 train_screw_yolo_obb.py 使用。
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SRC = _ROOT / "data" / "screw-dataset" / "dataset"
OUT = _ROOT / "data" / "yolo-screw-obb"

CLASS_NAMES = [
    "good",
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]


def _obb_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 127)
    if len(xs) < 10:
        return None
    cnt = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box.astype(np.float64)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    samples: list[tuple[Path, int]] = []
    for ci, name in enumerate(CLASS_NAMES):
        d = SRC / name
        if not d.is_dir():
            continue
        for img in sorted((d / "images").glob("*.png")):
            samples.append((img, ci))

    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.15))
    splits = {"train": samples[n_val:], "val": samples[:n_val]}

    for split, items in splits.items():
        for img_path, cls_id in items:
            mask_path = img_path.parent.parent / "masks" / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                mask_path = SRC / CLASS_NAMES[cls_id] / "masks" / f"{img_path.stem}_mask.png"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                obb = _obb_from_mask(mask) if mask is not None else None
            else:
                obb = None
            if obb is None:
                continue
            norm = []
            for x, y in obb:
                norm.extend([x / w, y / h])
            line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm)
            out_img = OUT / "images" / split / img_path.name
            out_lbl = OUT / "labels" / split / f"{img_path.stem}.txt"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_img)
            out_lbl.write_text(line + "\n", encoding="utf-8")

    yaml_path = OUT / "screw-obb.yaml"
    yaml_path.write_text(
        f"path: {OUT.as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)) + "\n",
        encoding="utf-8",
    )
    print(f"OBB 数据集已写入 {OUT}，配置 {yaml_path}")


if __name__ == "__main__":
    main()
