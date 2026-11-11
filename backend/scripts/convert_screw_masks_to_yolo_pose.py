"""
从 screw-dataset mask 生成 YOLO-Pose 标签（5 点：头部四角 + 杆尖近似）。
输出 data/yolo-screw-pose/ 供 ultralytics pose 训练（路径 1 权重）。
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "data" / "screw-dataset" / "dataset"
OUT = PROJECT_ROOT / "data" / "yolo-screw-pose"


def _mask_keypoints(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 127)
    if len(xs) < 10:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # 头部四角（mask 外接矩形上半部分四角近似）
    mid_y = (y1 + y2) // 2
    head_y2 = min(y2, mid_y + (y2 - y1) // 4)
    pts = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, head_y2],
            [x1, head_y2],
            [(x1 + x2) / 2, y2],
        ],
        dtype=np.float64,
    )
    return pts


def main():
    samples = []
    for cls_dir in sorted(SRC.iterdir()):
        if not cls_dir.is_dir():
            continue
        img_dir = cls_dir / "images"
        mask_dir = cls_dir / "masks"
        for img_path in sorted(img_dir.glob("*.png")):
            m = mask_dir / f"{img_path.stem}_mask.png"
            if not m.exists():
                m = mask_dir / f"{img_path.stem}.png"
            if m.exists():
                samples.append((img_path, m, cls_dir.name))

    if OUT.exists():
        shutil.rmtree(OUT)
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    n_val = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    splits = {
        "train": samples[: n - n_val - n_test],
        "val": samples[n - n_val - n_test : n - n_test],
        "test": samples[n - n_test :],
    }

    for split, items in splits.items():
        for img_path, mask_path, _cls in items:
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            h, w = img.shape[:2]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            kpts = _mask_keypoints(mask)
            if kpts is None:
                continue

            dst_img = OUT / "images" / split / img_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)

            ys, xs = np.where(mask > 127)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            flat = []
            for x, y in kpts:
                flat.extend([x / w, y / h, 2])  # v=2 visible
            line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(f"{v:.6f}" for v in flat) + "\n"
            lbl = OUT / "labels" / split / f"{img_path.stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text(line, encoding="utf-8")

    yaml = OUT / "screw-pose.yaml"
    yaml.write_text(
        f"path: {OUT.as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "kpt_shape: [5, 3]\n"
        "flip_idx: [0, 1, 2, 3, 4]\n"
        "names:\n  0: screw\n",
        encoding="utf-8",
    )
    print(f"YOLO-Pose 数据集: {OUT}")
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
