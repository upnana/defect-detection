"""
将 data/screw-dataset/dataset/{class}/images + masks 转为 YOLO 格式。
默认输出检测框（由 mask 外接矩形）与分割多边形两种标签，训练可任选 screw.yaml 中 task。
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import sys

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.defect_types import DEFECT_CLASS_IDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "data" / "screw-dataset" / "dataset"
DEFAULT_OUT = PROJECT_ROOT / "data" / "yolo-screw"


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return x1, y1, x2, y2


def _mask_polygon(mask: np.ndarray, max_points: int = 80) -> np.ndarray | None:
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None
    epsilon = 0.002 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = approx.reshape(-1, 2)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return pts


def collect_samples(src: Path) -> list[dict]:
    samples = []
    class_names = sorted(DEFECT_CLASS_IDS)
    class_to_id = {n: i for i, n in enumerate(class_names)}

    for class_name in class_names:
        img_dir = src / class_name / "images"
        mask_dir = src / class_name / "masks"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            stem = img_path.stem
            mask_path = mask_dir / f"{stem}_mask.png"
            if not mask_path.exists():
                mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                continue
            samples.append(
                {
                    "image": img_path,
                    "mask": mask_path,
                    "class_id": class_to_id[class_name],
                    "class_name": class_name,
                }
            )
    return samples


def write_labels(sample: dict, label_detect: Path, label_seg: Path) -> bool:
    img = cv2.imread(str(sample["image"]))
    mask = cv2.imread(str(sample["mask"]), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return False
    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    bbox = _mask_bbox(mask)
    if bbox is None:
        bbox = (0.0, 0.0, float(w - 1), float(h - 1))
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cid = sample["class_id"]
    label_detect.parent.mkdir(parents=True, exist_ok=True)
    label_detect.write_text(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

    poly = _mask_polygon(mask)
    if poly is None or len(poly) < 3:
        poly = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float64,
        )
    norm = []
    for x, y in poly:
        norm.append(f"{x / w:.6f}")
        norm.append(f"{y / h:.6f}")
    line = f"{cid} " + " ".join(norm) + "\n"
    label_seg.parent.mkdir(parents=True, exist_ok=True)
    label_seg.write_text(line, encoding="utf-8")
    return True


def split_samples(samples: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_class: dict[str, list] = {}
    for s in samples:
        by_class.setdefault(s["class_name"], []).append(s)
    splits = {"train": [], "val": [], "test": []}
    for _cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(n * 0.1))
        n_test = max(1, int(n * 0.1))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = max(1, n - 2)
            n_val = 1 if n > 1 else 0
            n_test = n - n_train - n_val
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])
    return splits


def build_dataset(src: Path, out: Path, seed: int = 42) -> Path:
    samples = collect_samples(src)
    if not samples:
        raise RuntimeError(f"未找到样本，请检查 {src}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    splits = split_samples(samples, seed=seed)
    class_names = sorted(DEFECT_CLASS_IDS)

    for split_name, split_items in splits.items():
        for s in split_items:
            rel = f"{s['class_name']}_{s['image'].name}"
            img_dst = out / "images" / split_name / rel
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s["image"], img_dst)
            stem = img_dst.stem
            det_lbl = out / "labels_detect" / split_name / f"{stem}.txt"
            seg_lbl = out / "labels_seg" / split_name / f"{stem}.txt"
            write_labels({**s, "image": img_dst}, det_lbl, seg_lbl)

    yaml_path = out / "screw.yaml"
    yaml_path.write_text(
        f"""# screw-dataset -> YOLO (auto-generated)
path: {out.as_posix()}
train: images/train
val: images/val
test: images/test

task: segment
names:
"""
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
        + """

# 检测任务可将 train 标签目录改为 labels_detect，task 改为 detect
labels_seg: labels_seg
labels_detect: labels_detect
""",
        encoding="utf-8",
    )

    print(f"样本总数: {len(samples)}")
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")
    print(f"已写入: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(Path(args.src), Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
