"""
训练品种/品类分类器（MobileNetV3-Small 或 EfficientNet-B0）。

数据目录结构（与 screw-dataset 一致）：
  <data_root>/<class_id>/images/*.png

示例：
  python screw_algo/scripts/train_variety_classifier.py ^
    --data data/screw-dataset/dataset ^
    --out screw_algo/models/screw_variety_mobilenet.pt ^
    --backbone mobilenet_v3_small --epochs 25
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np


def _collect_samples(data_root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    samples: list[tuple[Path, int]] = []
    for ci, d in enumerate(class_dirs):
        img_dir = d / "images"
        if not img_dir.is_dir():
            img_dir = d
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for p in img_dir.glob(ext):
                samples.append((p, ci))
    return samples, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train screw variety classifier")
    parser.add_argument("--data", type=str, required=True, help="Dataset root with class subfolders")
    parser.add_argument("--out", type=str, default="screw_algo/models/screw_variety_mobilenet.pt")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from screw_algo.ml.variety_net import build_variety_classifier

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data)
    samples, class_names = _collect_samples(data_root)
    if len(samples) < 10:
        raise SystemExit(f"样本过少: {len(samples)}，请检查 --data 路径")

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    tf_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    import cv2

    class ScrewSet(Dataset):
        def __init__(self, items: list[tuple[Path, int]], transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"无法读取 {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return self.transform(rgb), label

    train_loader = DataLoader(
        ScrewSet(train_samples, tf_train),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ScrewSet(val_samples, tf_val),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_variety_classifier(args.backbone, len(class_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(yb)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += len(yb)
        acc = correct / max(1, total)
        print(f"epoch {epoch}/{args.epochs}  loss={train_loss / max(1, len(train_samples)):.4f}  val_acc={acc:.3f}")

        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "backbone": args.backbone,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                    "imgsz": args.imgsz,
                    "val_acc": acc,
                },
                out_path,
            )

    print(f"已保存: {out_path}  (best val_acc={best_acc:.3f}, classes={class_names})")


if __name__ == "__main__":
    main()
