"""
训练路径 2：TinyDETR-Pose（弱监督：轮廓 PnP 伪标签）→ backend/models/screw_tinydetr_pose.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "data" / "screw-dataset" / "dataset"
DEFAULT_EXPORT = BACKEND / "models" / "screw_tinydetr_pose.pt"


def _pseudo_labels(limit: int = 0):
    from app.services.pose_common import load_camera_matrix, load_screw_model_3d, solve_pnp
    from app.services.pose_path_contour import contour_corners_2d

    names, pts_3d, _ = load_screw_model_3d()
    rows = []
    for cls_dir in sorted(SRC.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted((cls_dir / "images").glob("*.png")):
            import cv2

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            K, dist = load_camera_matrix(img.shape)
            try:
                image_pts = contour_corners_2d(img_path)
                rvec, tvec, _ = solve_pnp(pts_3d[:4], image_pts, K, dist)
            except Exception:
                continue
            from app.services.pose_common import cv2 as cv2mod

            cv = cv2mod()
            R, _ = cv.Rodrigues(rvec)
            rot6d = np.concatenate([R[:, 0], R[:, 1]])
            t = tvec.reshape(3)
            rows.append({"image": str(img_path), "t_mm": t.tolist(), "rot6d": rot6d.tolist()})
            if limit and len(rows) >= limit:
                return rows
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--export", default=str(DEFAULT_EXPORT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from app.ml.tinydetr_pose_net import TinyDETRPoseNet

    rows = _pseudo_labels(args.limit)
    if len(rows) < 8:
        raise SystemExit("伪标签样本不足，请先确认 data/screw-dataset 存在")

    t_all = np.array([r["t_mm"] for r in rows], dtype=np.float64)
    t_mean = t_all.mean(axis=0)
    t_std = np.maximum(t_all.std(axis=0), 1e-3)

    class PoseDS(Dataset):
        def __init__(self, items):
            self.items = items
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            import cv2

            row = self.items[i]
            bgr = cv2.imread(row["image"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = (np.array(row["t_mm"]) - t_mean) / t_std
            rot6d = np.array(row["rot6d"], dtype=np.float32)
            y = np.concatenate([t, rot6d]).astype(np.float32)
            return self.tf(rgb), torch.from_numpy(y)

    n_val = max(1, len(rows) // 10)
    train_rows = rows[n_val:]
    val_rows = rows[:n_val]
    train_loader = DataLoader(PoseDS(train_rows), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(PoseDS(val_rows), batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = TinyDETRPoseNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(args.epochs):
        net.train()
        tr = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = net(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tr += float(loss.item())
        net.eval()
        va = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                va += float(loss_fn(net(x), y).item())
        print(f"epoch {epoch+1}/{args.epochs} train={tr/len(train_loader):.4f} val={va/max(1,len(val_loader)):.4f}")

    export = Path(args.export)
    export.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "out_dim": 9,
            "norm": {
                "t_mean": t_mean.tolist(),
                "t_std": t_std.tolist(),
                "default_reproj_px": 8.0,
            },
        },
        export,
    )
    print(f"路径2 权重已导出: {export}")


if __name__ == "__main__":
    main()
