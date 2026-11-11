"""对比 YOLO 粗筛 vs 粗筛+DeepSeek VLM：延迟与 val 主类准确率（小样本）。"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.coarse_screen_service import coarse_screen
from app.services.doubao_service import analyze_screw_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAL = PROJECT_ROOT / "data" / "yolo-screw" / "images" / "val"


def _ground_truth(stem: str) -> str | None:
    if "_" not in stem:
        return None
    return stem.split("_", 1)[0]


def _vlm_primary(result: dict) -> str | None:
    types = result.get("violation_types") or result.get("defect_ids") or []
    if isinstance(types, list) and types:
        return str(types[0])
    if result.get("violation_detected") is False:
        return "good"
    return None


async def _run_vlm(path: Path, screen: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    out = await analyze_screw_image(str(path.resolve()), coarse_screen=screen)
    return out, time.perf_counter() - t0


async def main_async(limit: int, val_dir: Path):
    if not settings.yolo_model_path:
        print("请设置 YOLO_MODEL_PATH")
        sys.exit(1)
    if not settings.deepseek_api_key and (settings.vlm_provider or "").lower() == "deepseek":
        print("DeepSeek 未配置 DEEPSEEK_API_KEY，跳过 VLM 对比")
        sys.exit(1)

    images = sorted(val_dir.glob("*.png"))
    if limit > 0:
        images = images[:limit]

    yolo_times: list[float] = []
    vlm_times: list[float] = []
    yolo_ok = vlm_ok = total = 0

    for img in images:
        gt = _ground_truth(img.stem)
        if not gt:
            continue
        t0 = time.perf_counter()
        screen = coarse_screen(str(img.resolve()))
        yolo_times.append(time.perf_counter() - t0)
        pred_y = screen.get("primary_class")
        if pred_y == gt:
            yolo_ok += 1

        if screen.get("skip_vlm"):
            vlm_times.append(0.0)
            if pred_y == gt:
                vlm_ok += 1
        else:
            result, dt = await _run_vlm(img, screen)
            vlm_times.append(dt)
            pred_v = _vlm_primary(result)
            if pred_v == gt:
                vlm_ok += 1
        total += 1

    if total == 0:
        print("无验证样本")
        return

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) * 1000 if xs else 0.0

    print(f"样本数: {total}")
    print(f"YOLO 主类准确率: {yolo_ok/total*100:.1f}%")
    print(f"粗筛+VLM 主类准确率: {vlm_ok/total*100:.1f}%")
    print(f"YOLO 平均延迟: {_avg(yolo_times):.1f} ms")
    vlm_only = [t for t in vlm_times if t > 0]
    print(f"VLM 调用平均延迟(未 skip): {_avg(vlm_only):.0f} ms (n={len(vlm_only)})")
    print(f"端到端平均(粗筛+按需VLM): {_avg(yolo_times) + _avg(vlm_only):.0f} ms (近似)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=str, default=str(DEFAULT_VAL))
    parser.add_argument("--limit", type=int, default=14, help="默认跑满 val 集")
    args = parser.parse_args()
    asyncio.run(main_async(args.limit, Path(args.val_dir)))


if __name__ == "__main__":
    main()
