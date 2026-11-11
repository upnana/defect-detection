"""CLI: python -m screw_algo <image> [--no-vlm] [--json path] [--save-vlm-preview] [--save-pose-preview]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from screw_algo.core.pose_visual import draw_pose_6d_overlay
from screw_algo.core.vlm_visual import draw_coarse_detections
from screw_algo.pipeline import run_pipeline
from screw_algo.settings import PKG_ROOT


def _write_vlm_preview(image: str, coarse: dict, variety: dict, dest: Path) -> None:
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(draw_coarse_detections(Path(image), coarse, variety))
    print(f"画框预览已保存: {dest}", file=sys.stderr)


def _write_pose_preview(image: str, pose: dict, dest: Path) -> None:
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(draw_pose_6d_overlay(Path(image), pose))
    print(f"6D 叠图已保存: {dest}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ScrewInspect 离线算法流水线")
    parser.add_argument("image", type=str, help="螺丝图片路径")
    parser.add_argument("--no-vlm", action="store_true", help="仅 YOLO 粗筛，不调 VLM")
    parser.add_argument("--json", type=str, default="", help="结果写入 JSON 文件")
    parser.add_argument("--timeout", type=float, default=None, help="VLM 超时秒数")
    parser.add_argument(
        "--save-vlm-preview",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="保存与 VLM 相同的画框图（默认 screw_algo/runs/vlm_preview/<图名>.jpg）",
    )
    parser.add_argument(
        "--dump-vlm-prompt",
        action="store_true",
        help="仅输出完整 VLM Prompt（品种+粗筛附录）到 stderr，不调 API",
    )
    parser.add_argument(
        "--save-pose-preview",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="保存 6D 位姿叠图（绿框+RGB轴，默认 screw_algo/runs/pose_preview/<图名>.jpg）",
    )
    args = parser.parse_args(argv)

    if args.dump_vlm_prompt:
        from screw_algo.core.coarse import coarse_screen
        from screw_algo.core.variety import classify_variety
        from screw_algo.core.vlm_prompt import build_vlm_prompt
        from screw_algo.pipeline import _resolve_image

        image_path = _resolve_image(str(args.image))
        variety = classify_variety(image_path)
        coarse = coarse_screen(image_path, variety)
        print(build_vlm_prompt(coarse, variety), file=sys.stderr)
        return 0

    try:
        out = run_pipeline(
            args.image,
            use_vlm=not args.no_vlm,
            vlm_timeout_s=args.timeout,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    payload = out.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")
        print(f"\n已写入: {args.json}", file=sys.stderr)

    if args.save_vlm_preview is not None:
        detections = out.coarse.get("detections") or []
        if not detections:
            print("画框预览: 粗筛无检测框，未生成图片", file=sys.stderr)
        else:
            stem = Path(out.image).stem
            dest = (
                Path(args.save_vlm_preview)
                if args.save_vlm_preview
                else PKG_ROOT / "runs" / "vlm_preview" / f"{stem}.jpg"
            )
            _write_vlm_preview(out.image, out.coarse, out.variety, dest)

    if args.save_pose_preview is not None:
        if not out.pose.get("success"):
            print(f"6D 叠图: 位姿失败，未生成图片 — {out.pose.get('message', '')}", file=sys.stderr)
        else:
            stem = Path(out.image).stem
            dest = (
                Path(args.save_pose_preview)
                if args.save_pose_preview
                else PKG_ROOT / "runs" / "pose_preview" / f"{stem}.jpg"
            )
            try:
                _write_pose_preview(out.image, out.pose, dest)
            except RuntimeError as e:
                print(f"6D 叠图: {e}", file=sys.stderr)

    d = out.decision.get("action", "?")
    pose_ok = out.pose.get("success", False)
    var = out.variety.get("class_id") or "-"
    print(
        f"\n--- 摘要: 品种={var} | 分拣={d} | 6D={'OK' if pose_ok else 'FAIL'} | "
        f"总耗时 {out.timing_ms.get('total', 0):.0f} ms ---",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
