"""
ScrewInspect 算法流水线（独立模块，不依赖 FastAPI/前端）。
用法: python -m screw_algo <image> [--no-vlm] [--json out.json]
"""

from screw_algo.pipeline import run_pipeline

__all__ = ["run_pipeline"]
