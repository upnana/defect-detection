from __future__ import annotations

from typing import Any

from screw_algo.core.coarse import coarse_screen


def run_coarse(image_path: str, variety: dict[str, Any] | None = None) -> dict[str, Any]:
    return coarse_screen(image_path, variety)
