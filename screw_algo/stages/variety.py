from __future__ import annotations

from typing import Any

from screw_algo.core.variety import classify_variety


def run_variety(image_path: str) -> dict[str, Any]:
    return classify_variety(image_path)
