from __future__ import annotations



from typing import Any



from screw_algo.core.pose_pipeline import estimate_pose_6d





def run_pose_6d(image_path: str) -> dict[str, Any]:

    return estimate_pose_6d(image_path)

