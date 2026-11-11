"""离线分拣决策（无 SQLite）。"""

from __future__ import annotations

from typing import Any

from screw_algo.defect_types import DEFECT_ONLY_IDS, ID_TO_ZH

_BLOCK = {"manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"}
_REVIEW_LOW_CONF = 0.55


def decide(ai: dict[str, Any]) -> dict[str, Any]:
    types = ai.get("violation_types") or []
    defect_ids = [t for t in types if t in DEFECT_ONLY_IDS]
    conf = float(ai.get("confidence", 0))
    detected = bool(ai.get("violation_detected"))

    if not defect_ids and not detected:
        return {
            "action": "pass",
            "reason": "判定为良品（good）",
            "defect_ids": [],
        }

    if conf < _REVIEW_LOW_CONF:
        return {
            "action": "review",
            "reason": "置信度偏低，建议人工复检",
            "defect_ids": defect_ids,
        }

    if any(d in _BLOCK for d in defect_ids):
        zh = "、".join(ID_TO_ZH.get(d, d) for d in defect_ids)
        return {
            "action": "block",
            "reason": f"检出缺陷：{zh}",
            "defect_ids": defect_ids,
        }

    return {"action": "review", "reason": "未命中明确规则", "defect_ids": defect_ids}
