from sqlalchemy.orm import Session

from app.models import Rule
from app.services.faiss_service import faiss_service
from app.defect_types import DEFECT_ONLY_IDS, ID_TO_ZH

def _rules_for_defect_types(db: Session, defect_ids: list[str]) -> list[dict]:
    if not db or not defect_ids:
        return []
    rules = (
        db.query(Rule)
        .filter(Rule.is_active.is_(True), Rule.violation_type.in_(defect_ids))
        .order_by(Rule.priority.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "name": r.name,
            "action": r.action,
            "similarity": 1.0,
            "violation_type": r.violation_type,
        }
        for r in rules
    ]


def judge(ai_result: dict, db: Session | None = None) -> dict:
    confidence = float(ai_result.get("confidence", 0.0))
    violation_types = ai_result.get("violation_types", []) or []
    defect_ids = [t for t in violation_types if t in DEFECT_ONLY_IDS]
    risk_desc = ai_result.get("risk_description", "")

    direct_rules = _rules_for_defect_types(db, defect_ids) if defect_ids else []

    similar_rules = []
    if risk_desc:
        query = f"{' '.join(violation_types)} {risk_desc}"
        similar_rules = faiss_service.search_similar_rules(query, top_k=5)

    faiss_matched = [
        r for r in similar_rules if r.get("similarity", 0) > 0.3
    ]

    merged: dict[int, dict] = {}
    for r in direct_rules + faiss_matched:
        rid = r.get("id")
        if rid is None:
            continue
        if rid not in merged or r.get("similarity", 0) > merged[rid].get("similarity", 0):
            merged[rid] = r

    matched_list = sorted(merged.values(), key=lambda x: -x.get("similarity", 0))

    if not defect_ids and not ai_result.get("violation_detected"):
        return {
            "action": "pass",
            "reason": "判定为良品（good），分拣放行",
            "matched_rules": [
                {"id": r["id"], "name": r["name"], "action": r["action"], "similarity": round(r.get("similarity", 1), 3)}
                for r in matched_list
                if r.get("action") == "pass"
            ][:3],
        }

    block_rules = [r for r in matched_list if r.get("action") == "block"]
    review_rules = [r for r in matched_list if r.get("action") == "review"]

    if block_rules and confidence >= 0.55:
        action = "block"
        reason = f"命中拦截策略：{', '.join(ID_TO_ZH.get(d, d) for d in defect_ids)}"
    elif review_rules or (defect_ids and confidence < 0.55):
        action = "review"
        reason = "置信度中等或需人工复检"
    elif defect_ids and confidence >= 0.55:
        action = "block"
        reason = f"检出缺陷且无放行规则：{', '.join(ID_TO_ZH.get(d, d) for d in defect_ids)}"
    else:
        action = "pass"
        reason = "未命中明确拦截规则"

    matched = [
        {
            "id": r["id"],
            "name": r["name"],
            "action": r["action"],
            "similarity": round(float(r.get("similarity", 0)), 3),
        }
        for r in matched_list
    ]

    return {
        "action": action,
        "reason": reason,
        "matched_rules": matched,
    }
