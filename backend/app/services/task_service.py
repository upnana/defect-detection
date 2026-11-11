import asyncio
import json
from datetime import datetime

from app.database import SessionLocal
from app.models import Task
from app.services import doubao_service, rule_engine
from app.services.coarse_screen_service import coarse_result_to_ai_result, coarse_screen
from app.services.pose_service import estimate_pose_6d
from app.services.variety_service import classify_variety
from app.config import settings
from app.defect_types import ID_TO_ZH

TERMINAL_STATUS = {"pass", "block", "review", "error"}


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def append_log(db, task: Task, stage: str, message: str, payload=None):
    logs = []
    if task.process_logs:
        try:
            logs = json.loads(task.process_logs)
        except Exception:
            logs = []

    entry = {
        "time": _now_str(),
        "stage": stage,
        "message": message,
    }
    if payload is not None:
        entry["payload"] = payload

    logs.append(entry)
    task.process_logs = json.dumps(logs, ensure_ascii=False)
    task.updated_at = datetime.now()
    db.commit()


def _apply_ai_result(task: Task, ai_result: dict):
    task.ai_result = json.dumps(ai_result, ensure_ascii=False)
    task.confidence = ai_result.get("confidence", 0.0)
    types = ai_result.get("violation_types", [])
    task.violation_types = ",".join(types)
    desc = ai_result.get("risk_description", "")
    if types and types != ["good"]:
        zh = "、".join(ID_TO_ZH.get(t, t) for t in types)
        task.risk_description = f"{desc}（类别：{zh}）" if desc else f"检出缺陷：{zh}"
    else:
        task.risk_description = desc or "判定为良品"


async def process_task(task_id: int):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return

        task.status = "processing"
        db.commit()
        append_log(db, task, "start", "质检任务进入处理队列")

        if task.content_type != "image":
            raise ValueError("当前系统仅支持螺丝图片质检，请上传 JPG/PNG 等图像文件")

        append_log(db, task, "variety", "MobileNet / EfficientNet 品种分类")
        variety = classify_variety(task.content_url)
        task.variety_result = json.dumps(variety, ensure_ascii=False)
        db.commit()
        append_log(db, task, "variety", variety.get("message", "品种分类完成"), {
            "class_id": variety.get("class_id"),
            "label_zh": variety.get("label_zh"),
            "confidence": variety.get("confidence"),
            "backbone": variety.get("backbone"),
            "strategy_id": (variety.get("sorting_strategy") or {}).get("strategy_id"),
        })

        append_log(db, task, "coarse", "YOLOv8 粗筛")
        screen = coarse_screen(task.content_url)
        append_log(db, task, "coarse", screen.get("message", "粗筛完成"), {
            "enabled": screen.get("enabled"),
            "skip_vlm": screen.get("skip_vlm"),
            "trigger_vlm": screen.get("trigger_vlm"),
            "max_confidence": screen.get("max_confidence"),
            "primary_class": screen.get("primary_class"),
        })

        if screen.get("skip_vlm"):
            ai_result = coarse_result_to_ai_result(screen)
            append_log(db, task, "model", "粗筛快速放行，跳过 VLM 精查")
        else:
            append_log(db, task, "model", "VLM 精查（豆包多模态）")
            try:
                ai_result = await asyncio.wait_for(
                    doubao_service.analyze_screw_image(task.content_url, screen),
                    timeout=settings.vlm_timeout_seconds,
                )
                append_log(db, task, "model", "VLM 精查完成", {
                    "source": "vlm",
                    "violation_types": ai_result.get("violation_types"),
                })
            except asyncio.TimeoutError:
                ai_result = coarse_result_to_ai_result(screen)
                ai_result["risk_description"] = (
                    f"VLM 超时（>{settings.vlm_timeout_seconds}s），已回退 YOLO 粗筛结果"
                )
                ai_result["source"] = "yolo_fallback_timeout"
                append_log(db, task, "model", "VLM 超时，回退粗筛结果", {"timeout_s": settings.vlm_timeout_seconds})

        _apply_ai_result(task, ai_result)
        db.commit()
        append_log(db, task, "model", "缺陷识别完成", {
            "confidence": task.confidence,
            "violation_types": ai_result.get("violation_types", []),
            "violation_detected": ai_result.get("violation_detected", False),
        })

        append_log(db, task, "rule_match", "匹配分拣策略规则")
        judgment = rule_engine.judge(ai_result, db)
        task.matched_rules = json.dumps(judgment["matched_rules"], ensure_ascii=False)
        db.commit()
        append_log(db, task, "rule_match", "策略匹配完成", {
            "matched_count": len(judgment["matched_rules"]),
            "matched_rules": judgment["matched_rules"],
        })

        append_log(db, task, "decision", "分拣决策完成", {
            "final_action": judgment["action"],
            "reason": judgment["reason"],
        })

        append_log(db, task, "pose", "6D 位姿估计")
        pose = estimate_pose_6d(task.content_url)
        task.pose_result = json.dumps(pose, ensure_ascii=False)
        if pose.get("success"):
            cam = pose.get("camera_frame", {})
            rob = pose.get("robot_frame", {})
            append_log(db, task, "pose", "位姿估计完成", {
                "method": pose.get("method"),
                "confidence": pose.get("confidence"),
                "camera_t_mm": cam.get("translation_mm"),
                "robot_t_mm": rob.get("translation_mm"),
                "reprojection_error_px": pose.get("reprojection_error_px"),
                "vector_6d": (pose.get("pose_6d_robot") or pose.get("pose_6d") or {}).get("vector_6d"),
            })
        else:
            append_log(db, task, "pose", pose.get("message", "位姿估计未成功"), {
                "method": pose.get("method"),
                "errors": pose.get("errors"),
            })

        task.status = judgment["action"]
        task.updated_at = datetime.now()
        db.commit()

    except Exception as e:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "error"
            task.risk_description = str(e)
            task.updated_at = datetime.now()
            db.commit()
            append_log(db, task, "error", "任务处理失败", {"error": str(e)})
    finally:
        db.close()


def process_task_sync(task_id: int):
    asyncio.run(process_task(task_id))
