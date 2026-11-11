from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
import os

from app.database import engine, Base
from app.config import settings
from app.routers import tasks, policies, dashboard, batches, meta

Base.metadata.create_all(bind=engine)


def ensure_task_columns():
    with engine.connect() as conn:
        columns = conn.execute(text("PRAGMA table_info(tasks)")).fetchall()
        col_names = {row[1] for row in columns}
        if "process_logs" not in col_names:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN process_logs TEXT DEFAULT ''"))
            conn.commit()
        if "batch_id" not in col_names:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN batch_id INTEGER"))
            conn.commit()
        if "original_filename" not in col_names:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN original_filename TEXT DEFAULT ''"))
            conn.commit()
        if "pose_result" not in col_names:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN pose_result TEXT DEFAULT ''"))
            conn.commit()
        if "variety_result" not in col_names:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN variety_result TEXT DEFAULT ''"))
            conn.commit()


ensure_task_columns()

app = FastAPI(title="ScrewInspect 螺丝缺陷质检", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.upload_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(batches.router, prefix="/api/batches", tags=["batches"])
app.include_router(policies.router, prefix="/api/policies", tags=["policies"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(meta.router, prefix="/api/meta", tags=["meta"])


@app.get("/api/health")
def health():
    from app.config import settings
    from app.services import pose_path_tinydetr, pose_path_yolo, variety_service

    return {
        "status": "ok",
        "variety": {
            "enabled": settings.variety_enabled,
            "configured": variety_service.is_configured(),
            "backbone": settings.variety_backbone,
        },
        "pose": {
            "mode": settings.pose_mode,
            "yolo_pose_configured": pose_path_yolo.is_configured(),
            "tinydetr_configured": pose_path_tinydetr.is_configured(),
            "contour_fallback": settings.pose_contour_fallback,
        },
    }
