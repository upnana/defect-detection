"""初始化螺丝缺陷分拣策略与规则（对应 screw-dataset 六类缺陷）"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.database import engine, SessionLocal, Base
from app.models import Policy, Rule

Base.metadata.create_all(bind=engine)

SEED_POLICIES = [
    {
        "name": "良品放行策略",
        "description": "good 类无缺陷螺丝，粗筛/VLM 确认后快速放行",
        "rules": [
            {
                "name": "良品快速放行",
                "description": "外观完整、无篡改与划痕、螺纹正常",
                "violation_type": "good",
                "action": "pass",
                "priority": 10,
            },
        ],
    },
    {
        "name": "正面篡改检测策略",
        "description": "manipulated_front：正面被篡改或被动手脚",
        "rules": [
            {
                "name": "正面篡改拦截",
                "description": "头部正面纹理异常、装配痕迹或人为改动",
                "violation_type": "manipulated_front",
                "action": "block",
                "priority": 10,
            },
            {
                "name": "正面可疑复检",
                "description": "正面区域对比度异常但不确定",
                "violation_type": "manipulated_front",
                "action": "review",
                "priority": 6,
            },
        ],
    },
    {
        "name": "头部划痕检测策略",
        "description": "scratch_head：螺丝头部表面划痕",
        "rules": [
            {
                "name": "头部位划痕拦截",
                "description": "头部可见线性或点状划伤",
                "violation_type": "scratch_head",
                "action": "block",
                "priority": 9,
            },
        ],
    },
    {
        "name": "颈部划痕检测策略",
        "description": "scratch_neck：螺杆颈部区域划痕",
        "rules": [
            {
                "name": "颈部划痕拦截",
                "description": "颈部柱面出现划痕或磕碰",
                "violation_type": "scratch_neck",
                "action": "block",
                "priority": 9,
            },
        ],
    },
    {
        "name": "侧面螺纹缺陷策略",
        "description": "thread_side：侧面螺纹/线纹缺陷",
        "rules": [
            {
                "name": "侧面螺纹拦截",
                "description": "侧面牙型不完整、乱纹或缺损",
                "violation_type": "thread_side",
                "action": "block",
                "priority": 9,
            },
            {
                "name": "侧面螺纹复检",
                "description": "侧面纹理轻微异常",
                "violation_type": "thread_side",
                "action": "review",
                "priority": 5,
            },
        ],
    },
    {
        "name": "顶部螺纹缺陷策略",
        "description": "thread_top：顶部螺纹/线纹缺陷",
        "rules": [
            {
                "name": "顶部螺纹拦截",
                "description": "顶部螺纹区域缺损或加工异常",
                "violation_type": "thread_top",
                "action": "block",
                "priority": 9,
            },
        ],
    },
]


def seed():
    db = SessionLocal()
    existing = db.query(Policy).count()
    if existing > 0:
        print(f"数据库已有 {existing} 条策略，跳过种子数据初始化")
        print("如需重置为缺陷质检策略，请删除 screw_inspect.db 后重新运行本脚本")
        db.close()
        return

    for p_data in SEED_POLICIES:
        policy = Policy(name=p_data["name"], description=p_data["description"])
        for r_data in p_data["rules"]:
            rule = Rule(**r_data)
            policy.rules.append(rule)
        db.add(policy)

    db.commit()
    print(f"成功初始化 {len(SEED_POLICIES)} 条缺陷分拣策略")

    skip_faiss = os.getenv("SKIP_FAISS_REBUILD", "0") == "1"
    if skip_faiss:
        print("已跳过 FAISS 索引构建（SKIP_FAISS_REBUILD=1）")
    else:
        try:
            from app.services.faiss_service import faiss_service

            faiss_service.rebuild_from_db(db)
            print("FAISS索引构建完成")
        except Exception as e:
            print(f"FAISS索引构建失败（不影响策略入库）: {e}")

    db.close()


if __name__ == "__main__":
    seed()
