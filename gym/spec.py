#!/usr/bin/env python3
"""TaskSpec：场景/任务/条件/种子的参数化定义。

平台能力 #1（PRODUCT.md §2）：换参数不改代码。所有位姿、增益、成功判据、
条件参数都来自 scenes/*.json；代码只消费 TaskSpec。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GYM_DIR = Path(__file__).resolve().parent
REPO = GYM_DIR.parent
ASSETS = GYM_DIR / "assets"

# 官方 S10+Piper 模型根（含 S10_description/）。可用环境变量覆盖。
SDK_ROOT_DEFAULT = REPO / "spikes" / "s10-sdk2" / "sim"


def sdk_root() -> Path:
    return Path(os.environ.get("S10_SDK_ROOT", str(SDK_ROOT_DEFAULT)))


# ---------------------------------------------------------------- TaskSpec


@dataclass
class ObjectSpec:
    name: str
    # 摆放位姿（世界系，米 / [w,x,y,z] 四元数）
    pos: list[float] = field(default_factory=list)
    quat: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    # 物理参数（来源标注 per PRD F2.5：model / estimated / manual）
    mass_kg: float | None = None
    friction_slide: float | None = None      # None = 用模型自带值
    source: str = "manual"


@dataclass
class ControlSpec:
    dt: float = 0.002
    # 腿部 PD（保持站立姿态）
    leg_kp: float = 40.0
    leg_kd: float = 2.0
    leg_clip: float = 20.0
    # 臂部位置伺服
    arm_kp: float = 120.0
    arm_kd: float = 6.0
    arm_clip: float = 30.0
    # 夹爪位置伺服（物理上限 = 执行器 forceclip，两模式一致）
    grip_kp: float = 100.0
    grip_kd: float = 2.0
    grip_clip: float = 10.0
    # 底座处理：pd_stance（腿部 PD 物理保持）| kinematic_hold（运动学保持，披露）
    base_mode: str = "pd_stance"
    # MuJoCo 摩擦约束刚度（抓取防蠕滑，文档推荐 >10）
    impratio: float = 20.0
    noslip_iterations: int = 2


@dataclass
class CriteriaSpec:
    grasp_lift_cm: float = 5.0        # 离支撑面高度
    grasp_hold_s: float = 3.0         # 保持时长
    place_stable_s: float = 2.0
    place_speed_max: float = 0.02     # 判稳定用的速度阈值 m/s
    finger_force_limit_N: float = 30.0  # 触觉保护边界（两模式公开一致）
    contact_force_min_N: float = 0.3    # 判“有接触”的单指力阈值


@dataclass
class FeedbackSpec:
    """触觉反馈状态机参数（F4.3）。fixed 模式忽略本节。"""
    close_gap_m: float = 0.012       # 首次闭合目标开度（每指行程）
    min_gap_m: float = 0.002
    tighten_step_m: float = 0.004    # 每次加力收紧量
    max_tighten: int = 4
    slip_trigger_m: float = 0.004    # 相对爪滑动量触发阈值
    force_drop_trigger_N: float = 0.5  # 提起后单指力跌落触发
    resume_after_s: float = 0.5      # 收紧后等待


@dataclass
class TaskSpec:
    name: str
    version: int
    robot: dict                     # start_xy 等
    objects: dict[str, ObjectSpec]
    regions: dict[str, dict]        # place 区域
    criteria: CriteriaSpec
    control: ControlSpec
    feedback: FeedbackSpec
    conditions: dict[str, dict]     # hidden env params，仅仿真侧可见
    seed_jitter: dict               # xy / yaw_deg
    phases: dict[str, float]        # 各技能时长 s
    home_q: list[float] = field(default_factory=lambda: [0.0, 2.54, -1.26, 0.0, -1.22, 0.0])

    @property
    def config_hash(self) -> str:
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_scene(name: str) -> TaskSpec:
    """从 scenes/<name>.json 载入；缺省值由 dataclass 提供。"""
    path = REPO / "scenes" / f"{name}.json"
    raw = json.loads(path.read_text())
    objects = {k: ObjectSpec(name=k, **v) for k, v in raw.get("objects", {}).items()}
    crit = CriteriaSpec(**raw.get("criteria", {}))
    ctrl = ControlSpec(**raw.get("control", {}))
    fb = FeedbackSpec(**raw.get("feedback", {}))
    return TaskSpec(
        name=raw.get("name", name),
        version=raw.get("version", 1),
        robot=raw.get("robot", {}),
        objects=objects,
        regions=raw.get("regions", {}),
        criteria=crit,
        control=ctrl,
        feedback=fb,
        conditions=raw.get("conditions", {}),
        seed_jitter=raw.get("seed_jitter", {"xy": 0.003, "yaw_deg": 1.0}),
        phases=raw.get("phases", {}),
        home_q=raw.get("home_q", [0.0, 2.54, -1.26, 0.0, -1.22, 0.0]),
    )


def condition_params(spec: TaskSpec, condition: str) -> dict:
    """条件参数是隐藏环境参数（PRD F4.4）：只对仿真侧开放，绝不进控制器观测。"""
    return dict(spec.conditions.get(condition) or {})
