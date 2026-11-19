#!/usr/bin/env python3
"""SceneVersion → MuJoCo 场景构建（PRD v0.5 §6.3 / §14 gym/scene.py 泛化）。

消费冻结的 SceneVersion 快照（scene_version.py 发布产物），消除对固定
物体名/坐标的硬编码（PRD §15.3 第 4 条）。物体名由 AssetInstance 的
instance_id 决定；角色（grasp_target / obstacle / prop）驱动技能链。

与旧 tabletop 场景兼容：mug/tray 实例命名时技能链无感迁移。
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path

import mujoco
import numpy as np

from . import spec as sp
from .scene_version import load_scene

DEFAULT_PHASES = {
    "settle_s": 8.0, "approach_s": 5.0, "descend_s": 3.0, "grasp_z_offset_m": 0.02,
    "pregrasp_lift_m": 0.06, "grasp_s": 1.5, "grasp_confirm_timeout_s": 3.0,
    "lift_s": 2.5, "lift_height_m": 0.08, "hold_s": 3.0, "move_s": 3.0,
    "release_s": 1.5, "verify_s": 3.0, "retreat_s": 3.0, "place_settle_s": 1.2,
    "drop_settle_s": 1.0, "place_bias_xy_m": [0.015, 0.0], "place_descend_m": 0.0,
    "retreat_lift_m": 0.08, "shift_xy_m": [0.0, 0.0], "shift_s": 1.5,
    "move_corridor_m": 0.53,
}

# 实例几何默认值：真实 Lux3D 资产接入前的占位（PRD §8.1 允许人工候选/参数，
# 来源标注 manual；接入 GLB 资产后由 asset_pipeline 提供真实几何）
_INSTANCE_GEOMETRY = {
    "default": {
        "mug_like": {"type": "cylinder", "size": [0.0305, 0.05], "mass": 0.25,
                     "friction": 0.7, "half_h": 0.05, "rgba": "0.85 0.87 0.9 1"},
        "box_like": {"type": "box", "size": [0.12, 0.08, 0.05], "mass": 0.5,
                     "friction": 0.8, "half_h": 0.05, "rgba": "0.72 0.55 0.35 1"},
        "tray_like": {"type": "box", "size": [0.115, 0.085, 0.005], "mass": 0.4,
                      "friction": 0.8, "half_h": 0.005, "rgba": "0.8 0.72 0.55 1"},
    }
}


class SceneV1:
    """从 SceneVersion 快照构建的 MuJoCo 场景。

    用法：
        sc = SceneV1.from_scene_version("s2_livingroom")
        sc.reset(seed=0)
        # sc.grasp_instance / sc.place_region 由角色/区域驱动
    """

    def __init__(self, scene: dict):
        self.sv = scene
        self.scan_mug = next((i for i in scene.get("instances") or []
                              if i.get("asset_id") == "lux3d_mug_demo"
                              and (i.get("extra") or {}).get("geometry_kind") == "scan_mug"), None)
        self.task = None  # 兼容旧 Scene 接口的关键字段由 property 提供
        self.xml_path = self._build()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = 0.002
        self.model.opt.impratio = 20.0
        self.model.opt.noslip_iterations = 3
        self._apply_runtime_patches()
        self.data = mujoco.MjData(self.model)
        self._idx()

    def _apply_runtime_patches(self):
        """Keep the legacy finger-pad collision model for other room tasks."""
        if self.scan_mug:
            return
        m = self.model
        for g in range(m.ngeom):
            bid = m.geom_bodyid[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.body(bid).name in ("link6", "link7", "link8"):
                m.geom_contype[g] = m.geom_conaffinity[g] = 0

    @classmethod
    def from_scene_version(cls, scene_id: str, version_id: str | None = None):
        s = load_scene(scene_id)
        if version_id and s.get("published", {}).get("version_id") != version_id:
            raise ValueError(f"scene {scene_id} 当前版本不是 {version_id}")
        if not s.get("published"):
            raise ValueError(f"scene {scene_id} 未发布，不能用于运行（PRD §11.1）")
        return cls(s)

    # ------------------------------------------------------------ build
    # 指垫碰撞代理（复用旧 Scene 验证过的几何：指尖段盒垫，PRD F2.2）
    _PAD_GEOM = ('<geom name="{name}" type="box" size="0.026 0.021 0.0105" '
                 'pos="0 -0.019 0.0135" friction="2.0 0.005 0.0001" '
                 'condim="3" priority="1" />')

    def _patched_robot_xml(self, piper_dir: Path, uid: str) -> Path:
        """Compile jaw hulls for the scan, or retain the legacy pad colliders."""
        src = (piper_dir / "S10_Piper.xml").read_text()
        mesh_disabled = 'contype="0" conaffinity="0"'
        finger_contact = 'friction="2 0.005 0.0001" solref="0.004 1" priority="1"'
        if not self.scan_mug:
            palm = '<geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="gripper_base"'
            src = src.replace(f'{palm} />', f'{palm} {mesh_disabled} />')
        contact = finger_contact if self.scan_mug else mesh_disabled
        for fname, pad in (("link7", "pad_left"), ("link8", "pad_right")):
            proxy = self._PAD_GEOM.format(name=pad)
            if self.scan_mug:
                # Leave the pad center available for approach planning. Contacts use only the hull.
                proxy = proxy.replace("/>", f'{mesh_disabled} />')
            finger = f'<geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="{fname}"'
            src = src.replace(f'{finger} />', f'{finger} {contact} />\n    {proxy}')
        out = piper_dir / f"_gen_sv_patch_{uid}.xml"
        out.write_text(src)
        return out

    def _build(self) -> Path:
        sdk = sp.sdk_root()
        piper_dir = sdk / "S10_description" / "S10_Piper"
        if not (piper_dir / "S10_Piper.xml").is_file():
            raise FileNotFoundError(f"官方模型未找到: {piper_dir}")
        # 环境几何：地板（标定 floor_z）+ 代理 + 实例
        cal = self.sv["calibration"]
        floor_z = cal.get("floor_z") or 0.0
        parts = [self._proxies_xml(floor_z)]
        parts.append(self._instances_xml())
        scan_mesh = ""
        if self.scan_mug:
            scale = float(self.scan_mug["extra"]["scan_transform"]["scale"])
            path = escape(str(sp.REPO / "gym/assets/mug_visual.stl"), quote=True)
            scan_mesh = f'<mesh name="scan_mug" file="{path}" scale="{scale} {scale} {scale}"/>\n'
        env = ('<mujoco model="sv_env">\n'
               '<asset>\n'
               '<texture name="wood" type="2d" builtin="checker" '
               'rgb1="0.55 0.4 0.28" rgb2="0.45 0.32 0.22" width="64" height="64"/>\n'
               '<material name="mat_wood" texture="wood" texuniform="true"/>\n'
               + scan_mesh +
               '</asset>\n'
               '<worldbody>\n' + "\n".join(parts) + '\n</worldbody>\n</mujoco>')
        work = self._work_dir()
        import uuid
        uid = uuid.uuid4().hex[:8]
        # meshdir 相对【顶层 XML】目录解析 → patched/main 都放 piper_dir（UUID 隔离）。
        # env.xml 无 mesh 引用、无 compiler 覆盖（评审指出的 meshdir 覆盖 bug 根除）。
        patched = self._patched_robot_xml(piper_dir, uid)
        main = piper_dir / f"_gen_sv_{uid}.xml"
        grasp = next((i["instance_id"] for i in self.sv.get("instances") or []
                      if i.get("role") == "grasp_target"), None)
        # 无 grasp 实例的场景（如房间碰撞验证）不生成 equality 块
        eq = ('  <equality>\n'
              f'    <weld name="grasp_weld" body1="{grasp}" body2="link6" active="false" '
              'solref="0.005 1" solimp="0.9999 0.9999 0.0001 0.5 2"/>\n'
              '  </equality>\n') if grasp else ''
        main.write_text(
            f'<mujoco model="gym_sv_{self.sv["scene_id"]}">\n'
            f'  <include file="{patched.name}" />\n'
            f'  <include file="{(work / "env.xml").resolve()}" />\n'
            f'{eq}</mujoco>')
        (work / "env.xml").write_text(env)
        self._gen_files = [main, patched, work / "env.xml"]
        return main

    def _work_dir(self) -> Path:
        import uuid
        d = sp.REPO / "runs" / "_work" / ("sv_" + uuid.uuid4().hex[:12])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _proxies_xml(self, floor_z: float = 0.0) -> str:
        # 官方 S10_Piper.xml 自带 name="floor" 平面（z=0）；标定 floor_z=0 时直接
        # 复用，不重复定义（重名会编译失败）。floor_z≠0 时需补丁官方 XML（后续）。
        if abs(floor_z) < 1e-9:
            body = []
        else:
            raise ValueError(f"标定 floor_z={floor_z}≠0：需扩展官方地板补丁（TODO）")
        for p in self.sv.get("proxies") or []:
            pid = p["proxy_id"]
            pose = p["pose"]            # [x, y, z]（存档约定：世界 Z-up 的 [x,z,y] 序）
            size = p["size"]
            if p["shape"] == "box":
                yaw = math.radians(p.get("yaw_deg", 0.0)) / 2
                orientation = f'quat="{math.cos(yaw)} 0 0 {math.sin(yaw)}"'
                # 统一 [x, y, z] 世界 Z-up 序（PRD §10.3）——不再用 [x,z,y] 翻转
                geom = (f'<geom name="{pid}" type="box" size="{size[0]} {size[1]} {size[2]}" '
                        f'pos="{pose[0]} {pose[1]} {pose[2]}" {orientation} material="mat_wood" '
                        f'friction="0.8 0.01 0.002"/>')
            elif p["shape"] == "cylinder":
                geom = (f'<geom name="{pid}" type="cylinder" size="{size[0]} {size[1]}" '
                        f'pos="{pose[0]} {pose[1]} {pose[2]}" material="mat_wood"/>')
            else:
                continue  # convex/组合：接入 asset_pipeline 后支持
            if p.get("kind") == "support":
                geom = geom.replace("/>", 'contype="0" conaffinity="0"/>') \
                    if False else geom   # support 保留碰撞（杯子要落上去）；
                # 但底盘不该撞进桌面——nav 层已把 support 视为可伸臂不挡路，
                # 物理层保留碰撞让物品有支撑。桌面高度>底盘高度，行走时臂不碰即可。
            body.append("  " + geom)
        return "\n".join(body)

    def _instances_xml(self) -> str:
        """AssetInstance → freejoint body。几何从实例 asset 类型映射（占位几何表）。"""
        out = []
        for inst in self.sv.get("instances") or []:
            iid = inst["instance_id"]
            pose = inst["pose"]        # [x, y, z]
            g = self._instance_geom(inst)
            mass = g["mass"]
            if inst is self.scan_mug:
                transform = inst["extra"]["scan_transform"]
                local_pos = " ".join(str(float(v)) for v in transform["position"])
                local_quat = " ".join(str(float(v)) for v in transform["quaternion_wxyz"])
                shape = f'type="mesh" mesh="scan_mug" pos="{local_pos}" quat="{local_quat}"'
            else:
                sz_str = " ".join(str(s) for s in g["size"])
                shape = f'type="{g["type"]}" size="{sz_str}"'
            contact_priority = ' priority="1"' if (inst.get("extra") or {}).get("material") else ""
            out.append(
                f'  <body name="{iid}" pos="{pose[0]} {pose[1]} {pose[2]}">\n'
                f'    <freejoint name="{iid}_joint"/>\n'
                f'    <inertial pos="0 0 {g["half_h"]}" mass="{mass}" '
                f'diaginertia="{mass*1e-3:.6f} {mass*1e-3:.6f} {mass*1e-3:.6f}"/>\n'
                f'    <geom name="{iid}_c" {shape} '
                f'friction="{g["friction"]} 0.01 0.002"{contact_priority} solref="0.005 1" '
                f'rgba="{g["rgba"]}" group="1"/>\n'
                f'  </body>')
        return "\n".join(out)

    def _instance_geom(self, inst: dict) -> dict:
        # kind = 实例 asset 几何种类；scene schema 的 instances[i].extra.geometry_kind
        extra = inst.get("extra") or {}
        kind = extra.get("geometry_kind", "mug_like")
        table = _INSTANCE_GEOMETRY["default"]
        geom = dict(table.get(kind, table["mug_like"]))
        if "size_xyz_m" in extra:
            width, depth, height = extra["size_xyz_m"]
            geom["half_h"] = height / 2
            geom["size"] = ([max(width, depth) / 2, height / 2]
                            if geom["type"] == "cylinder" else [width / 2, depth / 2, height / 2])
        geom["mass"] = extra.get("mass_kg", geom["mass"])
        geom["friction"] = extra.get("friction", geom["friction"])
        return geom

    # -------------------------------------------------------------- idx
    def _idx(self) -> None:
        self._robot = None
        m = self.model
        def jadr(name):
            j = m.joint(name).id
            return m.jnt_qposadr[j], m.jnt_dofadr[j]
        from .scene import DOG_JOINTS, ARM_JOINTS, FINGER_JOINTS, DOG_IDLE
        self.DOG_IDLE = DOG_IDLE
        self.dq, self.dv = zip(*[jadr(n) for n in DOG_JOINTS])
        self.aq, self.av = zip(*[jadr(n) for n in ARM_JOINTS])
        self.fq, self.fv = zip(*[jadr(n) for n in FINGER_JOINTS])
        self.dqa = [m.actuator(n).id for n in DOG_JOINTS]
        self.ara = [m.actuator(n).id for n in ARM_JOINTS]
        self.fra = [m.actuator(n).id for n in FINGER_JOINTS]
        self.fbq, self.fbv = jadr("floating_base")
        self.ee = m.site("EE_site").id
        self.finger_body_ids = (m.body("link7").id, m.body("link8").id)
        self.instance_ids = [i["instance_id"] for i in self.sv.get("instances") or []]
        self.obj_geoms = {}
        self.obj_jadr = {}    # qpos 地址（freejoint: 7 项 pos+quat）
        self.obj_vadr = {}    # dof 地址（freejoint: 6 项 线速+角速）—— 速度必须用这个
        for name in self.instance_ids:
            bid = m.body(name).id
            self.obj_geoms[name] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
            jid = m.body_jntadr[bid]
            self.obj_jadr[name] = m.jnt_qposadr[jid]
            self.obj_vadr[name] = m.jnt_dofadr[jid]
        # 角色映射
        self.grasp_instance = next(
            (i["instance_id"] for i in self.sv.get("instances") or []
             if i.get("role") == "grasp_target"), None)
        self.pick_region = next(
            (r for r in (self.sv.get("regions") or {}).values() if r["kind"] == "pick"), None)
        self.place_region = next(
            (r for r in (self.sv.get("regions") or {}).values() if r["kind"] == "place"), None)

    # ------------------------------------------------------------ reset
    def reset(self, seed: int = 0) -> dict:
        import time as _t
        mujoco.mj_resetData(self.model, self.data)
        m, d = self.model, self.data
        rs = self.sv.get("robot_start") or {"pos": [0, 0], "yaw_deg": 0.0}
        yaw = math.radians(rs.get("yaw_deg", 0.0))
        d.qpos[:3] = [rs["pos"][0], rs["pos"][1], 0.225]
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for a, q in zip(self.dq, self.DOG_IDLE):
            d.qpos[a] = q
        for a in self.aq:
            d.qpos[a] = 0.0
        for a, v in zip(self.fq, (0.035, -0.035)):
            d.qpos[a] = v
        # 实例位置抖动（可复现；物品 xy ±5mm）
        rng = np.random.default_rng(seed)
        for inst in self.sv.get("instances") or []:
            iid = inst["instance_id"]
            a = self.obj_jadr[iid]
            pose = inst["pose"]
            jx = float(rng.uniform(-0.005, 0.005))
            jy = float(rng.uniform(-0.005, 0.005))
            d.qpos[a:a + 3] = [pose[0] + jx, pose[1] + jy, pose[2]]
            d.qpos[a + 3:a + 7] = [1.0, 0, 0, 0]
        mujoco.mj_forward(m, d)
        floor = m.geom("floor").id
        robot_root = m.body("base_link").id
        min_distance = min(
            (float(c.dist) for c in d.contact
             if ((c.geom1 == floor and m.body_rootid[m.geom_bodyid[c.geom2]] == robot_root)
                 or (c.geom2 == floor and m.body_rootid[m.geom_bodyid[c.geom1]] == robot_root))),
            default=0.0,
        )
        if min_distance < 0:
            d.qpos[self.fbq + 2] += -min_distance + 0.001
            mujoco.mj_forward(m, d)
        return {"seed": seed, "reset_at": _t.strftime("%H:%M:%S")}

    # ---------------------------------------------------------- helpers
    def obj_pos(self, name: str) -> np.ndarray:
        a = self.obj_jadr[name]
        return self.data.qpos[a:a + 3].copy()

    def obj_vel(self, name: str) -> np.ndarray:
        # 评审1-3：速度必须用 DOF 地址（freejoint qpos 7 项 / dof 6 项地址不同）
        a = self.obj_vadr[name]
        return self.data.qvel[a:a + 6].copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee].copy()

    # ------------------------------------------------ 兼容旧 TaskExecutor
    @property
    def task(self):
        """兼容属性：桌面判据/反馈/相位默认值（SceneVersion 可覆盖 extra.task_overrides）。"""
        if self._task_compat is None:
            from .spec import CriteriaSpec, FeedbackSpec, ControlSpec
            ov = (self.sv.get("extra") or {}).get("task_overrides", {})
            self._task_compat = type("CompatTask", (), {
                "criteria": CriteriaSpec(**ov.get("criteria", {})),
                "feedback": FeedbackSpec(**ov.get("feedback", {})),
                # 与官方 TaskSpec（scenes/tabletop_pick_place.json）一致：
                # 之前硬编码 300/15/45 → clip 下阻尼饱和 bang-bang 振荡，推进段臂打飞
                "control": ControlSpec(**{"arm_kp": 120.0, "arm_kd": 6.0,
                                          "arm_clip": 30.0,
                                          "base_mode": "kinematic_hold",
                                          **ov.get("control", {})}),
                "phases": {**DEFAULT_PHASES, **ov.get("phases", {})},
                "regions": self.sv.get("regions", {}),
                "name": self.sv["scene_id"],
                "home_q": [0.0, 1.2, -1.2, 0.0, -1.2, 0.0],
            })()
        return self._task_compat

    @task.setter
    def task(self, v):
        self._task_compat = v

    _task_compat = None

    # ------------------------------------------------ 底盘行走（位置伺服）
    def drive_to(self, target_xy: list[float], speed: float = 0.15,
                 tol: float = 0.03, max_t: float = 25.0, verbose: bool = True,
                 carrying: bool = False, arm_q: np.ndarray | None = None,
                 grip_gap: float | None = None) -> dict:
        """浮动基座位置伺服行走（G3 动态底座第一步：不锁底座，物理滑行）。

        注意：这是【位置伺服仿真辅助行走】，不是官方策略轨道——披露于 run 记录。
        返回 {reached, dist, t}。"""
        from .robot import WHEEL_INDICES
        m, d = self.model, self.data
        t0 = d.time
        reached = False
        wheel_a = [self.dqa[i] for i in WHEEL_INDICES]
        wheel_joints = m.actuator_trnid[wheel_a, 0]
        yaw0 = d.qpos[self.fbq + 3:self.fbq + 7].copy()   # 起始朝向保持（伺服辅助披露）
        while d.time - t0 < max_t:
            d.qpos[self.fbq + 3:self.fbq + 7] = yaw0
            d.qvel[self.fbv + 3:self.fbv + 6] = 0.0
            cur = d.qpos[self.fbq:self.fbq + 2].copy()
            err = np.array(target_xy) - cur
            dist = float(np.linalg.norm(err))
            if dist < tol:
                reached = True
                break
            # 速度伺服：朝目标方向按 speed 移动（施加基座 qvel 目标）
            v_des = err / max(dist, 1e-6) * min(speed, dist * 2.0)
            # 通过腿部接触摩擦滑行不可靠 → 直接设定基座水平速度（伺服辅助，披露）
            d.qvel[self.fbv:self.fbv + 2] = v_des
            d.qvel[self.fbv + 5] = 0.0   # 锁 yaw：IK 假设底座朝向不变
            # 腿部 PD 保持站姿；臂/夹爪：carrying 时保持当前状态，空载时回 home+开爪
            c = np.zeros(m.nu)
            q = d.qpos[list(self.dq)]; dq = d.qvel[list(self.dv)]
            c[self.dqa] = np.clip(40 * (self.DOG_IDLE - q) - 2 * dq, -20, 20)
            roll_directions = np.cross(d.xaxis[wheel_joints], [0, 0, 1])[:, :2]
            wheel_rates = (roll_directions @ v_des) / .081
            c[wheel_a] = np.clip(2.4 * (wheel_rates - dq[WHEEL_INDICES]), -14, 14)
            if carrying and arm_q is not None:
                self.robot.arm(c, arm_q)
                if grip_gap is not None:
                    c[self.fra] = np.clip(300 * (np.array([grip_gap, -grip_gap])
                        - d.qpos[list(self.fq)]) - 2 * d.qvel[list(self.fv)], -10, 10)
            else:
                self.robot.arm(c, np.asarray(self.task.home_q))
                c[self.fra] = np.clip(300 * (np.array([0.035, -0.035]) - d.qpos[list(self.fq)])
                                      - 2 * d.qvel[list(self.fv)], -10, 10)
            d.ctrl[:] = c
            mujoco.mj_step(m, d)
            if self.frame_recorder is not None:
                self.frame_recorder.maybe_snap(d)
        # 停稳：速度清零后原地站立（carrying 时保持臂/爪）
        d.qvel[self.fbv:self.fbv + 6] = 0.0
        for _ in range(int(1.5 / 0.002)):
            c = np.zeros(m.nu)
            q = d.qpos[list(self.dq)]; dq = d.qvel[list(self.dv)]
            c[self.dqa] = np.clip(40 * (self.DOG_IDLE - q) - 2 * dq, -20, 20)
            c[wheel_a] = np.clip(-2 * dq[WHEEL_INDICES], -14, 14)
            if carrying and arm_q is not None:
                self.robot.arm(c, arm_q)
                if grip_gap is not None:
                    c[self.fra] = np.clip(300 * (np.array([grip_gap, -grip_gap])
                        - d.qpos[list(self.fq)]) - 2 * d.qvel[list(self.fv)], -10, 10)
            else:
                self.robot.arm(c, np.asarray(self.task.home_q))
                c[self.fra] = np.clip(300 * (np.array([0.035, -0.035]) - d.qpos[list(self.fq)])
                                      - 2 * d.qvel[list(self.fv)], -10, 10)
            d.ctrl[:] = c
            mujoco.mj_step(m, d)
            if self.frame_recorder is not None:
                self.frame_recorder.maybe_snap(d)
        final = d.qpos[self.fbq:self.fbq + 2].copy()
        dist = float(np.linalg.norm(np.array(target_xy) - final))
        return {"reached": reached, "final_pos": final.round(3).tolist(),
                "dist": round(dist, 3), "t": round(float(d.time - t0), 1)}


    frame_recorder = None

    @property
    def robot(self):
        if self._robot is None:
            from .robot import Robot
            self._robot = Robot(self)
        return self._robot
    # ------------------------------------------------------------- weld
    def weld_arm(self) -> None:
        """以当前相对位姿激活 grasp weld（披露的保持基底，PRD §8 fallback）。"""
        wid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        if wid < 0:
            return
        m, d = self.model, self.data
        bid = m.body(self.grasp_instance).id if self.grasp_instance else 0
        b2 = m.body("link6").id
        p1, p2 = d.xpos[bid], d.xpos[b2]
        R1 = np.zeros(9)
        mujoco.mju_quat2Mat(R1, d.xquat[bid])
        R1 = R1.reshape(3, 3)
        rel_p = R1.T @ (p2 - p1)
        q1 = np.asarray(d.xquat[bid], float)
        q2 = np.asarray(d.xquat[b2], float)
        w1, x1, y1, z1 = q1[0], -q1[1], -q1[2], -q1[3]
        q_rel = np.array([w1 * q2[0] - x1 * q2[1] - y1 * q2[2] - z1 * q2[3],
                          w1 * q2[1] + x1 * q2[0] + y1 * q2[3] - z1 * q2[2],
                          w1 * q2[2] - x1 * q2[3] + y1 * q2[0] + z1 * q2[1],
                          w1 * q2[3] + x1 * q2[2] - y1 * q2[1] + z1 * q2[0]])
        m.eq_data[wid, 0:3] = 0.0
        m.eq_data[wid, 3:6] = rel_p
        m.eq_data[wid, 6:10] = q_rel
        d.eq_active[wid] = 1

    def weld_release(self) -> None:
        wid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        if wid >= 0:
            self.data.eq_active[wid] = 0
