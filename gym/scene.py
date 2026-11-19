#!/usr/bin/env python3
"""MuJoCo 场景：官方 S10+Piper 模型 + 任务物体，支持按种子重置与条件参数注入。

条件参数（摩擦/偏心）属于隐藏环境参数，只在这里应用；控制器永远读不到。
每次试次重建 MjModel，避免任何跨试次状态泄漏。
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from . import spec as sp
from typing import Any

DOG_JOINTS = [f"{leg}_{j}_joint" for leg in ("fl", "fr", "hl", "hr")
              for j in ("hipx", "hipy", "knee", "wheel")]
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["joint7", "joint8"]
# gym 调校站姿：knee 2.4→2.1（自由底座+臂平衡下平衡高度 ~0.249、pitch≈0），
# 接近官方策略站立高度 0.26（S1 验证）。抬高使抬升段腕部倾角从 16° 降到 ~8°，
# 避免抓持中腕部大角度旋转导致杯-垫摩擦扫滑。来源=仿真调校，非官方值。
DOG_IDLE = np.array([.08, -1.2, 2.1, 0., -.08, -1.2, 2.1, 0.,
                     .08, 1.2, -2.1, 0., -.08, 1.2, -2.1, 0.])


class Scene:
    """一个已构建的 MuJoCo 仿真场景。reset() 使其可复现地回到初始状态。"""

    def __init__(self, task: sp.TaskSpec):
        self.task = task
        self.xml_path = self._build_xml()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = task.control.dt
        # MuJoCo 文档：抓取类接触在默认 impratio=1 下摩擦约束会产生软性蠕滑
        # （物体以 ~cm/s 滑出指垫），提高 impratio 显著抑制。
        self.model.opt.impratio = float(getattr(task.control, "impratio", 20.0))
        # noslip 后处理求解器：消除摩擦约束的残余蠕滑（默认 0=关闭）
        self.model.opt.noslip_iterations = int(getattr(task.control, "noslip_iterations", 2))
        self.model.opt.noslip_tolerance = 1e-6
        self.data = mujoco.MjData(self.model)
        self._idx()
        from .robot import Robot
        self.robot = Robot(self)
        self.instance_ids = list(self.obj_jadr)

    # ------------------------------------------------------------- build
    # 指垫碰撞代理：放在指板靠指尖的下段（body y ∈ [-0.040, 0.002]）。
    # 完整指板 76mm 从 ee-0.036 延伸到 ee+0.040，上端越出杯顶沿 2.7cm，
    # 在杯口边缘产生向下楔力 ~5N 把杯子压在桌上——真实接触抓取失败的根因。
    # 42mm 垫 + z_off=0.013 → 垫面只覆盖杯身直段（local: x=垫宽 y=沿爪轴 z=闭合向）。
    _PAD_GEOM = ('<geom name="{name}" type="box" size="0.026 0.021 0.0105" '
                 'pos="0 -0.019 0.0135" friction="2.0 0.005 0.0001" '
                 'condim="3" priority="1" />')

    def _patched_piper_xml(self, piper_dir: Path, work_dir: Path) -> Path:
        """生成带指垫碰撞代理的 S10_Piper XML 副本。

        官方手指碰撞 = 网格凸包，指垫底缘是锐边：滑动接触中接触帧逐步抖动，
        摩擦传力退化（实测指垫上滑时垂直摩擦 ~0）。按 PRD F2.2 用盒子代理
        替换指垫碰撞，网格保留为视觉。

        产物写入 work_dir（运行隔离临时目录，PRD §10.1）：
        include 规则要求 patched XML 与 meshdir 相对可达——MuJoCo 的
        include file 相对【声明文件】目录解析，故 patched XML 必须与
        S10_Piper.xml 同目录或用绝对路径 include。这里用绝对路径 include
        （MuJoCo 支持），产物集中到 work_dir，不再写 SDK 源目录。"""
        src = (piper_dir / "S10_Piper.xml").read_text()
        # 夹爪基座网格凸包过大：腕朝下抓取时压在杯口上（实测 ~10N 把杯子按进
        # 桌面，指垫摩擦无法对抗）→ 关闭其碰撞，视觉保留。披露于 run 记录。
        src = src.replace(
            '<geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="gripper_base" />',
            '<geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="gripper_base" '
            'contype="0" conaffinity="0" />')
        # 指垫盒：局部几何 = 指板 x±0.028, y[-0.076,0], z[0,0.027]
        for fname, pad in (("link7", "pad_left"), ("link8", "pad_right")):
            # 关掉该指的网格碰撞（视觉保留），紧随其后插入盒代理
            src = src.replace(
                f'<geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="{fname}" />',
                f'<geom type="mesh" rgba="0.792157 0.819608 0.933333 1" mesh="{fname}" '
                f'contype="0" conaffinity="0" />\n    {self._PAD_GEOM.format(name=pad)}')
        out = work_dir / "patched_S10_Piper.xml"
        out.write_text(src)
        return out

    def _build_xml(self, work_dir: Path | None = None) -> Path:
        sdk = sp.sdk_root()
        piper_dir = sdk / "S10_description" / "S10_Piper"
        piper_xml = piper_dir / "S10_Piper.xml"
        if not piper_xml.is_file():
            raise FileNotFoundError(
                f"官方模型未找到: {piper_xml}\n"
                "请设置 S10_SDK_ROOT 指向 s10-piper-sdk/sim（README 有说明）")
        if work_dir is None:
            work_dir = self._work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        patched = self._patched_piper_xml(piper_dir, work_dir)
        # MuJoCo include 规则：meshdir 相对顶层 XML 目录解析，bare file 相对声明文件
        # 目录解析。meshdir 是相对路径 "./meshes/" → 顶层 XML 若不在 S10_Piper/ 内
        # 会解析失败。解决：顶层 XML 仍生成在 piper_dir（模型目录约束不可绕开），
        # 但文件名带运行唯一前缀；include 均用绝对路径指向 work_dir 的补丁副本，
        # 且每个运行持有自己的 work_dir + 唯一文件名，不互相覆盖（PRD §10.1）。
        import uuid
        stamp = uuid.uuid4().hex[:8]
        patched_abs = patched.resolve()
        tpl = f"""<mujoco model="gym_{self.task.name}">
  <include file="{patched_abs}" />
  <include file="{sp.ASSETS / 'demo_env.xml'}" />
  <equality>
    <weld name="grasp_weld" body1="mug" body2="link6" active="false" solref="0.005 1" solimp="0.9999 0.9999 0.0001 0.5 2"/>
  </equality>
</mujoco>"""
        out = piper_dir / f"_gen_gym_main_{stamp}.xml"
        out.write_text(tpl)
        self._gen_files = [out]
        return out

    def _work_dir(self) -> Path:
        """运行隔离工作目录：runs/_work/<uuid>/（PRD §10.1 隔离产物）。"""
        import uuid
        d = sp.REPO / "runs" / "_work" / uuid.uuid4().hex[:12]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------- weld
    def weld_arm(self) -> None:
        """以当前相对位姿激活 抓取 weld（披露的保持基底，PRD §8 fallback）。"""
        wid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        if wid < 0:
            return
        m, d = self.model, self.data
        b1 = m.body("mug").id
        b2 = m.body("link6").id
        p1, p2 = d.xpos[b1], d.xpos[b2]
        R1 = np.zeros(9)
        mujoco.mju_quat2Mat(R1, d.xquat[b1])
        R1 = R1.reshape(3, 3)
        rel_p = R1.T @ (p2 - p1)
        q1 = np.asarray(d.xquat[b1], float)
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

    def _idx(self) -> None:
        m, self.task = self.model, self.task
        def jadr(name):
            j = m.joint(name).id
            return m.jnt_qposadr[j], m.jnt_dofadr[j]
        self.dq, self.dv = zip(*[jadr(n) for n in DOG_JOINTS])
        self.aq, self.av = zip(*[jadr(n) for n in ARM_JOINTS])
        self.fq, self.fv = zip(*[jadr(n) for n in FINGER_JOINTS])
        self.dqa = [m.actuator(n).id for n in DOG_JOINTS]
        self.ara = [m.actuator(n).id for n in ARM_JOINTS]
        self.fra = [m.actuator(n).id for n in FINGER_JOINTS]
        self.fbq, self.fbv = jadr("floating_base")
        self.ee = m.site("EE_site").id
        self._body = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                      for n in ("mug", "tray", "carton", "link7", "link8")}
        # 夹指碰撞体 body id（观测与条件都可能用到）
        self.finger_body_ids = (self._body["link7"], self._body["link8"])
        self.obj_geoms: dict[str, list[int]] = {}
        for name in ("mug", "tray", "carton"):
            bid = self._body[name]
            if bid >= 0:
                self.obj_geoms[name] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        self.obj_jadr: dict[str, int] = {}
        self.obj_vadr: dict[str, int] = {}
        for name in ("mug", "tray"):
            bid = self._body[name]
            if bid >= 0 and m.body_jntnum[bid] > 0:
                jid = m.body_jntadr[bid]
                self.obj_jadr[name] = m.jnt_qposadr[jid]
                self.obj_vadr[name] = m.jnt_dofadr[jid]
        # 可调物理参数记录（条件注入用）
        self._mug_slide_geom = next(
            (g for g in self.obj_geoms.get("mug", []) if m.geom_contype[g] or m.geom_conaffinity[g]), None)

    # ------------------------------------------------------------- reset
    def reset(self, condition: str, seed: int) -> dict:
        """按条件+种子重置场景。返回应用了哪些隐藏参数（记入 run 记录）。"""
        mujoco.mj_resetData(self.model, self.data)
        m, d = self.model, self.data
        t = self.task
        start = [float(v) for v in t.robot.get("start_xy", [0.0, -0.636])]
        d.qpos[:3] = [start[0], start[1], 0.225]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for a, q in zip(self.dq, DOG_IDLE):
            d.qpos[a] = q
        for a in self.aq:
            d.qpos[a] = 0.0
        for a, v in zip(self.fq, (0.035, -0.035)):
            d.qpos[a] = v
        rng = np.random.default_rng(seed)
        jit = t.seed_jitter
        applied: dict[str, Any] = {}
        cond = sp.condition_params(t, condition)
        # 条件：摩擦（仅物体 geom；隐藏）
        if cond.get("mug_friction_slide") is not None and self._mug_slide_geom is not None:
            f = m.geom_friction[self._mug_slide_geom]
            m.geom_friction[self._mug_slide_geom] = np.array(
                [cond["mug_friction_slide"], f[1], f[2]])
            applied["mug_friction_slide"] = cond["mug_friction_slide"]
        # 条件：偏心初始摆放
        off = cond.get("mug_pos_offset_xy") or [0.0, 0.0]
        applied["mug_pos_offset_xy"] = list(off)
        for name, o in t.objects.items():
            if name not in self.obj_jadr:
                continue
            a = self.obj_jadr[name]
            pos = list(o.pos)
            if name == "mug":
                pos[0] += off[0] + float(rng.uniform(-jit.get("xy", 0.003), jit.get("xy", 0.003)))
                pos[1] += off[1] + float(rng.uniform(-jit.get("xy", 0.003), jit.get("xy", 0.003)))
                yaw = math.radians(float(rng.uniform(-jit.get("yaw_deg", 1.0), jit.get("yaw_deg", 1.0))))
                d.qpos[a:a + 3] = pos
                d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            else:
                d.qpos[a:a + 3] = pos
                d.qpos[a + 3:a + 7] = list(o.quat)
        mujoco.mj_forward(m, d)
        self.applied_hidden = applied
        return applied

    # ----------------------------------------------------------- helpers
    def obj_pos(self, name: str = "mug") -> np.ndarray:
        a = self.obj_jadr[name]
        return self.data.qpos[a:a + 3].copy()

    def obj_quat(self, name: str = "mug") -> np.ndarray:
        a = self.obj_jadr[name]
        return self.data.qpos[a + 3:a + 7].copy()

    def obj_vel(self, name: str = "mug") -> np.ndarray:
        a = self.obj_vadr[name]  # DOF 地址（评审1-3：qpos/dof 地址不同）
        return self.data.qvel[a:a + 6].copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee].copy()
