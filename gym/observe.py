#!/usr/bin/env python3
"""观测字典（PRD F4.4 信息隔离）：控制器唯一可见的信息通道。

只包含：两指法向接触力、接触数、物体相对爪的滑动量（ideal-pose-assisted，披露）、
EE/物体位姿、当前开度。绝不包含摩擦设定、质量、条件名等隐藏参数。
"""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import Scene


def quat2mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def finger_forces(scene: Scene, obj: str = "mug") -> tuple[float, float, int]:
    """(左指力N, 右指力N, 物体接触数)。mj_contactForce 法向分量按指分别求和。

    link7 = 左指，link8 = 右指（S10_Piper.xml 定义）。"""
    m, d = scene.model, scene.data
    obj_bodies = {m.geom_bodyid[g] for g in scene.obj_geoms.get(obj, [])}
    left_id, right_id = scene.finger_body_ids
    f_left = f_right = 0.0
    n_con = 0
    fc = np.zeros(6)
    for k in range(d.ncon):
        c = d.contact[k]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in obj_bodies or b2 in obj_bodies:
            n_con += 1
        finger = None
        if b1 in scene.finger_body_ids and b2 in obj_bodies:
            finger = b1
        elif b2 in scene.finger_body_ids and b1 in obj_bodies:
            finger = b2
        if finger is None:
            continue
        mujoco.mj_contactForce(m, d, k, fc)
        f = float(abs(fc[0]))
        if finger == left_id:
            f_left += f
        else:
            f_right += f
    return f_left, f_right, n_con


class Observer:
    """每次试次一个实例。grasp_ref() 在抓取确认时记录滑动基准。"""

    def __init__(self, scene: Scene, obj: str | None = None):
        self.s = scene
        # 角色驱动：抓取目标实例名从场景解析（SceneVersion），旧场景回退 'mug'
        self.obj = obj or getattr(scene, "grasp_instance", None) or "mug"
        self._ref_p: np.ndarray | None = None
        m = scene.model
        # 意外接触统计：臂身（link1..6）× 家具/障碍 geom。
        # 代理 geom 从 SceneVersion 派生（scene_v1），旧桌面场景从固定名回退。
        self.arm_body_ids = {m.body(f"link{i}").id for i in range(1, 7)}
        proxy_names = set()
        sv = getattr(scene, "sv", None)
        if sv:
            proxy_names = {p["proxy_id"] for p in sv.get("proxies") or []
                           if p.get("kind") in ("furniture", "wall", "obstacle")}
        self.obstacle_geom_ids = set()
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name in proxy_names:
                self.obstacle_geom_ids.add(g)

    def grasp_ref(self) -> None:
        self._ref_p = self._rel_p()

    def _rel_p(self) -> np.ndarray:
        d = self.s.data
        bid = self.s.model.body("link6").id
        R6 = quat2mat(d.xquat[bid])
        return R6.T @ (self.s.obj_pos(self.obj) - d.xpos[bid])

    def observe(self) -> dict:
        fl, fr, ncon = finger_forces(self.s, self.obj)
        m, d = self.s.model, self.s.data
        n_unexpected = 0
        for k in range(d.ncon):
            c = d.contact[k]
            if c.geom1 in self.obstacle_geom_ids or c.geom2 in self.obstacle_geom_ids:
                other = c.geom2 if c.geom1 in self.obstacle_geom_ids else c.geom1
                if m.geom_bodyid[other] in self.arm_body_ids:
                    n_unexpected += 1
        obs = {
            "t": round(float(self.s.data.time), 3),
            "finger_force_left_N": round(fl, 3),
            "finger_force_right_N": round(fr, 3),
            "n_contacts_obj": int(ncon),
            "n_con_unexpected": int(n_unexpected),
            "ee_pos": self.s.ee_pos().round(4).tolist(),
            "obj_pos": self.s.obj_pos(self.obj).round(4).tolist(),
            "slip_m": 0.0,
        }
        if self._ref_p is not None:
            obs["slip_m"] = round(float(np.linalg.norm(self._rel_p() - self._ref_p)), 4)
        return obs
