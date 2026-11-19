#!/usr/bin/env python3
"""机器人控制基元：腿 PD 站立、臂位置伺服、夹爪伺服、底座保持、数值 IK。

增益全部来自 TaskSpec.control（参数化，能力 #1）。
底座两种模式（诚实披露，二选一，写入 run 记录）：
  pd_stance       腿部 PD 物理保持站立（真物理，漂浮底座自由）
  kinematic_hold  站稳后底座运动学保持（长期试次的确定性选项，报告披露）
"""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import Scene, DOG_IDLE


def _world_to_base(p_world: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * p_world[0] + s * p_world[1], -s * p_world[0] + c * p_world[1],
                     p_world[2]])


def _quat_err(R_cur: np.ndarray, quat_des) -> np.ndarray:
    """朝向误差旋转向量（世界系）：R_err = R_des @ R_cur^T，返回 axis*angle。"""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R_cur, float).reshape(9))
    # Hamilton 乘法 q_rel = q_des ⊗ conj(q_cur) → R_des @ R_cur^T
    w1, x1, y1, z1 = quat_des
    w2, x2, y2, z2 = q[0], -q[1], -q[2], -q[3]  # conj(q_cur)
    q_rel = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    if q_rel[0] < 0:
        q_rel = -q_rel
    n = float(np.linalg.norm(q_rel[1:]))
    if n < 1e-9:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(n, q_rel[0]))
    if angle > np.pi:  # 等价短弧
        angle -= 2 * np.pi
    return q_rel[1:] / n * angle


# 夹爪轴目标方向：竖直朝下（俯视抓取，指垫横向闭合）。
# 杯为旋转对称体 → 只约束轴方向，绕轴 roll 自由（5D IK），避免全帧约束的 π 卷绕。
GRIP_AXIS = np.array([0.0, 0.0, -1.0])
WHEEL_INDICES = [3, 7, 11, 15]


class Robot:
    def __init__(self, scene: Scene):
        self.s = scene
        self.m, self.d = scene.model, scene.data
        c = scene.task.control
        self.base_mode = c.base_mode
        self.compensate_arm_gravity = (getattr(scene, "sv", {}).get("extra") or {}).get("demo_mode") == "table_return"
        self._hold_qpos: np.ndarray | None = None
        # IK 关节限位
        self._lim_lo = np.array([self.m.jnt_range[self.m.joint(f"joint{i}").id][0]
                                 for i in range(1, 7)])
        self._lim_hi = np.array([self.m.jnt_range[self.m.joint(f"joint{i}").id][1]
                                 for i in range(1, 7)])
        self._ee_base_offset = None  # 站稳后记录：EE 在底座系下位姿基准

    # ------------------------------------------------------------ legs
    def legs(self, ctrl: np.ndarray) -> None:
        c = self.s.task.control
        q = self.d.qpos[list(self.s.dq)]
        dq = self.d.qvel[list(self.s.dv)]
        error = DOG_IDLE - q
        error[WHEEL_INDICES] = 0.0
        ctrl[self.s.dqa] = np.clip(c.leg_kp * error - c.leg_kd * dq, -c.leg_clip, c.leg_clip)

    # ------------------------------------------------------------- arm
    def arm(self, ctrl: np.ndarray, q_target: np.ndarray) -> None:
        c = self.s.task.control
        q = self.d.qpos[list(self.s.aq)]
        dq = self.d.qvel[list(self.s.av)]
        torque = c.arm_kp * (q_target - q) - c.arm_kd * dq
        if self.compensate_arm_gravity:
            torque += self.d.qfrc_bias[list(self.s.av)]
        ctrl[self.s.ara] = np.clip(torque, -c.arm_clip, c.arm_clip)

    def grip(self, ctrl: np.ndarray, gap: float, clip: float | None = None) -> None:
        """gap = 每指行程目标（米）。joint7 → +gap, joint8 → -gap。
        clip 可临时软化伺服力限（闭合金防楔死），None = 配置值。"""
        c = self.s.task.control
        q = self.d.qpos[list(self.s.fq)]
        dq = self.d.qvel[list(self.s.fv)]
        target = np.array([gap, -gap])
        cap = c.grip_clip if clip is None else min(clip, c.grip_clip)
        ctrl[self.s.fra] = np.clip(c.grip_kp * (target - q) - c.grip_kd * dq, -cap, cap)
        self._grip_gap_cmd = float(gap)

    @property
    def grip_gap_cmd(self) -> float:
        return getattr(self, "_grip_gap_cmd", 0.035)

    # ------------------------------------------------------------ base
    def settle_base(self) -> None:
        """站稳时刻调用：kinematic_hold 模式记录底座位姿；pd_stance 不做任何事。"""
        if self.base_mode == "kinematic_hold":
            self._hold_qpos = self.d.qpos[self.s.fbq:self.s.fbq + 7].copy()

    def hold_base(self) -> None:
        if self._hold_qpos is not None:
            self.d.qpos[self.s.fbq:self.s.fbq + 7] = self._hold_qpos
            self.d.qvel[self.s.fbv:self.s.fbv + 6] = 0.0

    def base_pose(self) -> np.ndarray:
        return self.d.qpos[self.s.fbq:self.s.fbq + 7].copy()

    def shift_base(self, dx: float, dy: float = 0.0) -> None:
        """运动学底座挪位（披露）：四足站姿下整机平移，腿足随动。
        用于抓取点与放置点之间的小幅重定位，使两处均在臂的低倾角工作区。"""
        if self._hold_qpos is None:
            return
        self._hold_qpos[0] += dx
        self._hold_qpos[1] += dy

    # -------------------------------------------------------------- IK
    def _ik_pos_only(self, pos_target: np.ndarray, q_init: np.ndarray,
                     grid: bool = False) -> tuple[np.ndarray | None, float, float]:
        """纯位置 IK（无轴约束）。返回 (解, 位置误差, 0.0)。"""
        sol, perr = self.ik(pos_target, q_init, pos_only=True)
        return sol, perr, 0.0

    def _fd_jac_err(self, pos_target: np.ndarray, z_des: np.ndarray,
                    p0: np.ndarray, z0: np.ndarray) -> np.ndarray:
        """对实际误差函数直接做有限差分（位置误差向量 + 轴叉积误差向量）。

        不依赖任何推导公式：扰动每个关节，重新计算与主循环完全相同的误差，
        差分即真实雅可比。返回 6x6。调用方需保证当前 qpos 与 p0/z0 对应。"""
        m, d = self.m, self.d
        aq = list(self.s.aq)
        eps = 1e-5
        J = np.zeros((6, 6))
        for k in range(6):
            d.qpos[aq[k]] += eps
            mujoco.mj_kinematics(m, d)
            z_cur = d.site_xmat[self.s.ee].reshape(3, 3)[:, 2]
            J[:3, k] = (d.site_xpos[self.s.ee] - p0) / eps
            J[3:, k] = (np.cross(z_cur, z_des) - np.cross(z0, z_des)) / eps
            d.qpos[aq[k]] -= eps
            mujoco.mj_kinematics(m, d)
        return J

    def ik6(self, pos_target: np.ndarray, q_init: np.ndarray, grid: bool = False,
            axis_target: np.ndarray | None = None
            ) -> tuple[np.ndarray | None, float, float]:
        """5D 阻尼最小二乘 IK：3 位置 + 夹爪轴方向（绕轴 roll 自由）。

        误差 = [位置误差; z_cur × z_des]，雅可比对同一误差函数直接差分。
        返回 (解, 位置误差 m, 轴向角偏差 rad)。"""
        m, d = self.m, self.d
        pos_target = np.asarray(pos_target, float)
        if axis_target is None:
            return self._ik_pos_only(pos_target, q_init, grid=grid)
        z_des = np.asarray(axis_target, float)
        z_des = z_des / np.linalg.norm(z_des)
        aq = list(self.s.aq)
        seeds: list[np.ndarray] = [np.asarray(q_init, float)]
        if grid:
            for j1 in (-0.15, 0.0, 0.15):
                for j2 in (1.4, 1.8, 2.2, 2.6):
                    for j3 in (-2.6, -2.1, -1.6, -1.2):
                        for j5 in (-1.2, 1.2):
                            seeds.append(np.array([j1, j2, j3, 0.0, j5, 0.0]))
        best = None  # (score, q, perr, oerr)
        bak = d.qpos[aq].copy()

        def errors():
            R = d.site_xmat[self.s.ee].reshape(3, 3)
            z_cur = R[:, 2]
            p_err = pos_target - d.site_xpos[self.s.ee]
            o_err = np.cross(z_cur, z_des)
            oerr = float(np.arccos(np.clip(z_cur @ z_des, -1.0, 1.0)))
            return p_err, o_err, oerr, z_cur

        for q0 in seeds:
            q = np.clip(np.array(q0, float), self._lim_lo + 1e-3, self._lim_hi - 1e-3)
            for _ in range(100):
                d.qpos[aq] = q
                mujoco.mj_kinematics(m, d)
                p_err, o_err, oerr, z_cur = errors()
                pe = float(np.linalg.norm(p_err))
                if pe < 0.004 and oerr < 0.04:
                    break
                J = self._fd_jac_err(pos_target, z_des, d.site_xpos[self.s.ee].copy(), z_cur)
                e = np.r_[p_err, 1.5 * o_err]
                Jw = np.vstack([J[:3], 1.5 * J[3:]])
                dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + np.diag([2.5e-3] * 3 + [2.5e-2] * 3), e)
                q = np.clip(q + np.clip(dq, -0.12, 0.12), self._lim_lo + 1e-3, self._lim_hi - 1e-3)
            d.qpos[aq] = q
            mujoco.mj_kinematics(m, d)
            p_vec, _, oerr, _ = errors()
            pe = float(np.linalg.norm(p_vec))
            score = pe + 3.0 * oerr
            if best is None or score < best[0]:
                best = (score, q.copy(), pe, oerr)
            if pe < 0.02 and oerr < 0.08:
                break
        d.qpos[aq] = bak
        mujoco.mj_kinematics(m, d)
        if best is None:
            return None, float("inf"), float("inf")
        return best[1], best[2], best[3]

    def ik_pose(self, position: np.ndarray, orientation: np.ndarray,
                seed: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Solve a world-space hand pose, including the jaw roll omitted by ik6."""
        m, d = self.m, self.d
        aq = list(self.s.aq)
        av = list(self.s.av)
        backup = d.qpos[aq].copy()
        lo, hi = self._lim_lo + 1e-4, self._lim_hi - 1e-4
        q = np.clip(seed, lo, hi)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, orientation.ravel())
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        best = (float("inf"), q.copy(), float("inf"), float("inf"))
        try:
            for _ in range(150):
                d.qpos[aq] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                p_err = position - d.site_xpos[self.s.ee]
                r_err = _quat_err(d.site_xmat[self.s.ee].reshape(3, 3), quat)
                distance, angle = np.linalg.norm(p_err), np.linalg.norm(r_err)
                score = distance + .25 * angle
                if score < best[0]:
                    best = (score, q.copy(), float(distance), float(angle))
                if distance < .0004 and angle < .005:
                    break
                mujoco.mj_jacSite(m, d, jp, jr, self.s.ee)
                J = np.vstack((jp[:, av], .25 * jr[:, av]))
                change = J.T @ np.linalg.solve(J @ J.T + 8e-5 * np.eye(6), np.r_[p_err, .25*r_err])
                q = np.clip(q + np.clip(change, -.15, .15), lo, hi)
        finally:
            d.qpos[aq] = backup
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
        return best[1:]

    def ik(self, ee_target_world: np.ndarray, q_init: np.ndarray,
            pos_only: bool = True) -> tuple[np.ndarray | None, float]:
        """阻尼最小二乘数值 IK（确定性，网格重启）。返回 (解, 末端误差 m)。"""
        m, d = self.m, self.d
        base = d.qpos[self.s.fbq:self.s.fbq + 3]
        yaw = float(np.arctan2(2.0 * (d.qpos[self.s.fbq+3] * d.qpos[self.s.fbq+6]
                                      + d.qpos[self.s.fbq+4] * d.qpos[self.s.fbq+5]),
                               1.0 - 2.0 * (d.qpos[self.s.fbq+4]**2 + d.qpos[self.s.fbq+5]**2)))
        tgt = np.asarray(ee_target_world, float)
        rel = tgt - base
        c, s = np.cos(-yaw), np.sin(-yaw)
        t_rel = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
        best: tuple[float, np.ndarray] | None = None
        bak = d.qpos[list(self.s.aq)].copy()
        seeds = [np.asarray(q_init, float)]
        for j1 in (-0.2, 0.0, 0.2):
            for j2 in (1.4, 1.8, 2.2, 2.7):
                for j3 in (-2.6, -2.0, -1.4, -0.8):
                    seeds.append(np.array([j1, j2, j3, 0.0, 0.0, 0.0]))
        seen = set()
        for q0 in seeds:
            key = tuple(q0.round(3))
            if key in seen:
                continue
            seen.add(key)
            q = np.clip(np.array(q0, float), self._lim_lo + 1e-3, self._lim_hi - 1e-3)
            for _ in range(150):
                d.qpos[list(self.s.aq)] = q
                mujoco.mj_kinematics(m, d)
                p = d.site_xpos[self.s.ee] - base
                err = t_rel - _world_to_base(p, yaw)
                if np.linalg.norm(err) < 0.005:
                    break
                J = np.zeros((3, 6))
                eps = 1e-4
                for k in range(6):
                    d.qpos[list(self.s.aq)[k]] += eps
                    mujoco.mj_kinematics(m, d)
                    p2 = d.site_xpos[self.s.ee] - base
                    d.qpos[list(self.s.aq)[k]] -= eps
                    J[:, k] = (p2 - p) / eps
                dq = J.T @ np.linalg.solve(J @ J.T + 0.0025 * np.eye(3), err)
                if pos_only:
                    dq[3:] = 0.0
                q = np.clip(q + np.clip(dq, -0.1, 0.1), self._lim_lo + 1e-3, self._lim_hi - 1e-3)
            d.qpos[list(self.s.aq)] = q
            mujoco.mj_kinematics(m, d)
            ferr = float(np.linalg.norm(d.site_xpos[self.s.ee] - tgt))
            if best is None or ferr < best[0]:
                best = (ferr, q.copy())
            if ferr < 0.02:
                break
        d.qpos[list(self.s.aq)] = bak
        mujoco.mj_kinematics(m, d)
        return (best[1], best[0]) if best else (None, float("inf"))
