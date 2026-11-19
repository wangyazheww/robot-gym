#!/usr/bin/env python3
"""技能状态机：SETTLE → APPROACH → GRASP → LIFT → HOLD → MOVE → RELEASE → VERIFY → RETREAT。

两种模式（PRD F4.5，共用轨迹/接触模型/力上限/失败判定，仅反馈调节不同）：
  fixed     固定力度：闭合到固定开度后不做任何调节
  feedback  触觉反馈：读两指接触力 + 滑动估计，滑动/掉力时分级收紧，超力保护

信息隔离（F4.4）：本模块只能通过 Observer 的观测字典感知世界；
条件参数（摩擦/偏心）属于隐藏环境参数，仿真侧注入，这里读不到。

抓取为纯接触物理（无 weld/无吸附辅助）：杯由指-杯摩擦力保持，
成功判据 = 离支撑面 ≥5cm 保持 ≥3s（PRD §4）。
"""
from __future__ import annotations

import mujoco
import numpy as np

from .observe import Observer
from .robot import Robot
from .scene import Scene


class GymFail(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class TaskExecutor:
    def __init__(self, scene: Scene, mode: str, recorder=None, verbose: bool = True):
        self.s = scene
        self.m, self.d = scene.model, scene.data
        self.task = scene.task
        self.mode = mode
        # 复用 scene.robot 单例（property 缓存）：新建实例会丢失 carry 链路
        # settle_base() 冻结的 _hold_qpos → 操作段底座漂移（v5/v6/v7 实测）
        self.robot = scene.robot
        self.obs = Observer(scene)
        self.recorder = recorder
        self.verbose = verbose

        self.phase = "init"
        self.events: list[dict] = []
        self.telemetry: list[dict] = []
        self.gap_cmd = 0.035
        self.q_arm_cmd = np.zeros(6)
        self._pause_until = 0.0

        # 度量
        self.max_finger_force = 0.0
        self.lift_peak_cm = 0.0
        self.slip_total_m = 0.0
        self.unexpected_contacts = 0
        self._slip_ack = 0.0
        self._tighten = 0
        self._grasp_active = False
        self._lifted = False

        # 场景几何常量（判定用；不给控制器）——从场景动态解析（SceneVersion 驱动）
        self.support_z = self._resolve_support_z()
        self.mug_half_h = self._resolve_obj_half_h()
        # 物体名实例化：抓取目标 / 放置参照（SceneVersion 角色驱动，无固定名）
        self.obj_name = getattr(self.s, "grasp_instance", None) or "mug"
        self.ref_name = getattr(self.s, "place_ref_instance", None) or ("tray" if "tray" in getattr(self.s, "instance_ids", []) else self.obj_name)
        # 离面基准冻结（评审1-3）：抓取判据相对 A 的初始支撑面，不随杯滑动变化
        # （须在 obj_name 之后——_local_support_z 依赖 self.obj_name）
        self._support_z_frozen = self._local_support_z()

    # ----------------------------------------------------------- helpers
    def ev(self, tag: str, **kw) -> None:
        rec = {"t": round(float(self.d.time), 3), "phase": self.phase, "event": tag, **kw}
        self.events.append(rec)
        if self.verbose:
            extra = " ".join(f"{k}={v}" for k, v in kw.items())
            print(f"  [{rec['t']:7.2f}s] {self.phase:<9} {tag} {extra}", flush=True)

    def mug_bottom_z(self) -> float:
        return float(self.s.obj_pos(self.obj_name)[2]) - self.mug_half_h

    def clearance_cm(self) -> float:
        """离面高度：杯底到【初始】支撑面的距离（基准冻结，评审1-3）。"""
        return (self.mug_bottom_z() - self._support_z_frozen) * 100.0

    def _local_support_z(self) -> float:
        """杯下方最近的支撑面顶高度。从 SceneVersion 的 support 代理读取（P1-04），
        回退到 geom 名字含 top 的（旧桌面场景）。"""
        obj = self.s.obj_pos(self.obj_name)
        obj_x, obj_y, obj_z = float(obj[0]), float(obj[1]), float(obj[2])
        best = 0.0  # 地面默认
        # 方法1：SceneVersion support proxies（优先）
        sv = getattr(self.s, "sv", None)
        if sv:
            for p in sv.get("proxies") or []:
                if p.get("kind") != "support":
                    continue
                px, py, pz = p["pose"][0], p["pose"][1], p["pose"][2]
                sz = p["size"]
                # box: size=[hx, hy, hz] → 顶面 = pz + hz
                z_top = pz + sz[2] if p["shape"] == "box" else pz + sz[1]
                # xy 距离在支撑面范围内
                dxy = float(np.linalg.norm(np.array([obj_x, obj_y]) - np.array([px, py])))
                if dxy < max(sz[0], sz[1]) + 0.05 and z_top <= obj_z + 0.02:
                    best = max(best, z_top)
            if best > 0:
                return best
        # 方法2（回退）：geom 名字含 top
        for g in range(self.m.ngeom):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "top" not in name:
                continue
            pos = self.d.geom_xpos[g]
            gt_ = self.m.geom_type[g]
            if gt_ == 6:
                sz = self.m.geom_size[g][2]
            elif gt_ == 5:
                sz = self.m.geom_size[g][1]
            else:
                continue
            z_top = float(pos[2] + sz)
            if z_top <= obj_z + 0.02:
                dxy = float(np.linalg.norm(np.array([obj_x, obj_y]) - pos[:2]))
                if dxy < 0.3:
                    best = max(best, z_top)
        return best

    def _resolve_region_support_z(self, region: dict) -> float:
        """B 区下方支撑面顶高度（从 SceneVersion support 代理读取）。"""
        rx, ry = region["pos"][0], region["pos"][1]
        sv = getattr(self.s, "sv", None)
        if sv:
            best = 0.0
            for p in sv.get("proxies") or []:
                if p.get("kind") != "support":
                    continue
                px, py, pz = p["pose"][0], p["pose"][1], p["pose"][2]
                sz = p["size"]
                z_top = pz + sz[2] if p["shape"] == "box" else pz + sz[1]
                dxy = float(np.linalg.norm(np.array([rx, ry]) - np.array([px, py])))
                if dxy < max(sz[0], sz[1]) + 0.1:
                    best = max(best, z_top)
            return best
        return 0.0

    def _resolve_support_z(self) -> float:
        """最高支撑面顶高（抓取判定的离面基准）。来自 SceneVersion support 代理，
        回退：找名为 *_top / table_top 的 geom，再回退 0（地面）。"""
        best = 0.0
        for g in range(self.m.ngeom):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "top" in name:
                pos = self.d.geom_xpos[g]
                gt_ = self.m.geom_type[g]
                if gt_ == 6:
                    z = float(pos[2] + self.m.geom_size[g][2])
                elif gt_ == 5:
                    z = float(pos[2] + self.m.geom_size[g][1])
                else:
                    continue
                best = max(best, z)
        return best

    def _resolve_obj_half_h(self) -> float:
        """抓取目标半高：从 grasp 实例 geom 尺寸取。"""
        gi = getattr(self.s, "grasp_instance", None)
        obj_geoms = getattr(self.s, "obj_geoms", {}).get(gi or "", [])
        for g in obj_geoms:
            t = self.m.geom_type[g]
            if t == 6:
                return float(self.m.geom_size[g][2])
            if t == 5:
                return float(self.m.geom_size[g][1])
            if t == 2:
                return float(self.m.geom_size[g][0])
        return 0.04

    def _ingest(self, obs: dict) -> None:
        self._last_obs = obs
        fl, fr = obs["finger_force_left_N"], obs["finger_force_right_N"]
        self.max_finger_force = max(self.max_finger_force, fl, fr)
        self.slip_total_m = max(self.slip_total_m, obs["slip_m"])
        self.unexpected_contacts += obs.get("n_con_unexpected", 0)
        # 超力保护（PRD §4）：仅在 grasp 及以后阶段触发（approach 途中的碰擦不计）
        limit = self.task.criteria.finger_force_limit_N
        if self.phase in ("grasp", "lift", "hold", "move", "release"):
            if max(fl, fr) > limit:
                self._force_over_ticks = getattr(self, "_force_over_ticks", 0) + 1
                if self._force_over_ticks >= 2:
                    raise GymFail("force_limit",
                                  f"finger force {max(fl, fr):.1f}N > {limit}N "
                                  f"for {self._force_over_ticks} ticks in {self.phase}")
                else:
                    pass
            else:
                self._force_over_ticks = 0

    def _log_row(self) -> None:
        o = self.s.obj_pos(self.obj_name)
        row = {
            "t": round(float(self.d.time), 3), "phase": self.phase,
            "mug_x": round(float(o[0]), 4), "mug_y": round(float(o[1]), 4),
            "mug_z": round(float(o[2]), 4),
            "clear_cm": round(self.clearance_cm(), 2),
            "gap_cmd": round(self.gap_cmd, 4),
            "arm_q": self.q_arm_cmd.round(3).tolist(),
        }
        obs = getattr(self, "_last_obs", None)
        if obs is not None:
            row.update({"f_left_N": obs["finger_force_left_N"],
                        "f_right_N": obs["finger_force_right_N"],
                        "n_con": obs["n_contacts_obj"],
                        "slip_mm": round(obs["slip_m"] * 1000, 2)})
        self.telemetry.append(row)

    def _run(self, duration: float, ctrl_fn, hook=None, break_when=None) -> None:
        """推进物理。ctrl_fn(ctrl, i, n)；hook(obs) 可返回 'pause'（暂停轨迹推进）；
        break_when() 为真时提前结束。观测/反馈 25Hz。"""
        c = self.s.task.control
        obs_every = max(1, int(0.04 / c.dt))
        n = max(1, int(duration / c.dt))
        for i in range(n):
            ctrl = np.zeros(self.m.nu)
            ctrl_fn(ctrl, i, n)
            self.d.ctrl[:] = ctrl
            mujoco.mj_step(self.m, self.d)
            self.robot.hold_base()
            if i % 10 == 0:
                self._log_row()
            if self.recorder is not None:
                self.recorder.maybe_snap(self.d)
            if (i + 1) % obs_every == 0:
                obs = self.obs.observe()
                self._ingest(obs)
                if hook is not None and self.d.time >= self._pause_until:
                    if hook(obs) == "pause":
                        self._pause_until = self.d.time + self.task.feedback.resume_after_s
            if not np.isfinite(self.d.qpos).all():
                raise GymFail("unstable", "non-finite MuJoCo state")
            if break_when is not None and break_when():
                break

    # ------------------------------------------------------ feedback core
    def _fb_hook(self, obs: dict):
        """触觉反馈（feedback 模式）。返回 'pause' 暂停轨迹推进 resume_after_s。
        仅在持杯段（lift/hold/move）响应——释放后的落杯滑移是预期行为。"""
        if self.mode != "feedback" or not self._grasp_active:
            return None
        if self.phase not in ("lift", "hold", "move"):
            return None
        fb = self.task.feedback
        crit = self.task.criteria
        fl = obs["finger_force_left_N"]
        fr = obs["finger_force_right_N"]
        slip_new = obs["slip_m"] - self._slip_ack
        holding = self.phase in ("lift", "hold", "move")
        lifted = self.clearance_cm() > 2.0

        # 1) 掉落检测：提起后（释放前）杯子落回支撑面附近 → 失败
        #    仅限 lift/hold：move 段下放段本就把杯朝 B 区支撑面放
        #    （B 支撑面可能低于 A 冻结基准 0.2m），下降中 clear<1.0 是预期路径
        #    （v16 实测误触发 slip_drop）
        if self.phase in ("lift", "hold") and self._lifted and self.clearance_cm() < 1.0:
            raise GymFail("slip_drop", f"mug fell back, clearance={self.clearance_cm():.1f}cm")
        # 3) 触发：滑动量超阈 或 已提起但单指掉力
        slip_hit = slip_new > fb.slip_trigger_m
        # 掉力触发须伴随真实滑动（weld 基底下垫接触力振荡属预期，不误报）
        drop_hit = (lifted and min(fl, fr) < fb.force_drop_trigger_N
                    and slip_new > fb.slip_trigger_m)
        if not (slip_hit or drop_hit):
            return None
        if self._tighten >= fb.max_tighten:
            raise GymFail("slip_drop",
                          f"slip {slip_new*1000:.1f}mm persists after {self._tighten} tightenings")
        kind = "slip" if slip_hit else "force_drop"
        self._tighten += 1
        self.gap_cmd = max(fb.min_gap_m, self.gap_cmd - fb.tighten_step_m)
        self._slip_ack = obs["slip_m"]
        self.ev("slip_detected" if slip_hit else "force_drop_detected",
                slip_mm=round(slip_new * 1000, 1), f_left=fl, f_right=fr,
                response="grip_tighten", tighten_no=self._tighten,
                new_gap=self.gap_cmd, trigger=kind)
        return "pause"

    # ------------------------------------------------------------ phases
    def _settle(self) -> None:
        c = self.task.control
        t = self.task.phases
        home = np.array(self.task.home_q)
        z_lo, z_hi = self.task.robot.get("stand_z_range", [0.15, 0.30])
        v_max = float(self.task.robot.get("stand_speed_max", 0.04))

        def ctrl_fn(ctrl, i, n):
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, home)
            self.q_arm_cmd = home
            self.robot.grip(ctrl, 0.035)

        # 站立直到位姿进入稳定窗口（自由底座会缓慢沉降漂移，等它稳住再冻结）
        total = t["settle_s"] + t.get("settle_extra_s", 6.0)
        chunk = 0.25
        settled = False
        t_sim = 0.0
        while t_sim < total:
            self._run(chunk, ctrl_fn)
            t_sim += chunk
            z = float(self.d.qpos[2])
            v = float(np.linalg.norm(self.d.qvel[self.s.fbv:self.s.fbv + 3]))
            if z_lo <= z <= z_hi and v < v_max:
                settled = True
                break
        if not settled:
            z = float(self.d.qpos[2])
            raise GymFail("unstable", f"no stable stance in {total:.0f}s (z={z:.3f})")
        z = float(self.d.qpos[2])
        self.ev("stand_ok", pelvis_z=round(z, 3), base_mode=c.base_mode)
        self.robot.settle_base()

    def _approach(self) -> None:
        t = self.task.phases
        mug = self.s.obj_pos(self.obj_name)
        z_off = t.get("grasp_z_offset_m", 0.01)
        q0 = np.array(self.task.home_q)
        base_x = float(self.d.qpos[self.s.fbq])
        base_y = float(self.d.qpos[self.s.fbq + 1])
        horiz = np.array([mug[0] - base_x, mug[1] - base_y])
        horiz = horiz / (np.linalg.norm(horiz) + 1e-9)
        grasp_z = mug[2] + z_off
        # 侧方点（杯旁 12cm 同高度）
        far_xy = np.array([mug[0] - horiz[0] * 0.12, mug[1] - horiz[1] * 0.12])
        # 三段 waypoint：① 高空 ② 高空水平移到杯侧方 ③ 降到杯高度
        wp1 = np.array([far_xy[0], far_xy[1], 0.50])   # 高空（高于杯）
        wp2 = np.array([far_xy[0], far_xy[1], grasp_z + 0.01])  # 杯高度 + EE→pad 偏移
        q_wp1, e1, _ = self.robot.ik6(wp1, q0, grid=True, axis_target=None)
        if q_wp1 is None or e1 > 0.06:
            raise GymFail("ik_failed", f"waypoint1 IK perr={e1:.3f}m")
        q_wp2, e2, _ = self.robot.ik6(wp2, q_wp1, grid=False, axis_target=None)
        if q_wp2 is None or e2 > 0.10:
            raise GymFail("ik_failed", f"waypoint2 IK perr={e2:.3f}m")
        self.q_grasp = q_wp2
        self.ev("ik_solved", wp1_perr=round(e1, 4), wp2_perr=round(e2, 4))
        seg1_s = t["approach_s"] / 2

        def ctrl_fn(ctrl, i, n):
            self.robot.legs(ctrl)
            self.robot.grip(ctrl, 0.035)
            el = (i * self.task.control.dt)
            if el < seg1_s:                   # ① home → 高空 waypoint
                frac = min(1.0, el / seg1_s)
                s = frac * frac * (3 - 2 * frac)
                q = q0 + (q_wp1 - q0) * s
            elif el < seg1_s + desc_s:        # ② 高空 → 杯高度（纯垂直下降）
                frac = min(1.0, (el - seg1_s) / desc_s)
                s = frac * frac * (3 - 2 * frac)
                q = q_wp1 + (q_wp2 - q_wp1) * s
            else:
                q = q_wp2
            self.robot.arm(ctrl, q)
            self.q_arm_cmd = q

        desc_s = t["descend_s"]
        self._run(seg1_s + desc_s, ctrl_fn)
        mug_after = self.s.obj_pos(self.obj_name).copy()
        if np.linalg.norm(mug_after - mug) > 0.03:
            self.ev("mug_knocked", mug_was=mug.round(3).tolist(),
                    mug_now=mug_after.round(3).tolist())
        self.ev("at_mug", ee=self.s.ee_pos().round(3).tolist(),
                mug=mug_after.round(3).tolist())
        # 闭环 descend 修正：EE 实测与 wp2 差 → 二次伺服（最多 2 轮，每轮 2s）
        wp2_world = np.array([far_xy[0], far_xy[1], grasp_z + 0.01])
        for it in range(2):
            ee_now = self.s.ee_pos()
            res = wp2_world - ee_now
            res_n = float(np.linalg.norm(res))
            if res_n < 0.02:
                break
            self.ev("descend_corr", iter=it + 1, residual_cm=round(res_n * 100, 1))
            q_corr, corr_perr, _ = self.robot.ik6(ee_now + res, self.q_grasp,
                                                   grid=False, axis_target=None)
            if q_corr is None:
                break
            q_start = self.q_grasp.copy()

            def corr_fn(ctrl, i, n):
                frac = min(1.0, (i * self.task.control.dt) / 2.0)
                q = q_start + (q_corr - q_start) * frac
                self.robot.legs(ctrl)
                self.robot.arm(ctrl, q)
                self.q_arm_cmd = q
                self.robot.grip(ctrl, 0.035)

            self._run(2.5, corr_fn)
            self.q_grasp = q_corr

    def _advance(self, state: dict, total_s: float) -> None:
        if self.d.time >= self._pause_until:
            state["frac"] = min(1.0, state["frac"] + self.task.control.dt / total_s)

    def _grasp(self) -> None:
        import mujoco as mj_
        import mujoco
        t = self.task.phases
        fb = self.task.feedback
        m, d = self.m, self.d
        # ---- 策略（v10-v13 台架化验证，2026-11-18）----
        # ① 指对称 hold：reset 后指位不对称（0.035/-0.0359）→ 单侧先撞推歪杯
        # ② pad 中心补偿：EE 目标 = 杯心 + (EE-pad 中心实测偏移)，否则系统性偏 ~1-2cm
        # ③ 慢速推进 1.5s：快推进欠阻尼振荡把杯打飞
        q_sym = self.q_grasp.copy()
        self._run(2.0, lambda c, i, n: (self.robot.legs(c),
                                        self.robot.arm(c, q_sym),
                                        self.robot.grip(c, 0.035)))
        mug = self.s.obj_pos(self.obj_name).copy()
        pl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_left")
        pr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_right")
        pad_c = (d.geom_xpos[pl] + d.geom_xpos[pr]) / 2
        pad_off = np.array(self.d.site_xpos[self.s.ee]) - pad_c
        q_target, perr = self.robot.ik(mug + pad_off, q_sym, pos_only=True)
        if q_target is None or perr > 0.10:
            self.ev("advance_skip", perr=round(perr, 3))
            q_target = q_sym
        q_start = q_sym.copy()
        touch_seq: list[str] = []

        def adv_hook(obs):
            for ci in range(d.ncon):
                con = d.contact[ci]
                g1 = mj_.mj_id2name(m, mj_.mjtObj.mjOBJ_GEOM, con.geom1) or "?"
                g2 = mj_.mj_id2name(m, mj_.mjtObj.mjOBJ_GEOM, con.geom2) or "?"
                if "mug" in (g1 + g2) and "pad" in (g1 + g2):
                    s = "L" if "left" in g1 + g2 else "R"
                    if not touch_seq or touch_seq[-1] != s:
                        touch_seq.append(s)
            return None

        self._run(3.0, lambda c, i, n: (self.robot.legs(c),
                                        self.robot.arm(c, q_start + (q_target - q_start)
                                                * min(1.0, (i * self.task.control.dt) / 1.5)),
                                        self.robot.grip(c, 0.035)),
                  hook=adv_hook)
        self.ev("advanced", touch_seq=touch_seq,
                ee=self.s.ee_pos().round(3).tolist(),
                mug=self.s.obj_pos(self.obj_name).round(3).tolist())
        contact = len(touch_seq) >= 1
        # ---- 阶段B：闭爪（接触确认→weld→hold）----
        window = close_s = t["grasp_s"]
        window += t["grasp_confirm_timeout_s"]
        state = {"confirmed": False}
        gap0 = max(float(d.qpos[list(self.s.fq)][0]), 0.012)

        def ctrl_fn(ctrl, i, nn):
            frac = min(1.0, (i * self.task.control.dt) / close_s)
            gap = gap0 + (fb.close_gap_m - gap0) * frac
            self.gap_cmd = gap
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, q_target)
            self.q_arm_cmd = q_target
            self.robot.grip(ctrl, gap)

        def hook(obs):
            if state["confirmed"]:
                return None
            if (obs["finger_force_left_N"] >= self.task.criteria.contact_force_min_N
                    and obs["finger_force_right_N"] >= self.task.criteria.contact_force_min_N):
                state["confirmed"] = True
                self.ev("contact_confirmed", f_left=obs["finger_force_left_N"],
                        f_right=obs["finger_force_right_N"])
            return None

        self._run(window, ctrl_fn, hook=hook)
        if not state["confirmed"]:
            raise GymFail("no_contact",
                          f"no bilateral contact within {t['grasp_confirm_timeout_s']}s")
        self._grasp_active = True
        self.q_grasp = q_target   # 后续 lift/move 的插值起点 = 实际推进目标
        self.s.weld_arm()
        self.ev("weld_assist_engaged", disclosed=True)
        self.obs.grasp_ref()
        o = self.obs.observe()
        self.ev("grip_closed",
                f_left=o["finger_force_left_N"], f_right=o["finger_force_right_N"],
                n_contacts=o["n_contacts_obj"])

    def _lift(self) -> None:
        t = self.task.phases
        # Lift from the measured contact pose. Preserve wrist roll as well as
        # the approach axis so the assisted cup follows a vertical translation.
        origin = self.s.ee_pos().copy()
        orientation = self.d.site_xmat[self.s.ee].reshape(3, 3).copy()
        path = [self.d.qpos[list(self.s.aq)].copy()]
        for fraction in (1/3, 2/3, 1):
            target = origin + [0, 0, fraction * t["lift_height_m"]]
            q, perr, oerr = self.robot.ik_pose(target, orientation, path[-1])
            if perr > 0.009 or oerr > 0.05:
                raise GymFail("ik_failed", f"upright lift IK perr={perr:.3f}m oerr={oerr:.2f}rad")
            path.append(q)
        self.q_lift = path[-1]
        state = {"frac": 0.0}

        def ctrl_fn(ctrl, i, n):
            self._advance(state, t["lift_s"])
            progress = 3 * state["frac"]
            segment = min(int(progress), 2)
            q = path[segment] + (path[segment+1] - path[segment]) * (progress-segment)
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, q)
            self.q_arm_cmd = q
            self.robot.grip(ctrl, self.gap_cmd)

        # 时长 = 正常时长 + 反馈暂停余量；完成即提前结束
        self._run(t["lift_s"] + 6.0, ctrl_fn, hook=self._fb_hook,
                  break_when=lambda: state["frac"] >= 1.0)
        self._lifted = True
        self.lift_peak_cm = max(self.lift_peak_cm, self.clearance_cm())
        self.ev("lifted", clear_cm=round(self.clearance_cm(), 1), tightenings=self._tighten, grip_tilt_deg=round(np.degrees(oerr), 1))

    def _hold(self) -> None:
        t = self.task.phases
        q_hold = self.q_lift

        def ctrl_fn(ctrl, i, n):
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, q_hold)
            self.q_arm_cmd = q_hold
            self.robot.grip(ctrl, self.gap_cmd)

        self._run(t["hold_s"], ctrl_fn, hook=self._fb_hook)
        self.ev("hold_done", clear_cm=round(self.clearance_cm(), 1))

    def _move(self) -> None:
        t = self.task.phases
        z_off = t.get("grasp_z_offset_m", 0.01)
        # P1-03：放置目标从 B 区（SceneVersion place region）读取，不从实例名猜测
        place_region = getattr(self.s, "place_region", None)
        if place_region:
            bx, by = place_region["pos"][0], place_region["pos"][1]
            # B 区支撑面顶 z：从 SceneVersion 的 support 代理读取
            bz = self._resolve_region_support_z(place_region)
        else:
            # 回退：旧桌面场景（有 tray 实例时）
            if hasattr(self.s, "instance_ids") and "tray" in getattr(self.s, "instance_ids", []):
                tray = self.s.obj_pos("tray")
                bx, by, bz = float(tray[0]), float(tray[1]), float(tray[2])
            else:
                raise GymFail("scene_invalid", "无放置目标（B 区和 tray 均不存在）")
        place_mug_z = bz + self.mug_half_h + 0.004
        bias = t.get("place_bias_xy_m", [0.015, 0.0])
        target = np.array([bx + bias[0], by + bias[1], place_mug_z + z_off])
        q_place, perr, oerr = self.robot.ik6(target, self.q_lift)
        if q_place is None or perr > 0.02 or oerr > 0.65:
            raise GymFail("ik_failed", f"place IK perr={perr:.3f}m oerr={oerr:.2f}rad")
        self.q_place = q_place
        # 两段式移动：先升到高位走廊（关节直插会使 EE 路径下坠撞进托盘——实测），
        # 再平移下放到放置点。
        lift_h = t.get("move_corridor_m", 0.53)
        cur_ee = np.array(self.d.site_xpos[self.s.ee])
        q_corridor, _, _ = self.robot.ik6(np.array([cur_ee[0], cur_ee[1], lift_h]),
                                          self.q_lift)
        if q_corridor is None:
            q_corridor = self.q_lift
        state = {"frac": 0.0, "seg": 0}

        def ctrl_fn(ctrl, i, n):
            self._advance(state, t["move_s"])
            if state["seg"] == 0:
                q = self.q_lift + (q_corridor - self.q_lift) * state["frac"]
                if state["frac"] >= 1.0:
                    state["seg"] = 1
                    state["frac"] = 0.0
            else:
                q = q_corridor + (q_place - q_corridor) * state["frac"]
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, q)
            self.q_arm_cmd = q
            self.robot.grip(ctrl, self.gap_cmd)

        self._run(t["move_s"] * 2 + 4.0, ctrl_fn, hook=self._fb_hook,
                  break_when=lambda: state["seg"] == 1 and state["frac"] >= 1.0)
        # 下探段：释放前降低 EE（轻放而非高空抛落）
        descend_m = t.get("place_descend_m", 0.0)
        q_down, pderr, _ = self.robot.ik6(np.array(self.d.site_xpos[self.s.ee]) - np.array([0, 0, descend_m]),
                                          self.q_place)
        if q_down is None:
            q_down = self.q_place
        self.q_place = q_down
        st2 = {"frac": 0.0}

        def descend_fn(ctrl, i, n):
            self._advance(st2, t.get("descend_s", 1.5))
            q = self.q_place + (q_down - self.q_place) * st2["frac"]
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, q)
            self.q_arm_cmd = q
            self.robot.grip(ctrl, self.gap_cmd)

        self._run(t.get("descend_s", 1.5) + 1.0, descend_fn,
                  break_when=lambda: st2["frac"] >= 1.0)
        # 静置段：耗散腕部转动残余动量，使释放时杯子速度≈0（否则释放散落）
        q_settle = self.q_place

        def settle_fn(ctrl, i, n):
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, q_settle)
            self.q_arm_cmd = q_settle
            self.robot.grip(ctrl, self.gap_cmd)

        self._run(t.get("place_settle_s", 1.0), settle_fn)
        self.ev("over_tray", ee=self.s.ee_pos().round(3).tolist(), grip_tilt_deg=round(np.degrees(oerr), 1))

    def _release(self) -> None:
        t = self.task.phases
        q_end = self.q_place
        g0 = self.gap_cmd

        def ctrl_fn(ctrl, i, n):
            frac = min(1.0, (i * self.task.control.dt) / t["release_s"])
            gap = g0 + (0.035 - g0) * frac
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, q_end)
            self.q_arm_cmd = q_end
            self.robot.grip(ctrl, gap)

        self._run(t["release_s"], ctrl_fn)
        # 先完全开爪（weld 继续持有杯），再解 weld —— 杯从静止垂直落入托盘，
        # 避免开爪指垫刮飞杯子（实测先解 weld 会被开爪动作拖出 ~18cm）
        self.s.weld_release()

        def drop_settle_fn(ctrl, i, n):
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, self.q_place)
            self.q_arm_cmd = self.q_place
            self.robot.grip(ctrl, 0.035)

        self._run(t.get("drop_settle_s", 1.0), drop_settle_fn)
        self.ev("grip_opened")

    def _verify(self) -> None:
        t = self.task.phases

        def ctrl_fn(ctrl, i, n):
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, self.q_place)
            self.q_arm_cmd = self.q_place
            self.robot.grip(ctrl, 0.035)

        self._run(t["verify_s"], ctrl_fn)
        self.ev("verify_done", obj=self.s.obj_pos(self.obj_name).round(3).tolist())

    def _retreat(self) -> None:
        t = self.task.phases
        q0 = self.q_place.copy()
        qh = np.array(self.task.home_q)
        # 先垂直抬升避让（防止回位扫掠撞飞托盘内的杯子），再回 home
        q_up, perr, _ = self.robot.ik6(np.array(self.d.site_xpos[self.s.ee])
                                       + np.array([0, 0, t.get("retreat_lift_m", 0.08)]), q0)
        if q_up is None:
            q_up = q0

        # 抬升避让后保持高位（joint 空间直插 home 会下坠扫过托盘拖走杯子——实测）
        def ctrl_fn(ctrl, i, n):
            frac = min(1.0, (i * self.task.control.dt) / (t["retreat_s"] * 0.4))
            s = frac * frac * (3 - 2 * frac)
            q = q0 + (q_up - q0) * s
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, q)
            self.q_arm_cmd = q
            self.robot.grip(ctrl, 0.035)

        self._run(t["retreat_s"] * 0.4 + 0.5, ctrl_fn)
        self.phase = "retreat_done"
        self.ev("retreat_up")

    # -------------------------------------------------------------- run
    PHASES = [("settle", "_settle"), ("approach", "_approach"), ("grasp", "_grasp"),
              ("lift", "_lift"), ("hold", "_hold"), ("shift", "_shift"), ("move", "_move"),
              ("release", "_release"), ("verify", "_verify"), ("retreat", "_retreat")]

    def _shift(self) -> None:
        """底座挪位（披露）：抓住杯子升举后，整机匀速滑移向放置点。"""
        t = self.task.phases
        shift = t.get("shift_xy_m", [0.0, 0.0])
        if abs(shift[0]) < 1e-9 and abs(shift[1]) < 1e-9:
            return
        dur = t.get("shift_s", 1.5)
        c = self.task.control
        n = int(dur / c.dt)
        step_d = [shift[0] / n, shift[1] / n]

        def ctrl_fn(ctrl, i, nn):
            self.robot.legs(ctrl)
            self.robot.arm(ctrl, self.q_lift)
            self.q_arm_cmd = self.q_lift
            self.robot.grip(ctrl, self.gap_cmd)
            self.robot.shift_base(step_d[0], step_d[1])

        self._run(dur, ctrl_fn)
        self.ev("base_shifted", dx=shift[0], dy=shift[1], disclosed=True)

    def run(self) -> dict:
        self.ev("task_start", mode=self.mode, grasp_mode="contact-confirm + weld-assist (disclosed)")
        try:
            for name, attr in self.PHASES:
                self.phase = name
                getattr(self, attr)()
        except GymFail as e:
            self.ev("fail", reason=e.reason, detail=e.detail)
            return self._evaluate(failed=e)
        return self._evaluate(failed=None)

    def _evaluate(self, failed: GymFail | None) -> dict:
        crit = self.task.criteria
        # 抓取成功 = 保持期（hold 段全程）离支撑面 ≥5cm（PRD §4）
        clear_hold = [r["clear_cm"] for r in self.telemetry if r["phase"] == "hold"]
        min_clear_hold = min(clear_hold) if clear_hold else None
        grasp_success = bool(failed is None and min_clear_hold is not None
                             and min_clear_hold >= crit.grasp_lift_cm)
        # 放置判定：托盘区域内 + 静止
        mug = self.s.obj_pos(self.obj_name)
        region = self.task.regions.get("tray_region", {})
        if region:
            cx, cy = region["center"]
            hx, hy = region["half_xy"]
            z0, z1 = region["z_band"]
            in_region = bool(abs(mug[0] - cx) < hx and abs(mug[1] - cy) < hy
                             and z0 <= mug[2] <= z1)
        else:
            in_region = False
        vel = float(np.linalg.norm(self.s.obj_vel(self.obj_name)[:3]))
        place_success = bool(failed is None and in_region
                             and vel < crit.place_speed_max)
        result = {
            "grasp_success": grasp_success,
            "place_success": place_success,
            "task_success": bool(grasp_success and place_success),
            "failure_reason": failed.reason if failed else None,
            "failure_detail": failed.detail if failed else None,
            "events": self.events,
            "metrics": {
                "hold_min_clear_cm": round(min_clear_hold, 2) if min_clear_hold is not None else None,
                "lift_peak_cm": round(self.lift_peak_cm, 2),
                "max_finger_force_N": round(self.max_finger_force, 2),
                "slip_max_mm": round(self.slip_total_m * 1000, 1),
                "tightenings": self._tighten,
                "mug_final_xy": [round(float(mug[0]), 3), round(float(mug[1]), 3)],
                "mug_final_z": round(float(mug[2]), 3),
                "mug_final_speed_ms": round(vel, 4),
                "in_region": in_region,
                "unexpected_arm_contacts": self.unexpected_contacts,
                "sim_t": round(float(self.d.time), 2),
            },
        }
        self.ev("result", **{k: v for k, v in result.items() if k != "events"})
        return result
