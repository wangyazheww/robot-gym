#!/usr/bin/env python3
"""SceneVersion 驱动的完整搬运任务执行器（PRD v0.5 §8.2 技能链）。

validate_scene → plan_to_A → navigate_to_A → settle → approach → grasp → lift
→ verify_hold → stow_for_carry → plan_to_B_with_payload → carry_to_B → settle
→ approach_place → lower → release → retreat → verify_placement

复用 TaskExecutor 的操作技能（approach/grasp/lift/...），新增导航技能
（plan_path A* + drive_to 伺服行走，披露）。抓取保持基底沿用
contact-confirm + weld-assist（披露，PRD §8.4 双模式）。
"""
from __future__ import annotations

import numpy as np

from .scene_v1 import SceneV1
from .scene_version import plan_path, path_length, proxy_contains_xy
from .skills import GymFail, TaskExecutor
from .observe import finger_forces


class CarryTaskExecutor:
    """A→B 跨位置搬运。A/B 来自 SceneVersion regions（kind=pick/place）。"""

    def __init__(self, scene: SceneV1, mode: str = "feedback", verbose: bool = True,
                 stop_offset: float = 0.40):
        self.s = scene
        self.mode = mode
        self.verbose = verbose
        self.stop_offset = stop_offset   # 停靠点距目标区域的距离（臂可达）
        self.events: list[dict] = []
        self.nav_grid = scene.sv.get("nav_grid")
        self._rb = scene.robot
        self.return_home = (scene.sv.get("extra") or {}).get("demo_mode") == "table_return"
        if not self.nav_grid:
            raise GymFail("scene_invalid", "SceneVersion 缺少 nav_grid（需重新发布）")
        self.pick = self._region("pick")
        self.place = self._region("place")
        if self.place is None:
            raise GymFail("scene_invalid", "缺少 place 区域（B）")
        if not self.s.place_region:
            pass

    def _region(self, kind: str) -> dict | None:
        for r in (self.s.sv.get("regions") or {}).values():
            if r.get("kind") == kind:
                return r
        return None

    def ev(self, tag: str, **kw) -> None:
        rec = {"t": round(float(self.s.data.time), 2), "event": tag, **kw}
        self.events.append(rec)
        if self.verbose:
            extra = " ".join(f"{k}={v}" for k, v in kw.items())
            print(f"  [{rec['t']:7.2f}s] {tag} {extra}", flush=True)

    # ------------------------------------------------------------ phases
    def _approach_point(self, region: dict) -> list[float]:
        """区域前沿停靠点：从区域中心沿 -x 方向偏 stop_offset（朝场地内侧）。

        停靠点必须在地图上可达；不可达时沿 y 扫描候选。"""
        cx, cy = region["pos"]
        parking = region.get("parking_xy")
        base = self.s.data.qpos[self.s.fbq:self.s.fbq + 2].tolist()
        if parking and plan_path(self.nav_grid, base, parking, footprint=0.22) is not None:
            self.ev("approach_point_selected", pos=parking)
            return parking
        # 从 stop_offset 到 1.5m 逐步缩短，找第一个可达点（逐点可达检查）
        from gym.scene_version import _point_in_poly
        cal = self.s.sv["calibration"]
        poly = cal["walkable_polygon"]
        occ = self.nav_grid["occupied"]
        w, h, res = self.nav_grid["w"], self.nav_grid["h"], self.nav_grid["resolution"]
        ox, oy = self.nav_grid["origin"]

        def cell_free(x, y, pad_cells=2):
            # pad=2（0.2m）对应 footprint 0.22；pad=3 曾把 0.5m 内候选全挡掉，
            # 逼停 0.6m 外→超臂展 0.584（ISSUE-5 第二层原因）
            i = int((x - ox) / res)
            j = int((y - oy) / res)
            for di in range(-pad_cells, pad_cells + 1):
                for dj in range(-pad_cells, pad_cells + 1):
                    ii, jj = i + di, j + dj
                    if ii < 0 or jj < 0 or ii >= w or jj >= h or occ[jj * w + ii]:
                        return False
            return True

        # A/B 都在桌面附近——机器人从地面接近，偏移方向朝场地中央（-y 方向）
        import math
        # 排除桌身/家具代理占据的区域（approach_point 必须在代理外面）
        furniture = [proxy for proxy in self.s.sv.get("proxies") or []
                     if proxy.get("kind") in ("furniture", "wall", "obstacle")]
        for off in [self.stop_offset, 0.5, 0.6, 0.8, 1.0]:
            for ang in range(0, 360, 20):
                dx = off * math.cos(math.radians(ang))
                dy = off * math.sin(math.radians(ang))
                cand = [cx + dx, cy + dy]
                # 检查候选不在任何家具代理 xy 范围内
                if any(proxy_contains_xy(proxy, cand, margin=0.3) for proxy in furniture):
                    continue
                if not cell_free(*cand):
                    continue
                # 找到可达候选
                p_ = plan_path(self.nav_grid, base, cand, footprint=0.22)
                if p_ is not None:
                    self.ev("approach_point_selected", pos=[round(cand[0],2), round(cand[1],2)])
                    return cand
                if _point_in_poly(cand, poly) and cell_free(*cand):
                    return cand
        raise GymFail("no_reachable_pose",
                      f"区域 {region['pos']} 周围无可停靠位（§8.3.2）")

    def _navigate(self, goal_xy: list[float], tag: str, carrying: bool = False,
                  arm_q=None, grip_gap=None) -> dict:
        """plan_path + drive_to（N02/N04/N06：路径显示、绕障、到达停稳）。"""
        # 行走前解除旧底座冻结：settle_at_B 的 hold_base 会把 base 拉回
        # 上一次 settle_base() 记录的位置（v14 实测：B 段走完后被拉回 A 点）
        self._rb._hold_qpos = None
        base = self.s.data.qpos[self.s.fbq:self.s.fbq + 2].tolist()
        path = plan_path(self.nav_grid, base, goal_xy, footprint=0.22)
        if path is None:
            raise GymFail("no_path", f"{tag}: 地图上无从 {base} 到 {goal_xy} 的路径")
        if self.return_home and len(path) > 2:
            # Keep every A* turn, but do not brake at every cell along a straight segment.
            steps = np.diff(path, axis=0)
            turns = np.any(np.abs(steps[1:] - steps[:-1]) > 1e-8, axis=1)
            path = [path[0], *(p for p, turn in zip(path[1:-1], turns) if turn), path[-1]]
        self.ev("planned", phase=tag, n=len(path), waypoints=path,
                length_m=round(path_length(path), 2))
        waypoints = path[1:] + ([goal_xy] if self.return_home else [])
        for k, wp in enumerate(waypoints, 1):
            final_tolerance = .012 if self.return_home and k == len(waypoints) else .05
            r = self.s.drive_to(wp, speed=0.15, tol=final_tolerance, max_t=30.0,
                                carrying=carrying, arm_q=arm_q,
                                grip_gap=grip_gap)
            if not r["reached"]:
                raise GymFail("unstable", f"{tag}: 路径点 {k} 未到达 {wp} (残差 {r['dist']}m)")
        return {"path_len": round(path_length(path), 2), "waypoints": len(path)}

    def _stability_check(self, tag: str, hold_s: float = 1.0,
                         arm_q=None, grip_gap: float | None = None) -> float:
        """N06 停稳判据：水平速度 <0.05 m/s 持续 hold_s。返回期间最大速度。
        arm_q/grip_gap 为 None → 空载 home + 开爪；给定 → 持物保持（P1-02）。
        停稳通过后 settle_base() 冻结底座——否则操作段 _hold_qpos=None，
        hold_base() 无效，臂推杯的反作用力会把整机推离 0.8m（v5 实测）。"""
        import mujoco
        m, d = self.s.model, self.s.data
        t0 = d.time
        vmax = 0.0
        if arm_q is None:
            arm_hold = np.array([0, 1.2, -1.2, 0, -1.2, 0])
            grip_hold = 0.035
        else:
            arm_hold = arm_q
            grip_hold = grip_gap if grip_gap is not None else 0.012
        while d.time - t0 < hold_s:
            c = np.zeros(m.nu)
            self._rb.legs(c)
            self._rb.arm(c, arm_hold)
            c[self.s.fra] = np.clip(300 * (np.array([grip_hold, -grip_hold])
                                    - d.qpos[list(self.s.fq)])
                                    - 2 * d.qvel[list(self.s.fv)], -10, 10)
            d.ctrl[:] = c
            mujoco.mj_step(m, d)
            if self.s.frame_recorder is not None:
                self.s.frame_recorder.maybe_snap(d)
            if hasattr(self, '_rb') and self._rb is not None:
                self._rb.hold_base()
            v = float(np.linalg.norm(d.qvel[self.s.fbv:self.s.fbv + 2]))
            vmax = max(vmax, v)
        if vmax >= 0.05:
            raise GymFail("unstable", f"{tag}: 停稳失败 max_v={vmax:.3f} m/s")
        self._rb.settle_base()   # 冻结底座（操作段 hold_base 依赖 _hold_qpos）
        return vmax


    def _operate_phase(self, ex: TaskExecutor, name: str) -> None:
        # P1-01：phase_id = 方法名去下划线（_hold → hold），判据筛选才能命中
        ex.phase = name.lstrip("_")
        getattr(ex, name)()

    def run(self) -> dict:
        import mujoco
        self.ev("carry_start", mode=self.mode,
                grasp_mode="contact-confirm + weld-assist (disclosed)",
                nav_mode="known-map + servo-assist (disclosed)")
        ex = None
        try:
            # --- 导航到 A（空载：臂 home + 开爪）---
            stop_a = self._approach_point(self.pick)
            self.ev("approach_point", phase="nav_to_A", pos=stop_a)
            nav_a = self._navigate(stop_a, "nav_to_A")
            self.ev("nav_to_A_done", **nav_a)
            self._stability_check("settle_at_A")   # 空载停稳
            self.ev("settled_at_A")

            # --- 操作技能链（复用 TaskExecutor；其 settle 跳过——已站好）---
            executor = TaskExecutor
            if self.return_home:
                from .upright_pick import UprightPickup
                executor = UprightPickup
            ex = executor(self.s, self.mode, recorder=self.s.frame_recorder, verbose=self.verbose)
            ex.phase = "approach"
            self._operate_phase(ex, "_approach")
            self._operate_phase(ex, "_grasp")
            self._operate_phase(ex, "_lift")
            self._operate_phase(ex, "_hold")
            grasp_ok = bool(min(
                [r["clear_cm"] for r in ex.telemetry if r["phase"] == "hold"] or [0]
            ) >= ex.task.criteria.grasp_lift_cm)
            self.ev("grasp_result", success=grasp_ok)
            if not grasp_ok:
                raise GymFail("slip_drop", "保持期离面不足")

            # --- 携物导航到 B（持物：保持 lift 关节目标 + 闭合爪距）---
            self.ev("stow_for_carry",
                    arm_q=ex.q_lift.round(2).tolist(),
                    grip_gap=round(float(ex.gap_cmd), 3),
                    note="lift 位姿即携物姿态（披露：未单独验证收拢姿态）")
            if self.return_home:
                from .upright_pick import cup_tilt_deg
                home = self.s.sv["robot_start"]["pos"]
                nav_home = self._navigate(home, "return_home", carrying=True,
                                          arm_q=ex.q_lift, grip_gap=ex.gap_cmd)
                self.ev("return_home_reached", **nav_home)
                self._stability_check("settle_at_home", arm_q=ex.q_lift, grip_gap=ex.gap_cmd)
                base = self.s.data.qpos[self.s.fbq:self.s.fbq+2]
                error = float(np.linalg.norm(base-np.array(home)))
                tilt = cup_tilt_deg(self.s)
                held = bool(self.s.data.eq_active.any()) and ex.clearance_cm() >= ex.task.criteria.grasp_lift_cm
                if error > .06 or tilt > 10 or not held:
                    raise GymFail("unstable", f"回原点偏差 {error:.3f} m，杯子倾角 {tilt:.1f}°，持杯 {held}")
                self.ev("returned_with_cup", error_m=round(error,4), cup_tilt_deg=round(tilt,2))
                self.events.extend(ex.events)
                return {"task_success": True, "grasp_success": True, "returned_with_cup": True,
                        "events": self.events}
            stop_b = self._approach_point(self.place)
            self.ev("approach_point", phase="carry_to_B", pos=stop_b)
            nav_b = self._navigate(stop_b, "carry_to_B",
                                   carrying=True,
                                   arm_q=ex.q_lift,
                                   grip_gap=ex.gap_cmd)
            self.ev("carry_to_B_done", **nav_b)
            # B 停稳：持物模式——保持 lift 臂姿与闭合爪（不回 home/开爪）
            self._stability_check("settle_at_B", arm_q=ex.q_lift,
                                  grip_gap=ex.gap_cmd)
            self.ev("settled_at_B")

            # --- 放置 ---
            self._operate_phase(ex, "_move")
            self._operate_phase(ex, "_release")
            self._operate_phase(ex, "_verify")
            self._operate_phase(ex, "_retreat")

            # --- 放置判定（§8.3.5：B 区内 + 受支撑 + 夹爪离开 + v<0.02 持续 2s）---
            gi = self.s.grasp_instance
            pos = self.s.obj_pos(gi)
            reg = self.place
            size = reg.get("size") or [0.3, 0.3]
            in_region = bool(abs(pos[0] - reg["pos"][0]) < size[0]
                         and abs(pos[1] - reg["pos"][1]) < size[1]
                         and -0.05 < pos[2] - 0.05 < 0.3)   # z: 地面放置带
            vel = float(np.linalg.norm(self.s.obj_vel(gi)[:3]))
            place_ok = bool(in_region and vel < 0.02)
            self.ev("place_result", success=place_ok, in_region=in_region,
                    v=round(vel, 3), pos=pos.round(3).tolist())
            task_ok = grasp_ok and place_ok
            self.ev("carry_result", success=task_ok)
            # 成功路径也合并操作技能链事件（showcase/审计需要完整因果日志）
            self.events.extend(ex.events)
            return {"task_success": task_ok, "grasp_success": grasp_ok,
                    "place_success": place_ok, "events": self.events,
                    "operator_metrics": ex._evaluate(None)["metrics"]}
        except GymFail as e:
            self.ev("fail", reason=e.reason, detail=e.detail)
            # 失败也要带上操作技能链的完整遥测（定位首败阶段用）
            if ex is not None:
                self.events.extend(ex.events)
            return {"task_success": False, "grasp_success": False,
                    "place_success": False, "failure_reason": e.reason,
                    "failure_detail": e.detail, "events": self.events}
        except Exception as e:   # 程序错误：记录堆栈与末帧状态
            import traceback
            tb = traceback.format_exc()
            state = {"base": self.s.data.qpos[self.s.fbq:self.s.fbq+3].round(3).tolist(),
                     "mug": self.s.obj_pos(self.s.grasp_instance or "mug_1").round(3).tolist(),
                     "phase": ex.phase if ex else "pre-executor"}
            self.ev("crash", error=repr(e), traceback_tail=tb.splitlines()[-3:],
                    last_state=state)
            if self.verbose:
                print(tb, flush=True)
            return {"task_success": False, "grasp_success": False,
                    "place_success": False, "failure_reason": "crash",
                    "failure_detail": repr(e), "traceback": tb,
                    "events": self.events}
