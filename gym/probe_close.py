#!/usr/bin/env python3
"""闭合策略对比实验：找出能让杯子跟随指垫上升的闭合策略。

背景：抓取成立（双指接触、力正常）后，抬升时杯被"楔"在桌面
（垫摩擦向下饱和 ~11N + 重力 → 桌面法向 ~13N），杯不跟随。
本实验在真实场景测不同闭合策略的抬升跟随率。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco

from gym.spec import load_scene
from gym.scene import Scene
from gym.skills import TaskExecutor, GymFail
from gym.observe import finger_forces


def run_strategy(tag, close_plan, lift_cm=5.0, lift_s=2.0):
    """close_plan: [(target_gap, hold_s, kp), ...] 分段闭合"""
    task = load_scene("tabletop_pick_place")
    sc = Scene(task)
    sc.reset("normal", 0)
    m, d = sc.model, sc.data
    ex = TaskExecutor(sc, "fixed", verbose=False)
    try:
        for name, attr in ex.PHASES[:2]:   # settle + approach
            ex.phase = name
            getattr(ex, attr)()
    except GymFail as e:
        print(f"{tag}: approach fail {e.reason}")
        return
    mug0 = float(sc.obj_pos("mug")[2])
    contact_seen = False
    for target, hold_s, kp in close_plan:
        for step in range(int(hold_s / task.control.dt)):
            frac = min(1.0, step * task.control.dt / hold_s)
            gap = 0.035 + (target - 0.035) * frac
            c = np.zeros(m.nu)
            ex.robot.legs(c)
            ex.robot.arm(c, ex.q_grasp)
            ex.q_arm_cmd = ex.q_grasp
            ex.robot.grip_kp_override = kp
            c[sc.fra] = np.clip(kp * (np.array([gap, -gap]) - d.qpos[list(sc.fq)])
                                - 2 * d.qvel[list(sc.fv)], -10, 10)
            d.ctrl[:] = c
            mujoco.mj_step(m, d)
            ex.robot.hold_base()
        fl, fr, nc = finger_forces(sc)
        if fl > 0.3 and fr > 0.3:
            contact_seen = True
    if not contact_seen:
        print(f"{tag}: 无双指接触")
        return

    # 抬升
    cur_ee = d.site_xpos[sc.ee].copy()
    q_up, perr, oerr = ex.robot.ik6(cur_ee + np.array([0, 0, lift_cm]), ex.q_grasp)
    q0 = ex.q_grasp
    peak = mug0
    for step in range(int((lift_s + 1.5) / task.control.dt)):
        frac = min(1.0, step * task.control.dt / lift_s)
        # smoothstep 起步
        s = frac * frac * (3 - 2 * frac)
        qq = q0 + (q_up - q0) * s
        c = np.zeros(m.nu)
        ex.robot.legs(c)
        ex.robot.arm(c, qq)
        ex.q_arm_cmd = qq
        kp = close_plan[-1][2]
        ex.robot.grip_kp_override = kp
        c[sc.fra] = np.clip(kp * (np.array([close_plan[-1][0], -close_plan[-1][0]])
                                  - d.qpos[list(sc.fq)]) - 2 * d.qvel[list(sc.fv)], -10, 10)
        d.ctrl[:] = c
        mujoco.mj_step(m, d)
        ex.robot.hold_base()
        peak = max(peak, float(sc.obj_pos("mug")[2]))
    rise_mm = (peak - mug0) * 1000
    fl, fr, _ = finger_forces(sc)
    verdict = "FOLLOW" if rise_mm > lift_cm * 1000 * 0.7 else ("PARTIAL" if rise_mm > 10 else "SLIP")
    print(f"{tag:44s} 抬升跟随 {rise_mm:5.1f}mm/{lift_cm*1000:.0f}mm fL={fl:.1f} fR={fr:.1f} -> {verdict}")


if __name__ == "__main__":
    K = 300
    run_strategy("S1 当前: 直接闭合0.012@kp300", [(0.012, 1.5, K)])
    run_strategy("S2 两段: 0.020@kp300 1.5s -> 0.012", [(0.020, 1.5, K), (0.012, 1.0, K)])
    run_strategy("S3 慢闭合 3s", [(0.012, 3.0, K)])
    run_strategy("S4 软闭合 kp100 -> kp300", [(0.016, 2.0, 100), (0.012, 1.5, K)])
    run_strategy("S5 重夹: 0.012 -> 开0.022 -> 0.006",
                 [(0.012, 1.5, K), (0.022, 0.8, K), (0.006, 1.5, K)])
    run_strategy("S6 轻夹+先抬3mm+再收紧",
                 [(0.016, 1.5, K)], lift_cm=5.0, lift_s=2.0)
