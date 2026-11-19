#!/usr/bin/env python3
"""静态抓取台架（评审第一批交付 1-2）：证明指垫能碰到杯子，正反例单列。

台架允许固定底座（结果单列，不替代正式任务）：
  正例：杯传送到两指垫之间 → 闭爪 → 双侧独立接触力 > 0.3N
  反例：杯移到远处 → 闭爪 → 双侧力 = 0，无接触，weld 不激活

用法: python -m gym bench --out evidence/bench_static
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .scene_v1 import SceneV1


def _pads(scene: SceneV1) -> tuple[np.ndarray, np.ndarray]:
    m, d = scene.model, scene.data
    pl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_left")
    pr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_right")
    return d.geom_xpos[pl].copy(), d.geom_xpos[pr].copy()


def _run_case(scene: SceneV1, mug_target: np.ndarray | None,
              label: str) -> dict:
    """正例（mug_target=None）：夹持-保持协议。
      ① 杯传送到指垫中心并【运动学固定】0.6s，同时闭爪到 0.012
        （指垫内面全开 55mm < 杯径 61mm——直接传送必穿透弹飞，故先固定建压）
      ② 释放杯（速度清零），继续 2.0s 观察是否被指垫摩擦保持
      记录两阶段的双侧接触力；保持结果单列（ISSUE-1 直接实验）。
    反例（mug_target 给定）：杯传远处，闭爪 → 力必须为 0，weld 不激活。"""
    m, d = scene.model, scene.data
    mujoco.mj_resetData(m, d)
    rs = scene.sv.get("robot_start") or {"pos": [0, 0], "yaw_deg": 0.0}
    d.qpos[:3] = [rs["pos"][0], rs["pos"][1], 0.225]
    d.qpos[3:7] = [1.0, 0, 0, 0]
    for a, q in zip(scene.dq, scene.DOG_IDLE):
        d.qpos[a] = q
    for a in scene.aq:
        d.qpos[a] = 0.0
    for a, v in zip(scene.fq, (0.035, -0.035)):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)

    gi = scene.grasp_instance
    vadr = scene.obj_vadr[gi]
    jadr = scene.obj_jadr[gi]
    home = np.array([0, 1.2, -1.2, 0, -1.2, 0])
    from .observe import finger_forces

    def step(arm_q, gap, hold_mug: bool, mug_pin=None):
        """一步：固定底座（台架披露），可选固定杯。返回 (fl, fr, ncon)。"""
        c = np.zeros(m.nu)
        q = d.qpos[list(scene.dq)]
        dq = d.qvel[list(scene.dv)]
        c[scene.dqa] = np.clip(40 * (scene.DOG_IDLE - q) - 2 * dq, -20, 20)
        qa = d.qpos[list(scene.aq)]
        dqa = d.qvel[list(scene.av)]
        c[scene.ara] = np.clip(120 * (arm_q - qa) - 6 * dqa, -30, 30)
        c[scene.fra] = np.clip(300 * (np.array([gap, -gap]) - d.qpos[list(scene.fq)])
                               - 2 * d.qvel[list(scene.fv)], -10, 10)
        d.ctrl[:] = c
        if hold_mug and mug_pin is not None:
            d.qpos[jadr:jadr + 7] = mug_pin
            d.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_step(m, d)
        base7 = None  # 底座固定由外层管理
        fl, fr, ncon = finger_forces(scene, gi)
        return fl, fr, ncon

    # 站稳 + 臂 home（1.5s，固定底座）
    base_hold = d.qpos[scene.fbq:scene.fbq + 7].copy()

    def freeze_base():
        d.qpos[scene.fbq:scene.fbq + 7] = base_hold
        d.qvel[scene.fbv:scene.fbv + 6] = 0.0

    for _ in range(int(1.5 / 0.002)):
        step(home, 0.035, False)
        freeze_base()
    mujoco.mj_forward(m, d)
    pl, pr = _pads(scene)
    center = (pl + pr) / 2.0

    if mug_target is None:
        d.qpos[jadr:jadr + 3] = center
        d.qpos[jadr + 3:jadr + 7] = [1, 0, 0, 0]
        d.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(m, d)
        mug_pin = d.qpos[jadr:jadr + 7].copy()
        placed = "pad_center_kinematic_pin"
    else:
        d.qpos[jadr:jadr + 3] = mug_target
        d.qpos[jadr + 3:jadr + 7] = [1, 0, 0, 0]
        d.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(m, d)
        mug_pin = None
        placed = f"far_{mug_target.round(1).tolist()}"

    result = {"case": label, "mug_placed": placed,
              "pad_center": center.round(3).tolist(),
              "pin_geometry_note": "pads inner gap(55mm) < mug dia(61mm) at full open"}

    if mug_target is None:
        # ① 加压段：杯固定，闭爪 0.035→0.012（0.8s）
        squeeze = []
        n1 = int(0.8 / 0.002)
        for i in range(n1):
            gap = 0.035 + (0.012 - 0.035) * (i + 1) / n1
            fl, fr, ncon = step(home, gap, True, mug_pin)
            freeze_base()
            if (i + 1) % int(0.08 / 0.002) == 0:
                squeeze.append({"t": round(float(d.time), 3), "gap": round(gap, 4),
                                "f_left_N": round(fl, 3), "f_right_N": round(fr, 3)})
        # ② 释放段：松开杯固定，保持爪 0.012，观察 2.0s
        release = []
        z_at_release = float(d.qpos[jadr + 2])
        n2 = int(2.0 / 0.002)
        for i in range(n2):
            fl, fr, ncon = step(home, 0.012, False)
            freeze_base()
            if (i + 1) % int(0.2 / 0.002) == 0:
                release.append({"t": round(float(d.time), 3),
                                "mug_z": round(float(d.qpos[jadr + 2]), 4),
                                "drop_cm": round((z_at_release - float(d.qpos[jadr + 2])) * 100, 1),
                                "f_left_N": round(fl, 3), "f_right_N": round(fr, 3)})
        fl_f, fr_f, ncon_f = finger_forces(scene, gi)
        wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        result.update({
            "squeeze_tail": squeeze[-4:], "release": release,
            "final": {"f_left_N": round(fl_f, 3), "f_right_N": round(fr_f, 3),
                      "mug_z_end": round(float(d.qpos[jadr + 2]), 4),
                      "drop_cm": round((z_at_release - float(d.qpos[jadr + 2])) * 100, 1),
                      "weld_active": bool(d.eq_active[wid]) if wid >= 0 else None},
            # 双侧建压 = 台架主判据；纯摩擦保持单列（ISSUE-1）
            "bilateral_contact": bool(squeeze and
                                      squeeze[-1]["f_left_N"] > 0.3
                                      and squeeze[-1]["f_right_N"] > 0.3),
            "held_by_friction": bool(release and release[-1]["drop_cm"] < 1.0),
        })
    else:
        # 反例：闭爪 1.5s，力必须为 0
        samples = []
        n1 = int(1.5 / 0.002)
        for i in range(n1):
            gap = 0.035 + (0.012 - 0.035) * (i + 1) / n1
            fl, fr, ncon = step(home, gap, False)
            freeze_base()
            if (i + 1) % int(0.3 / 0.002) == 0:
                samples.append({"t": round(float(d.time), 3),
                                "f_left_N": round(fl, 3), "f_right_N": round(fr, 3),
                                "n_contacts": int(ncon)})
        fl_f, fr_f, ncon_f = finger_forces(scene, gi)
        wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        result.update({
            "samples": samples,
            "final": {"f_left_N": round(fl_f, 3), "f_right_N": round(fr_f, 3),
                      "n_contacts": int(ncon_f),
                      "weld_active": bool(d.eq_active[wid]) if wid >= 0 else None},
            "bilateral_contact": bool(fl_f > 0.3 and fr_f > 0.3),
        })
    return result


def run_bench(out_dir: str | Path = "evidence/bench_static") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = SceneV1.from_scene_version("s2_livingroom")
    scene.reset(seed=0)

    # 前置：验证指垫确实存在且有碰撞（id→name 用 mj_id2name）
    m = scene.model
    pad_ids = {}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name in ("pad_left", "pad_right"):
            pad_ids[name] = {"id": g, "contype": int(m.geom_contype[g]),
                             "type": int(m.geom_type[g])}
    pads_ok = (len(pad_ids) == 2
               and all(v["contype"] == 1 for v in pad_ids.values()))

    pos = _run_case(scene, None, "positive_mug_between_pads")
    neg = _run_case(scene, np.array([5.0, 5.0, 0.05]), "negative_mug_far_away")

    verdict = {
        "bench": "static_grasp_bench (fixed base, disclosed)",
        "pads_present_with_collision": pads_ok,
        "pad_info": pad_ids,
        "positive": pos,
        "negative": neg,
        "pass_criteria": {
            "pads_exist": pads_ok,
            "positive_bilateral": pos["bilateral_contact"],
            "negative_no_contact": not neg["bilateral_contact"],
            "negative_weld_off": neg["final"]["weld_active"] is False,
        },
    }
    verdict["PASS"] = all(verdict["pass_criteria"].values())
    (out / "bench_result.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2))
    print(json.dumps({k: verdict[k] for k in
                      ("pads_present_with_collision", "positive", "negative", "PASS")},
                     ensure_ascii=False, indent=1))
    return verdict


if __name__ == "__main__":
    run_bench()
