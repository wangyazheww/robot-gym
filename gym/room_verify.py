#!/usr/bin/env python3
"""房间碰撞验证（v0.5 本轮第三步，PRD「同一 scene_version → MuJoCo 验证」）。

从房间场景版本（SceneVersion 草稿，gym/scene_version 保存）构建 SceneV1 MuJoCo 场景，
机器人于标定起点做短距受控行走，逐步记录：

- 起点穿透检查（reset 后 t=0 与任何代理的真实接触 = fail）
- 行走全程机器人与碰撞代理（box 代理 geom）的接触（mj_contactForce 法向力）
- 底座轨迹采样与最终位置（运动是否被代理挡住的判据）

行走方式：底座速度伺服（与 SceneV1.drive_to 同一实现，【伺服辅助】披露——
非官方策略轨道，仅用于碰撞验证的受控动作）。

用法（humanoid venv，含 mujoco）：
  spikes/humanoid/.venv/bin/python -m gym.room_verify \
      --scene-id room_wXXXX [--without-proxies] [--drive-m 0.6] [--seed 0]
输出 runs/room_verify/<run_id>/run.json；stdout 打印单行摘要 JSON（后端解析）。
"""
from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from pathlib import Path

import mujoco
import numpy as np

from .scene_version import load_scene
from .scene_v1 import SceneV1

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "room_verify"


def _proxy_geom_ids(m) -> set[int]:
    names = set()
    for p_ in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, p_)
        if n:
            names.add(n)
    return set()


def run_verify(scene: dict, with_proxies: bool = True, drive_m: float = 0.6,
               seed: int = 0) -> dict:
    sv = json.loads(json.dumps(scene))          # 深拷贝隔离
    proxy_ids = [p["proxy_id"] for p in (sv.get("proxies") or [])]
    if not with_proxies:
        sv["proxies"] = []
    sc = SceneV1(sv)
    m, d = sc.model, sc.data
    sc.reset(seed=seed)

    rs = sv.get("robot_start") or {"pos": [0, 0], "yaw_deg": 0.0}
    yaw = math.radians(rs.get("yaw_deg", 0.0))
    start_xy = np.array(rs["pos"], dtype=float)
    target_xy = start_xy + drive_m * np.array([math.cos(yaw), math.sin(yaw)])

    # 代理 geom 与非代理 geom 集合（floor 属于世界体，代理 geom 名 = proxy_id）
    proxy_geom = set()
    for pid in proxy_ids:
        try:
            proxy_geom.add(m.geom(pid).id)
        except KeyError:
            pass
    root = int(m.body_rootid[m.body("link7").id])
    robot_geoms = set()
    for g in range(m.ngeom):
        bid = m.geom_bodyid[g]
        if bid == 0 or int(m.body_rootid[bid]) != root:
            continue
        if g in proxy_geom:
            continue
        robot_geoms.add(g)

    def gather_contacts(t: float) -> list[dict]:
        out = []
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            proxy_hit = None
            if g1 in proxy_geom and g2 in robot_geoms:
                proxy_hit = g1
            elif g2 in proxy_geom and g1 in robot_geoms:
                proxy_hit = g2
            if proxy_hit is None:
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, f6)
            out.append({
                "proxy": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, proxy_hit),
                "normal_force_n": float(f6[0]),
                "t": round(t, 3),
            })
        return out

    # ① 起点穿透检查（reset 后 forward 的瞬时接触）
    mujoco.mj_forward(m, d)
    start_contacts = gather_contacts(0.0)

    # ② 短距受控行走（伺服辅助，drive_to 同款逻辑 + 逐步接触记录）
    dq, dv, dqa = sc.dq, sc.dv, sc.dqa
    fbq, fbv = sc.fbq, sc.fbv
    yaw0 = d.qpos[fbq + 3:fbq + 7].copy()
    t0 = d.time
    max_t = 25.0
    speed = 0.15
    tol = 0.03
    reached = False
    contacts: list[dict] = []
    traj: list[list[float]] = []
    next_sample = 0.0
    while d.time - t0 < max_t:
        d.qpos[fbq + 3:fbq + 7] = yaw0
        d.qvel[fbv + 3:fbv + 6] = 0.0
        cur = d.qpos[fbq:fbq + 2].copy()
        err = np.array(target_xy) - cur
        dist = float(np.linalg.norm(err))
        if dist < tol:
            reached = True
            break
        v_des = err / max(dist, 1e-6) * min(speed, dist * 2.0)
        d.qvel[fbv:fbv + 2] = v_des
        d.qvel[fbv + 5] = 0.0
        c = np.zeros(m.nu)
        q = d.qpos[list(dq)]
        dqv = d.qvel[list(dv)]
        c[dqa] = np.clip(40 * (sc.DOG_IDLE - q) - 2 * dqv, -20, 20)
        mujoco.mj_step(m, d)
        # 伺服每步覆写底座速度（mj_step 内积分后位置更新）——接触在本步之后有效
        contacts.extend(gather_contacts(d.time))
        if d.time >= next_sample:
            p = d.qpos[fbq:fbq + 3].copy()
            traj.append([round(d.time, 3), round(float(p[0]), 4),
                         round(float(p[1]), 4), round(float(p[2]), 4)])
            next_sample += 0.25
    # ③ 到站/超时后保持 1.0s 继续记录（静止顶压接触也要算）
    hold_until = d.time + 1.0
    while d.time < hold_until:
        c = np.zeros(m.nu)
        q = d.qpos[list(dq)]
        dqv = d.qvel[list(dv)]
        c[dqa] = np.clip(40 * (sc.DOG_IDLE - q) - 2 * dqv, -20, 20)
        d.qvel[fbv:fbv + 2] = 0.0
        mujoco.mj_step(m, d)
        contacts.extend(gather_contacts(d.time))
    final_pos = d.qpos[fbq:fbq + 3].copy()

    # 汇总接触（按代理聚合）
    agg: dict[str, dict] = {}
    for h in contacts:
        a = agg.setdefault(h["proxy"], {"proxy": h["proxy"], "n": 0,
                                        "max_force_n": 0.0, "first_t": h["t"], "last_t": h["t"]})
        a["n"] += 1
        a["max_force_n"] = max(a["max_force_n"], h["normal_force_n"])
        a["last_t"] = h["t"]
    summary = {
        "scene_id": sv["scene_id"],
        "draft_version": sv.get("draft_version"),
        "with_proxies": bool(with_proxies),
        "proxy_ids": proxy_ids,
        "drive_target_xy": [round(float(v), 3) for v in target_xy],
        "seed": seed,
        "reached": reached,
        "final_pos": [round(float(v), 4) for v in final_pos],
        "displacement_m": round(float(np.linalg.norm(final_pos[:2] - start_xy)), 4),
        "start_penetrated": bool(start_contacts),
        "start_contacts": start_contacts,
        "contacts": sorted(agg.values(), key=lambda a: -a["max_force_n"]),
        "contact_steps_total": len(contacts),
        "traj": traj,
        "sim_time_s": round(float(d.time), 2),
        "mujoco_version": mujoco.__version__,
        "disclosure": "底座速度伺服辅助行走（SceneV1.drive_to 同款，非官方策略轨道）",
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--without-proxies", action="store_true")
    ap.add_argument("--drive-m", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scene = load_scene(args.scene_id)
    t_wall0 = time.monotonic()
    summary = run_verify(scene, with_proxies=not args.without_proxies,
                         drive_m=args.drive_m, seed=args.seed)
    run_id = f"rv_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "wall_ms": int((time.monotonic() - t_wall0) * 1000),
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary}
    (out_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print("RUN_SUMMARY " + json.dumps(
        {k: record[k] for k in ("run_id", "with_proxies", "reached", "final_pos",
                                 "displacement_m", "start_penetrated", "contacts",
                                 "scene_id", "draft_version", "traj", "disclosure")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
