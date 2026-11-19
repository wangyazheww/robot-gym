#!/usr/bin/env python3
"""真实 MuJoCo 搬运任务 demo 录制：跑完整 CarryTaskExecutor，25fps 离屏渲染出 MP4。

用法: .venv-sim/bin/python -m gym.record_demo [--seed 0] [--out evidence/demo]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mujoco

from .carry import CarryTaskExecutor
from .scene_v1 import SceneV1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="s2_livingroom")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="evidence/demo")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "mujoco_carry_demo.mp4"
    events_path = out_dir / "mujoco_carry_events.json"

    print("=== 构建 SceneV1 ===", flush=True)
    s = SceneV1.from_scene_version(args.scene)
    # 离屏渲染需要更大的 framebuffer（XML 默认 640x480）
    s.model.vis.global_.offwidth = max(args.width, 640)
    s.model.vis.global_.offheight = max(args.height, 480)
    # 生成的 env.xml 没有 camera —— 用 free camera 由代码控制
    CAM = None  # None → 默认 free camera，逐帧从机器人位姿推算
    print(f"    scene={args.scene} model nq={s.model.nq}", flush=True)
    info = s.reset(args.seed)
    print(f"    reset {info}", flush=True)

    # ---- 帧采集：patch mujoco.mj_step，每 1/fps 秒抓一帧 ----
    import imageio

    frames: list[np.ndarray] = []
    real_cam = mujoco.mj_step
    step_every = max(1, int(round((1.0 / args.fps) / s.model.opt.timestep)))  # 0.002→20 步/帧@25fps
    state = {"n": 0, "t0": None}

    renderer = mujoco.Renderer(s.model, height=args.height, width=args.width)

    def patched(m, d, nstep=1):
        real_cam(m, d, nstep)
        state["n"] += nstep
        if state["n"] >= step_every:
            state["n"] = 0
            # 跟随相机：底座后方上方看向前方，轻微追踪
            base = d.qpos[s.fbq:s.fbq + 2]
            yaw = d.qpos[s.fbq + 6]
            fx, fy = np.cos(yaw), np.sin(yaw)
            # 相机放在机器人后上方（沿 -前进方向 1.6m，高 1.0m）
            cam_pos = np.array([base[0] - fx * 1.7, base[1] - fy * 1.7, 1.05])
            look = np.array([base[0] + fx * 0.5, base[1] + fy * 0.5, 0.25])
            fwd = look - cam_pos
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, np.array([0, 0, 1.0]))
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            d.cam_xpos[:] = cam_pos
            d.cam_xmat[:] = np.stack([right, up, -fwd], axis=1).flatten()  # MuJoCo cam xmat: columns = x(right) y(up) z(-forward)
            renderer.update_scene(d)
            frames.append(renderer.render().copy())
            if len(frames) % 100 == 0:
                el = time.time() - state["t0"]
                print(f"    [rec] {len(frames)} 帧 @ sim t={d.time:6.1f}s "
                      f"(wall {el:.0f}s)", flush=True)

    mujoco.mj_step = patched

    print("=== 运行 CarryTaskExecutor（真实 MuJoCo 仿真） ===", flush=True)
    t0 = time.time()
    state["t0"] = t0
    ex = CarryTaskExecutor(s, mode="feedback", verbose=True)
    try:
        result = ex.run()
        ok = result.get("task_success", result.get("success", False))
    except Exception as e:
        ok = False
        result = {"task_success": False, "error": str(e)}
        print(f"    !! 运行失败: {e}", flush=True)
    finally:
        mujoco.mj_step = real_cam
        renderer.close()

    wall = time.time() - t0
    print(f"=== 仿真完成: success={ok} sim_t={s.data.time:.1f}s "
          f"frames={len(frames)} wall={wall:.0f}s ===", flush=True)

    # ---- 写 MP4 ----
    if frames:
        print(f"=== 编码 MP4 ({mp4_path}) ===", flush=True)
        with imageio.get_writer(str(mp4_path), fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1) as w:
            for f in frames:
                w.append_data(f)
        print(f"✅ 视频: {mp4_path} ({mp4_path.stat().st_size//1024}KB, "
              f"{len(frames)/args.fps:.1f}s)", flush=True)

    events_path.write_text(json.dumps({
        "scene": args.scene, "seed": args.seed, "success": bool(ok),
        "sim_time": float(s.data.time), "frames": len(frames),
        "result": result, "events": ex.events,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"✅ 事件: {events_path}", flush=True)


if __name__ == "__main__":
    main()
