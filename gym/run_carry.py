#!/usr/bin/env python3
"""一键正式任务运行（评审第一批交付 1-4）。

入口：SceneV1.from_scene_version → reset(seed) → CarryTaskExecutor.run()
失败时保存：阶段、原因、异常堆栈（程序错误时）、最后一帧状态。

用法: python -m gym.run_carry --scene s2_livingroom --seed 0 \
        --out runs/carry_s2_seed0
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .carry import CarryTaskExecutor
from .scene_v1 import SceneV1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="s2_livingroom")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/carry_run")
    ap.add_argument("--mode", default="feedback")
    ap.add_argument("--stop-offset", type=float, default=0.40)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result: dict = {}
    scene = None
    try:
        scene = SceneV1.from_scene_version(args.scene)
        scene.reset(seed=args.seed)
        ex = CarryTaskExecutor(scene, mode=args.mode,
                               stop_offset=args.stop_offset)
        result = ex.run()
    except Exception as e:  # 场景构建/reset 本身崩掉也要留痕
        import traceback
        result = {"task_success": False, "failure_reason": "setup_crash",
                  "failure_detail": repr(e), "traceback": traceback.format_exc()}

    result.update({
        "entry": "gym.run_carry",
        "scene": args.scene,
        "seed": args.seed,
        "mode": args.mode,
        "stop_offset": args.stop_offset,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    (out / "run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str))

    # 最后一帧快照（无论成败）
    if scene is not None:
        d = scene.data
        frame = {
            "time": float(d.time),
            "base_pos": d.qpos[scene.fbq:scene.fbq + 3].round(4).tolist(),
            "mug_pos": scene.obj_pos(scene.grasp_instance).round(4).tolist(),
            "mug_vel": scene.obj_vel(scene.grasp_instance).round(4).tolist(),
        }
        (out / "last_frame.json").write_text(
            json.dumps(frame, ensure_ascii=False, indent=2))

    ok = result.get("task_success")
    print(f"[run_carry] scene={args.scene} seed={args.seed} "
          f"task_success={ok} reason={result.get('failure_reason', '-')}")
    if not ok:
        for ev in result.get("events", []):
            print(f"  {ev}")
    print(f"artifacts -> {out}/run.json, {out}/last_frame.json")


if __name__ == "__main__":
    main()
