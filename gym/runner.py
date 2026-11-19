#!/usr/bin/env python3
"""单次试次运行器：构建场景 → 重置 → 执行任务 → 写 run 记录。

输出 runs/<batch_or_single>/<run_id>/ ：
  run.json       结果 + 事件 + 度量 + 配置哈希 + 隐藏参数（审计用）
  telemetry.jsonl  10Hz 时序（phase, 杯高, 开度指令, 臂关节）
  video.mp4      可选
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import mujoco

from . import spec as sp
from .record import Recorder
from .scene import Scene
from .skills import TaskExecutor

GYM_DIR = Path(__file__).resolve().parent
REPO = GYM_DIR.parent


def run_trial(scene_name: str, condition: str, mode: str, seed: int,
              out_dir: Path, video: bool = False, verbose: bool = True,
              run_tag: str = "") -> dict:
    assert mode in ("fixed", "feedback"), mode
    task = sp.load_scene(scene_name)
    scene = Scene(task)
    hidden = scene.reset(condition, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_tag or f"{scene_name}_{condition}_{mode}_s{seed}"
    rec = Recorder(scene, out_dir / "video.mp4") if video else None

    ex = TaskExecutor(scene, mode, recorder=rec, verbose=verbose)
    t0 = time.monotonic()
    result = ex.run()
    wall = time.monotonic() - t0
    if rec is not None:
        rec.close()

    record = {
        "run_id": run_id,
        "scene": scene_name,
        "scene_version": task.version,
        "config_hash": task.config_hash,
        "condition": condition,
        "mode": mode,
        "seed": seed,
        "grasp_mode": "contact-confirm + weld-assist (disclosed)",  # 接触确认真实；保持基底为披露 weld
        "base_mode": task.control.base_mode,
        "hidden_params_applied": hidden,               # 审计用：不进控制器
        "env": {"mujoco": mujoco.__version__, "python": None,
                "platform": __import__("sys").platform},
        "result": result,
        "wall_time_s": round(wall, 1),
    }
    (out_dir / "run.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2))
    with (out_dir / "telemetry.jsonl").open("w") as f:
        for row in ex.telemetry:
            f.write(json.dumps(row) + "\n")
    if verbose:
        m = result["metrics"]
        print(f"[run] {run_id}: grasp={result['grasp_success']} place={result['place_success']} "
              f"fail={result['failure_reason']} hold_min={m['hold_min_clear_cm']}cm "
              f"Fmax={m['max_finger_force_N']}N tighten={m['tightenings']} wall={wall:.0f}s",
              flush=True)
    return record
