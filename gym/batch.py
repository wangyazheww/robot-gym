#!/usr/bin/env python3
"""配对实验协议（PRD F6）：3 条件 × N 对 × 2 模式，同种子配对。

每对内 fixed 与 feedback 共用同一种子（同初始抖动）、同轨迹、同接触模型、
同失败判定，仅反馈调节不同。全部失败样本保留并计入汇总。
输出：runs/<batch_id>/report.json + summary.csv + 各 run 子目录。
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from . import spec as sp
from .runner import run_trial

REPO = Path(__file__).resolve().parent.parent


def run_batch(scene_name: str, pairs: int = 5, conditions: list[str] | None = None,
              base_seed: int = 100, batch_id: str | None = None,
              video_first_pair: bool = True, verbose: bool = True) -> dict:
    task = sp.load_scene(scene_name)
    conditions = conditions or list(task.conditions.keys())
    batch_id = batch_id or time.strftime(f"{scene_name}_%Y%m%d_%H%M%S")
    out = REPO / "runs" / batch_id
    out.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    total = len(conditions) * pairs * 2
    done = 0
    for cond in conditions:
        for i in range(pairs):
            seed = base_seed + i
            for j, mode in enumerate(("fixed", "feedback")):
                vid = video_first_pair and cond == conditions[0] and i == 0
                done += 1
                if verbose:
                    print(f"\n=== [{done}/{total}] {cond} seed={seed} mode={mode} ===", flush=True)
                try:
                    rec = run_trial(scene_name, cond, mode, seed, out / f"{cond}_{mode}_s{seed}",
                                    video=vid, verbose=verbose,
                                    run_tag=f"{cond}_{mode}_s{seed}")
                except Exception as e:  # 单试次崩溃不拖垮整批
                    rec = {"run_id": f"{cond}_{mode}_s{seed}", "condition": cond,
                           "mode": mode, "seed": seed,
                           "result": {"grasp_success": False, "place_success": False,
                                      "task_success": False, "failure_reason": "crash",
                                      "failure_detail": repr(e)[:300], "events": [],
                                      "metrics": {}}}
                    print(f"[batch] CRASH {e!r}", flush=True)
                runs.append(rec)

    report = summarize(scene_name, task.config_hash, pairs, conditions, base_seed, runs)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_csv(out / "summary.csv", runs)
    if verbose:
        print(f"\n[batch] {batch_id}: report -> {out/'report.json'}", flush=True)
    return report


def summarize(scene_name, config_hash, pairs, conditions, base_seed, runs) -> dict:
    by_key: dict[tuple, list] = {}
    for r in runs:
        by_key.setdefault((r["condition"], r["mode"]), []).append(r)

    def stats(rows):
        grasp = sum(1 for r in rows if r["result"]["grasp_success"])
        place = sum(1 for r in rows if r["result"]["place_success"])
        task = sum(1 for r in rows if r["result"]["task_success"])
        fails: dict[str, int] = {}
        fmax, slip, tighten = [], [], []
        for r in rows:
            reason = r["result"].get("failure_reason")
            if reason:
                fails[reason] = fails.get(reason, 0) + 1
            m = r["result"]["metrics"]
            if r["result"]["grasp_success"]:
                fmax.append(m.get("max_finger_force_N") or 0)
                slip.append(m.get("slip_max_mm") or 0)
                tighten.append(m.get("tightenings") or 0)
        return {
            "n": len(rows), "grasp_success": grasp, "place_success": place,
            "task_success": task,
            "grasp_rate": round(grasp / len(rows), 3) if rows else None,
            "task_rate": round(task / len(rows), 3) if rows else None,
            "failures": fails,
            "F_max_N_mean": round(sum(fmax) / len(fmax), 2) if fmax else None,
            "slip_max_mm_mean": round(sum(slip) / len(slip), 1) if slip else None,
            "tighten_mean": round(sum(tighten) / len(tighten), 1) if tighten else None,
        }

    cond_summary = {}
    for cond in conditions:
        fixed = stats(by_key.get((cond, "fixed"), []))
        fb = stats(by_key.get((cond, "feedback"), []))
        cond_summary[cond] = {"fixed": fixed, "feedback": fb,
                              "pairs": pairs}
    return {
        "protocol": {
            "name": "PRD v0.3 F6 paired comparison",
            "scene": scene_name,
            "config_hash": config_hash,
            "conditions": conditions,
            "pairs": pairs,
            "base_seed": base_seed,
            "modes": ["fixed", "feedback"],
            "trials_total": len(runs),
            "grasp_mode": "contact (no assist)",
        },
        "summary": cond_summary,
        "overall": {
            "fixed": stats([r for r in runs if r["mode"] == "fixed"]),
            "feedback": stats([r for r in runs if r["mode"] == "feedback"]),
        },
        "runs": [{"run_id": r["run_id"], "condition": r["condition"], "mode": r["mode"],
                  "seed": r["seed"], "grasp": r["result"]["grasp_success"],
                  "place": r["result"]["place_success"],
                  "failure": r["result"].get("failure_reason"),
                  "metrics": r["result"]["metrics"]} for r in runs],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_csv(path: Path, runs: list[dict]) -> None:
    cols = ["run_id", "condition", "mode", "seed", "grasp", "place", "task",
            "failure_reason", "hold_min_clear_cm", "lift_peak_cm",
            "max_finger_force_N", "slip_max_mm", "tightenings", "sim_t", "wall_time_s"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            res = r["result"]
            m = res.get("metrics", {})
            w.writerow([r["run_id"], r["condition"], r["mode"], r["seed"],
                        res["grasp_success"], res["place_success"], res["task_success"],
                        res.get("failure_reason"),
                        m.get("hold_min_clear_cm"), m.get("lift_peak_cm"),
                        m.get("max_finger_force_N"), m.get("slip_max_mm"),
                        m.get("tightenings"), m.get("sim_t"), r.get("wall_time_s")])
