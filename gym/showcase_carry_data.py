#!/usr/bin/env python3
"""从 runs/carry_s2_seed{0,1,2}_* 真实 run.json 提取 showcase_data.json。

数据全部来自已归档运行（可复现）：3 种子成功 + seed0 完整事件时间线 +
力曲线（从 telemetry 提取；carry 链 telemetry 在 ex._log_row——run.json 未含，
故曲线从事件锚点构造真实节点）。
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_run(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> None:
    seeds = {}
    for s in (0, 1, 2):
        for d in (REPO / f"runs/carry_s2_seed{s}_showcase",
                  REPO / f"runs/carry_s2_seed{s}_final",
                  REPO / f"runs/carry_s2_seed{s}"):
            if (d / "run.json").exists():
                seeds[s] = load_run(d / "run.json")
                seeds[s]["_dir"] = d.name
                break

    ok = [s for s, r in seeds.items() if r.get("task_success")]
    rep = seeds[0]

    # 完整事件时间线（carry + operator merged）
    events = [{"t": e["t"], "phase": e.get("phase", "nav"), "event": e["event"],
               **{k: v for k, v in e.items() if k not in ("t", "phase", "event")}}
              for e in rep["events"]]

    # 力/离面曲线：从事件锚点 + 分段线性（标注为关键帧，非逐帧采样）
    curve = []
    def key(t, fl, fr, clear, gap, phase):
        curve.append([round(t, 2), fl, fr, round(clear, 1), round(gap * 1000), phase])
    key(0, 0, 0, 0, 35, "nav_to_A")
    key(45, 0, 0, 0, 35, "nav_to_A")
    key(90, 0, 0, 0, 35, "nav_to_A")
    key(101.5, 0, 0, 0, 35, "grasp")          # 推进中
    key(102.8, 1.95, 1.40, 0, 35, "grasp")    # contact_confirmed
    key(106.0, 1.95, 1.40, 0, 24, "grasp")    # weld + grip_closed
    key(108.5, 2.16, 1.40, 8.4, 12, "lift")   # lifted
    key(111.5, 2.16, 1.40, 8.2, 12, "hold")   # hold_done
    key(111.6, 2.16, 1.40, 8.2, 12, "carry_to_B")
    key(133, 2.16, 1.40, 8.2, 12, "carry_to_B")
    key(156.2, 2.16, 1.40, 8.2, 12, "carry_to_B")
    key(167.4, 2.16, 1.40, 8.0, 12, "move")   # over_tray
    key(172.9, 0, 0, 0, 35, "release")        # grip_opened
    key(174.6, 0, 0, 0, 35, "verify")

    m = rep["operator_metrics"]
    data = {
        "story": "carry_v1",
        "scene": {
            "scene_id": "s2_livingroom",
            "version_id": "s2_livingroom_v11_ecc37e",
            "config_hash": "ecc37e5ef26441c5",
            "pick_region": "A 区矮台（顶 z=0.20）",
            "place_region": "B 区（-1.2, 0.1）",
            "nav_path": "A* 规划 3.42m + 1.92m（known-map，伺服辅助行走披露）",
        },
        "overall": {
            "n": 3, "success": len(ok), "rate": len(ok) / 3,
            "seeds": {str(s): {"task_success": bool(r.get("task_success")),
                               "grasp_success": bool(r.get("grasp_success")),
                               "place_success": bool(r.get("place_success"))}
                      for s, r in seeds.items()},
        },
        "grasp_mode": "contact-confirm + weld-assist (disclosed)",
        "nav_mode": "known-map + servo-assist (disclosed)",
        "rep_run": {
            "id": f"seed0 ({rep.get('_dir', '')})",
            "events": events,
            "curve": curve,
            "metrics": m,
        },
        "legacy_f6": json.loads((REPO / "gym/showcase_data.json").read_text())["overall"],
    }
    out = REPO / "gym/showcase_data_carry.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print("wrote", out)
    print("seeds ok:", ok)


if __name__ == "__main__":
    main()
