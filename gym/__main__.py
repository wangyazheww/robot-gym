#!/usr/bin/env python3
"""CLI：python -m gym run|batch|report|probe"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .batch import run_batch
from .runner import run_trial
from .spec import load_scene, sdk_root


def main(argv=None):
    p = argparse.ArgumentParser(prog="gym", description=__doc__)
    p.add_argument("--version", action="version", version=f"gym {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="单次试次")
    pr.add_argument("--scene", default="tabletop_pick_place")
    pr.add_argument("--condition", default="normal")
    pr.add_argument("--mode", default="feedback", choices=["fixed", "feedback"])
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--video", action="store_true")
    pr.add_argument("--out", default=None)
    pr.add_argument("--quiet", action="store_true")

    pb = sub.add_parser("batch", help="配对实验批量")
    pb.add_argument("--scene", default="tabletop_pick_place")
    pb.add_argument("--pairs", type=int, default=5)
    pb.add_argument("--conditions", default=None, help="逗号分隔，默认场景全部条件")
    pb.add_argument("--base-seed", type=int, default=100)
    pb.add_argument("--batch-id", default=None)
    pb.add_argument("--no-video", action="store_true")
    pb.add_argument("--quiet", action="store_true")

    psc = sub.add_parser("showcase", help="生成评委展示页（自包含单文件）")

    prp = sub.add_parser("report", help="生成自包含 HTML 报告")
    prp.add_argument("--batch", required=True)

    pinfo = sub.add_parser("info", help="显示场景与 SDK 解析结果")
    pinfo.add_argument("--scene", default="tabletop_pick_place")

    args = p.parse_args(argv)
    if args.cmd == "run":
        out = Path(args.out) if args.out else Path("runs") / "single"
        rec = run_trial(args.scene, args.condition, args.mode, args.seed, out,
                        video=args.video, verbose=not args.quiet)
        return 0 if rec["result"]["task_success"] else 1
    if args.cmd == "batch":
        conds = args.conditions.split(",") if args.conditions else None
        report = run_batch(args.scene, pairs=args.pairs, conditions=conds,
                           base_seed=args.base_seed, batch_id=args.batch_id,
                           video_first_pair=not args.no_video,
                           verbose=not args.quiet)
        ov = report["overall"]
        print(json.dumps({m: {k: ov[m][k] for k in ("n", "grasp_success", "task_rate",
                                                    "failures")}
                          for m in ("fixed", "feedback")}, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "showcase":
        from .showcase import build
        print(f"showcase -> {build()}")
        return 0
    if args.cmd == "report":
        from .report import render
        print(f"report -> {render(args.batch)}")
        return 0
    if args.cmd == "info":
        task = load_scene(args.scene)
        print(f"scene      : {task.name} v{task.version} (hash {task.config_hash})")
        print(f"conditions : {list(task.conditions.keys())}")
        print(f"base_mode  : {task.control.base_mode}")
        print(f"sdk_root   : {sdk_root()}  exists={sdk_root().exists()}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
