"""Run the saved room in MuJoCo and export the robot/object states for the room viewer."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .carry import CarryTaskExecutor
from .scene_v1 import SceneV1
from .scene_version import load_scene


class RoomTrace:
    def __init__(self, scene: SceneV1):
        self.scene = scene
        self.frames: list[list[float]] = []
        self.next_time = 0.0
        self.object_address = scene.obj_jadr[scene.grasp_instance]
        self.joints = [*scene.dq, *scene.aq, *scene.fq]

    def maybe_snap(self, data) -> None:
        if data.time < self.next_time:
            return
        s = self.scene
        self.frames.append([
            round(float(data.time), 3),
            *data.qpos[s.fbq:s.fbq + 7].round(5).tolist(),
            *data.qpos[self.joints].round(5).tolist(),
            *data.qpos[self.object_address:self.object_address + 7].round(5).tolist(),
        ])
        self.next_time = float(data.time) + 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--demo-bundle", type=Path, help="Export a successful run with the exact scene")
    args = parser.parse_args()
    scene = SceneV1(load_scene(args.scene_id))
    scene.reset(seed=0)
    trace = RoomTrace(scene)
    scene.frame_recorder = trace
    trace.maybe_snap(scene.data)
    run = CarryTaskExecutor(scene, verbose=False).run()
    trace.next_time = 0
    trace.maybe_snap(scene.data)
    record = {
        "run_id": "carry_" + uuid.uuid4().hex[:12],
        "scene_id": args.scene_id,
        "draft_version": scene.sv["draft_version"],
        "scene_version_id": (scene.sv.get("published") or {}).get("version_id"),
        "task_success": run["task_success"],
        "returned_with_cup": bool(run.get("returned_with_cup")),
        "demo_mode": (scene.sv.get("extra") or {}).get("demo_mode"),
        "failure_reason": run.get("failure_reason"),
        "failure_detail": run.get("failure_detail"),
        "events": sorted(run["events"], key=lambda event: event["t"]),
        "frames": trace.frames,
        "disclosure": "MuJoCo 实测回放。底座伺服辅助，双指确认接触后焊接辅助。",
    }
    out = Path(__file__).resolve().parent.parent / "runs" / "room_carry"
    out.mkdir(parents=True, exist_ok=True)
    (out / (record["run_id"] + ".json")).write_text(json.dumps(record, ensure_ascii=False))
    if args.demo_bundle and record["task_success"]:
        args.demo_bundle.parent.mkdir(parents=True, exist_ok=True)
        args.demo_bundle.write_text(json.dumps({"scene": scene.sv, "run": record},
                                               ensure_ascii=False, separators=(",", ":")) + "\n")
    print("RUN_SUMMARY " + json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
