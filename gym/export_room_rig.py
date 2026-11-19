"""Export the official MuJoCo joint hierarchy for the room viewer."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco

from .scene_v1 import SceneV1
from .scene_version import load_scene


def main() -> None:
    model = SceneV1(load_scene("s2_livingroom")).model
    root = model.body("base_link").id
    directory = Path(__file__).resolve().parent.parent / "frontend/public/models"
    entries = []
    for i in range(root, model.nbody):
        if model.body_rootid[i] != root:
            continue
        body = model.body(i)
        name = body.name
        meshes = [name] + (["gripper_base"] if name == "link6" else [])
        if not all((directory / f"{mesh}.glb").is_file() for mesh in meshes):
            raise FileNotFoundError(f"Missing visual mesh for {name}")
        row = {"name": name, "parent": None if i == root else model.body(model.body_parentid[i]).name,
               "position": body.pos.tolist(), "quaternion_wxyz": body.quat.tolist(), "meshes": meshes}
        if i != root and model.body_jntnum[i]:
            j = model.body_jntadr[i]
            row["joint"] = {"type": "slide" if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                            "axis": model.jnt_axis[j].tolist()}
        entries.append(row)
    (directory / "s10-piper-rig.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Exported {len(entries)} official robot links")


if __name__ == "__main__":
    main()
