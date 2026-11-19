"""Record a synchronized upright pickup clip and tactile samples from the S2 tea table.

Run with python -m gym.record_upright --out runs/upright_demo.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco

from .carry import CarryTaskExecutor
from .scene_v1 import SceneV1
from .upright_pick import UprightPickup, cup_tilt_deg


class RecordedPickup(UprightPickup):
    def __init__(self, scene, record_frame):
        super().__init__(scene, "feedback", verbose=False)
        self.record_frame = record_frame

    def _ingest(self, obs):
        super()._ingest(obs)
        self.record_frame({
            "sim_time_s": float(self.d.time),
            "phase": self.phase,
            "left_N": obs["finger_force_left_N"],
            "right_N": obs["finger_force_right_N"],
            "clearance_cm": self.clearance_cm(),
            "cup_tilt_deg": cup_tilt_deg(self.s),
        })


def record(out):
    scene = SceneV1.from_scene_version("s2_livingroom")
    scene.reset(seed=0)
    carry = CarryTaskExecutor(scene, verbose=False)
    carry._navigate(carry._approach_point(carry.pick), "nav_to_A")
    carry._stability_check("settle_at_A")
    cup = scene.obj_pos(scene.grasp_instance).copy()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = cup + [.013, 0, .053]
    camera.azimuth, camera.elevation, camera.distance = 27, -25, .40
    scene.model.vis.global_.offwidth = 840
    scene.model.vis.global_.offheight = 588
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    samples = []
    out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(scene.model, height=588, width=840) as renderer, imageio.get_writer(
        out / "pickup.mp4", fps=25, macro_block_size=1, codec="libx264", quality=9
    ) as video:
        def capture(sample):
            renderer.update_scene(scene.data, camera=camera, scene_option=options)
            video.append_data(renderer.render())
            samples.append(sample)

        pickup = RecordedPickup(scene, capture)
        for phase in ("_approach", "_grasp", "_lift", "_hold"):
            carry._operate_phase(pickup, phase)
    result = {
        "scene_version": scene.sv["published"]["version_id"],
        "seed": 0,
        "fps": 25,
        "grasp_mode": "bilateral contact followed by weld assist",
        "max_cup_tilt_deg": max(s["cup_tilt_deg"] for s in samples),
        "hold_min_clearance_cm": min(s["clearance_cm"] for s in samples if s["phase"] == "hold"),
        "tightenings": pickup._tighten,
        "events": pickup.events,
        "samples": samples,
    }
    (out / "run.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/upright_demo"))
    args = parser.parse_args()
    data = record(args.out)
    print(f"Max tilt {data['max_cup_tilt_deg']:.2f}°, hold clearance {data['hold_min_clearance_cm']:.2f} cm")
