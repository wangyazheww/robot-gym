"""Side pickup for the S2 tea table, with a constant world-space wrist orientation."""
from __future__ import annotations

import mujoco
import numpy as np

from .skills import GymFail, TaskExecutor


class UprightPickup(TaskExecutor):
    """Close on the side of a standing cup before enabling the disclosed weld assist."""

    def _ingest(self, obs):
        super()._ingest(obs)
        if self._grasp_active and self.phase in ("lift", "hold"):
            tilt = cup_tilt_deg(self.s)
            if tilt > 8:
                raise GymFail("unstable", f"持杯阶段杯口倾角 {tilt:.1f}°")

    def mug_bottom_z(self):
        if not self.s.scan_mug:
            return super().mug_bottom_z()
        if not hasattr(self, "_cup_vertices"):
            self._cup_geom = self.s.obj_geoms[self.obj_name][0]
            mesh = self.m.geom_dataid[self._cup_geom]
            start = self.m.mesh_vertadr[mesh]
            self._cup_vertices = self.m.mesh_vert[start:start+self.m.mesh_vertnum[mesh]]
        geom = self._cup_geom
        z_axis = self.d.geom_xmat[geom].reshape(3, 3)[2]
        return float(self.d.geom_xpos[geom, 2] + np.min(self._cup_vertices @ z_axis))

    def _hand_pose(self, point, seed, orientation=None, *, staging=False):
        q, distance, angle = self.robot.ik_pose(
            np.asarray(point), self.pick_orientation if orientation is None else orientation,
            np.asarray(seed),
        )
        if distance > (.016 if staging else .009) or angle > (.09 if staging else .05):
            raise GymFail("ik_failed", f"upright hand target error {distance:.3f} m / {angle:.3f} rad")
        return q

    def _servo(self, start, target, seconds, gap):
        def control(c, i, _):
            t = min(1., (i*self.task.control.dt) / seconds)
            blend = t*t*(3-2*t)
            q = start + blend*(target-start)
            self.robot.legs(c)
            self.robot.arm(c, q)
            self.robot.grip(c, gap)
            self.q_arm_cmd = q

        self._run(seconds + .45, control)

    def _approach(self):
        cup = self.s.obj_pos(self.obj_name)
        base = self.d.qpos[self.s.fbq:self.s.fbq+2]
        horizontal = cup[:2] - base
        forward = np.r_[horizontal / np.linalg.norm(horizontal), 0.]
        downward = np.array([0., 0., -1.])
        # Hand X is vertical, while the two pads close along hand Y.
        self.pick_orientation = np.column_stack((downward, np.cross(forward, downward), forward))
        ready = cup - .06*forward
        q0 = self.d.qpos[list(self.s.aq)].copy()
        q_high = self._hand_pose(ready + [0,0,.06], [0,2.06,-1.15,0,-.8,0])
        self._servo(q0, q_high, 2.8, .035)
        self.q_grasp = self._hand_pose(ready, q_high, staging=True)
        self._servo(q_high, self.q_grasp, 1.35, .035)
        self.ev("at_mug", ee=self.s.ee_pos().round(4).tolist(), mug=cup.round(4).tolist())

    def _grasp(self):
        d = self.d
        cup = self.s.obj_pos(self.obj_name)
        pad_centers = [d.geom_xpos[self.m.geom(name).id] for name in ("pad_left", "pad_right")]
        # Compensate using pad positions at the actual, settled wrist pose.
        offset = self.s.ee_pos() - np.mean(pad_centers, axis=0)
        q_target = self._hand_pose(cup + offset, self.q_grasp)
        self._servo(self.q_grasp, q_target, 2.2, .035)
        touch = {"samples":0, "done":False}

        def control(c, i, _):
            g = .035 - .017 * min(1., i*self.task.control.dt / 1.7)
            self.robot.legs(c)
            self.robot.arm(c, q_target)
            self.robot.grip(c, g)
            self.gap_cmd = g
            self.q_arm_cmd = q_target

        def contact(obs):
            threshold = self.task.criteria.contact_force_min_N
            left, right = obs["finger_force_left_N"], obs["finger_force_right_N"]
            touch["samples"] = touch["samples"] + 1 if min(left, right) >= threshold else 0
            if touch["samples"] >= 2:
                touch["done"] = True
                self.ev("contact_confirmed", f_left=left, f_right=right)

        self._run(3.2, control, hook=contact, break_when=lambda: touch["done"])
        if not touch["done"]:
            raise GymFail("no_contact", "侧向抓取未检测到连续双指接触")
        tilt = cup_tilt_deg(self.s)
        if tilt > 8:
            raise GymFail("unstable", f"闭爪时杯子倾角 {tilt:.1f}°")
        self.q_grasp = d.qpos[list(self.s.aq)].copy()
        self.lift_orientation = d.site_xmat[self.s.ee].reshape(3,3).copy()
        self._grasp_active = True
        self.s.weld_arm()
        self.obs.grasp_ref()
        self.ev("weld_assist_engaged", disclosed=True, cup_tilt_deg=round(tilt,2))

    def _lift(self):
        start = self.s.ee_pos().copy()
        q = self.q_grasp
        for height in (.035,.070,.105):
            next_q = self._hand_pose(start+[0,0,height], q, self.lift_orientation)
            self._servo(q, next_q, .52, self.gap_cmd)
            q = next_q
        self.q_lift = q
        self._lifted = True
        self.lift_peak_cm = max(self.lift_peak_cm, self.clearance_cm())
        self.ev("lifted", clear_cm=round(self.clearance_cm(),1), cup_tilt_deg=round(cup_tilt_deg(self.s),2))


def cup_tilt_deg(scene) -> float:
    address = scene.obj_jadr[scene.grasp_instance]
    _, x, y, _ = scene.data.qpos[address+3:address+7]
    return float(np.degrees(np.arccos(np.clip(1-2*(x*x+y*y), -1, 1))))
