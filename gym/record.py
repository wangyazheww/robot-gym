#!/usr/bin/env python3
"""可选视频录制：offscreen 渲染，仅代表性试次启用（渲染显著拖慢批量）。"""
from __future__ import annotations

from pathlib import Path

import mujoco


class Recorder:
    def __init__(self, scene, out_path: Path, fps: int = 25, cam: str = "demo_cam",
                 height: int = 480, width: int = 640):
        self.renderer = mujoco.Renderer(scene.model, height, width)
        self.cam_name = cam
        self.fps = fps
        self.out_path = Path(out_path)
        self._frames: list = []
        self._since_snap = 0
        self._snap_every = max(1, int(round(1.0 / (fps * scene.task.control.dt))))

    def maybe_snap(self, data) -> None:
        self._since_snap += 1
        if self._since_snap >= self._snap_every:
            self._since_snap = 0
            self.renderer.update_scene(data, camera=self.cam_name)
            self._frames.append(self.renderer.render())

    def close(self) -> None:
        if not self._frames:
            return
        try:
            import imageio
            with imageio.get_writer(str(self.out_path), fps=self.fps) as w:
                for f in self._frames:
                    w.append_data(f)
        except Exception as e:  # 录制失败不影响实验数据
            print(f"[recorder] video write failed: {e}", flush=True)
        finally:
            self._frames = []
            self.renderer.close()
