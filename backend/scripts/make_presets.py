"""预置资产库生成（PRD F1.6）：离线兜底，演示稳定性的保险。

生成 5 个物体：红色马克杯 / 牛奶盒 / 小黄鸭 / 白瓷盘 / 蓝色积木
每个资产: visual.glb（Y-up、真实尺寸、底面 y=0）+ collision.obj（凸包）+ meta.json

用法: cd backend && python scripts/make_presets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

RNG = np.random.default_rng(42)


def colored(mesh: trimesh.Trimesh, rgb, roughness=0.65, metallic=0.0) -> trimesh.Trimesh:
    """统一上色：顶点色 + PBR 材质（GLTFLoader/_three.js 直接可用）。"""
    mesh = mesh.copy()
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, face_colors=np.array([*rgb, 255], dtype=np.uint8))
    mat = PBRMaterial(
        baseColorFactor=[*rgb, 255], metallicFactor=metallic,
        roughnessFactor=roughness, doubleSided=True)
    mesh.visual.material = mat
    return mesh


def cyl(r, h, seg=48):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=seg)
    # trimesh 圆柱沿 Z 轴 → 转成 Y-up，底面 y=0
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, h / 2, 0])
    return m


def box(w, d, h):
    m = trimesh.creation.box([w, h, d])
    m.apply_translation([0, h / 2, 0])
    return m


# ---------------------------------------------------------------- 物体定义

def make_mug():
    # 杯径 6.6cm：Panda 指垫净开口只有 ~7cm（指尖各占 1cm），必须留出余量
    body = cyl(0.033, 0.10)
    handle = trimesh.creation.torus(major_radius=0.02, minor_radius=0.006, major_sections=32)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    handle.apply_translation([0.04, 0.052, 0])
    red = (196, 40, 40)
    inner = cyl(0.029, 0.012)
    inner.apply_translation([0, 0.088, 0])
    visual = trimesh.util.concatenate([colored(body, red), colored(handle, red, roughness=0.4),
                                       colored(inner, (225, 220, 210))])
    collision = cyl(0.0335, 0.10, seg=24)  # 忽略把手；半径必须 < 夹爪最大开口的一半(0.035)，否则松爪时会被挤出
    return visual, collision, dict(name="红色马克杯", category="mug", mass_kg=0.30, friction=0.7)


def make_carton():
    body = box(0.07, 0.07, 0.155)
    roof = box(0.0495, 0.0495, 0.09)
    roof.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0]))
    roof.apply_translation([0, 0.152, 0])
    white = (240, 240, 235)
    blue = (52, 100, 190)
    visual = trimesh.util.concatenate([colored(body, white), colored(roof, blue)])
    collision = box(0.07, 0.07, 0.185)
    return visual, collision, dict(name="牛奶盒", category="carton", mass_kg=0.35, friction=0.6)


def make_duck():
    body = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
    body.apply_translation([0, 0.04, 0])
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.027)
    head.apply_translation([0.02, 0.082, 0])
    beak = trimesh.creation.cone(radius=0.011, height=0.022, sections=16)
    beak.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
    beak.apply_translation([0.048, 0.078, 0])
    yellow = (245, 195, 40)
    orange = (235, 120, 30)
    visual = trimesh.util.concatenate([colored(body, yellow, roughness=0.35),
                                       colored(head, yellow, roughness=0.35),
                                       colored(beak, orange)])
    collision = trimesh.convex.convex_hull(visual)
    return visual, collision, dict(name="小黄鸭", category="toy", mass_kg=0.08, friction=0.5)


def make_plate():
    base = cyl(0.10, 0.014)
    rim = cyl(0.095, 0.012)
    rim.apply_translation([0, 0.012, 0])
    visual = trimesh.util.concatenate([colored(base, (235, 235, 235), roughness=0.25),
                                       colored(rim, (225, 225, 225), roughness=0.25)])
    collision = cyl(0.10, 0.022, seg=32)
    return visual, collision, dict(name="白瓷盘", category="plate", mass_kg=0.40, friction=0.8)


def make_block():
    visual = box(0.05, 0.05, 0.05)
    visual = colored(visual, (40, 100, 220), roughness=0.5)
    collision = box(0.05, 0.05, 0.05)
    return visual, collision, dict(name="蓝色积木", category="block", mass_kg=0.06, friction=0.6)


PRESETS = {
    "preset_mug": make_mug,
    "preset_carton": make_carton,
    "preset_duck": make_duck,
    "preset_plate": make_plate,
    "preset_block": make_block,
}


def main() -> None:
    for asset_id, fn in PRESETS.items():
        visual, collision, info = fn()
        d = config.ASSETS_DIR / asset_id
        d.mkdir(parents=True, exist_ok=True)
        visual.export(d / "visual.glb", file_type="glb")
        collision.export(d / "collision.obj", file_type="obj")
        size = [round(float(v), 4) for v in collision.extents]
        meta = {
            "id": asset_id,
            "name": info["name"],
            "category": info["category"],
            "source": {"type": "preset", "image": None, "lux3dTaskId": None},
            "files": {"visual": "visual.glb", "collision": "collision.obj"},
            "size_m": size,
            "mass_kg": info["mass_kg"],
            "friction": info["friction"],
            "articulated": None,
        }
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"  {asset_id}: {info['name']:6s} size_m={size}")


if __name__ == "__main__":
    main()
