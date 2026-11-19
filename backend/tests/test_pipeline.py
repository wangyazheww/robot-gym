"""A2 验收测试（pytest backend/tests/test_pipeline.py）。

覆盖验收标准：
  1. 扁平 Y-up 圆盘处理后高度轴不变（不立起来）
  2. 节点缩放 0.5 的盒子尺寸正确（保留节点变换）
  3. 输出 GLB 的贴图数量和输入一致（保留原始 GLB）
  4. 缩放后最长边误差 < 2%
  5. 体积/质量：封闭网格用真实体积；不闭合用凸包
用 trimesh 程序化构造 GLB（无外部依赖），再加真实 Lux3D GLB（如存在）。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.asset_pipeline import process_asset, _load_mesh, _concat, _mesh_volume  # noqa: E402

ROOT = BACKEND.parent


def _glb_bytes(mesh: trimesh.Trimesh, transform=None) -> bytes:
    """mesh → GLB 字节；transform 是 (4,4) 节点变换。"""
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="n", geom_name="g",
                       transform=transform)
    buf = io.BytesIO()
    scene.export(buf, file_type="glb")
    return buf.getvalue()


def _disk_yup(radius=0.05, thickness=0.004):
    """扁平圆盘：Y-up，厚度在 Y 轴（像盘子）。"""
    c = trimesh.creation.cylinder(radius=radius, height=thickness, sections=64)
    # cylinder 默认沿 Z；旋转到 Y-up：厚度应在 Y 方向
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return c


def _count_images(glb: bytes) -> int:
    import struct
    chunk_len, _ = struct.unpack_from("<II", glb, 12)
    js = json.loads(glb[20:20 + chunk_len])
    return len(js.get("images", []))


def _process(tmp_path, glb, **kw):
    d = tmp_path / "asset_x"
    meta = process_asset(d, glb, kw.pop("vlm", {"name": "t", "size_cm": [10, 10, 10]}), **kw)
    return d, meta


# 1. 扁平 Y-up 圆盘：处理后 Y 高度仍是最小轴
def test_flat_disc_stays_flat(tmp_path):
    disc = _disk_yup()
    assert disc.extents[1] < disc.extents[0]  # Y 是薄轴
    glb = _glb_bytes(disc)
    d, meta = _process(tmp_path, glb, vlm={"name": "盘", "size_cm": [20, 1, 20]})
    size = np.array(meta["size_m"])
    assert size[1] <= 0.012, f"盘子被立起来了: {size}"      # Y≈1cm
    assert size[0] > 0.15                                   # X≈20cm


# 2. 节点缩放 0.5 的盒子：尺寸正确
def test_node_scale_preserved(tmp_path):
    box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    T = np.eye(4); T[:3, :3] *= 0.5   # 节点级 uniform scale 0.5
    glb = _glb_bytes(box, transform=T)
    scene = _load_mesh(glb)
    m = _concat(scene)                 # _concat 必须应用节点变换
    assert abs(m.extents[0] - 0.1) < 1e-4, f"节点缩放丢失: {m.extents}"


# 3. 贴图数量一致（视觉模型保留原始 GLB）
def test_textures_preserved(tmp_path):
    tex = np.zeros((8, 8, 3), dtype=np.uint8)
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    mat = trimesh.visual.material.PBRMaterial(name="m")
    box.visual = trimesh.visual.TextureVisuals(image=trimesh.visual.material.Image.fromarray(tex) if hasattr(trimesh.visual.material, "Image") else tex, material=mat)
    scene = trimesh.Scene(); scene.add_geometry(box)
    buf = io.BytesIO(); scene.export(buf, file_type="glb")
    glb = buf.getvalue()
    d, meta = _process(tmp_path, glb)
    out_glb = (d / "visual.glb").read_bytes()
    assert out_glb == glb, "visual.glb 必须等于输入原始 GLB 字节"
    assert _count_images(glb) == _count_images(out_glb)


# 4. 最长边误差 < 2%
def test_longest_edge_error(tmp_path):
    box = trimesh.creation.box(extents=[0.3, 0.1, 0.2])
    glb = _glb_bytes(box)
    d, meta = _process(tmp_path, glb, vlm={"name": "盒", "size_cm": [15, 5, 10]})
    target = max(0.15, 0.05, 0.10)
    got = max(meta["size_m"])
    assert abs(got - target) / target < 0.02, f"{got} vs {target}"


# 5. 体积：watertight 用真体积；非闭合用凸包
def test_volume_and_mass(tmp_path):
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    glb = _glb_bytes(box)
    d, meta = _process(tmp_path, glb, vlm={"name": "b", "size_cm": [10, 10, 10], "density_kgm3": 2000})
    # 10cm 立方体 → 0.001 m³ × 2000 = 2 kg
    assert abs(meta["volume_m3"] - 0.001) / 0.001 < 0.15
    assert abs(meta["mass_kg"] - 2.0) < 0.4

    # 不闭合网格（删面）
    broken = box.copy()
    broken.faces = broken.faces[:-4]
    glb2 = _glb_bytes(broken)
    scene = _load_mesh(glb2)
    m = _concat(scene)
    vol = _mesh_volume(m)
    hull_vol = abs(m.convex_hull.volume)
    assert abs(vol - hull_vol) / hull_vol < 0.05, "不闭合网格应退化为凸包体积"


# 6. 真实 Lux3D GLB（存在才跑）
REAL_GLBS = list((ROOT / "evidence" / "A1").glob("*.glb"))

@pytest.mark.skipif(not REAL_GLBS, reason="无真实 Lux3D GLB")
def test_real_lux3d_glb(tmp_path):
    for g in REAL_GLBS[:3]:
        d = tmp_path / g.stem
        meta = process_asset(d, g.read_bytes(), {"name": g.stem, "size_cm": [10, 10, 12], "lux3dTaskId": "123"})
        assert (d / "visual.glb").exists()
        assert (d / "collision.obj").exists()
        assert all(v > 0.001 for v in meta["size_m"])
        assert meta["source"]["type"] == "lux3d"
