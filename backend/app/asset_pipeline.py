"""资产流水线（PRD F1.4）：Lux3D GLB → 可仿真资产。

A2 设计（验收标准驱动）：
  - glTF 规范就是 Y-up：不做任何启发式翻转（实测 Lux3D 输出亦 Y-up）。
  - **视觉模型保留原始 GLB 和贴图**（visual.glb == 输入字节，贴图不重打包不丢失）；
    缩放/偏移写进 meta.transform，由前端在加载 visual.glb 后应用。
  - 碰撞体用处理后的网格导出 OBJ（凸包）。
  - 质量 = 体积 × 密度；体积优先用封闭网格的 watertight 体积，
    不闭合时退化为凸包体积。

输出: data/assets/{id}/  visual.glb / collision.obj / meta.json / input.jpg
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh


def _load_mesh(glb_bytes: bytes) -> trimesh.Scene:
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")
    if isinstance(scene, trimesh.Trimesh):
        s = trimesh.Scene(); s.add_geometry(scene); scene = s
    return scene


def _concat(scene: trimesh.Scene) -> trimesh.Trimesh:
    """合并场景所有 mesh，保留每个节点的世界变换（缩放/旋转/平移）。

    GLB 的顶点存在节点局部系里，直接 concatenate scene.geometry 会丢节点变换。
    """
    meshes = []
    for node in scene.graph.nodes_geometry:
        T, geom_ref = scene.graph.get(node)
        geom = scene.geometry.get(geom_ref)
        if not (isinstance(geom, trimesh.Trimesh) and len(geom.faces)):
            continue
        g = geom.copy()
        g.apply_transform(T)
        meshes.append(g)
    if not meshes:
        raise ValueError("GLB 内没有可用 mesh")
    return trimesh.util.concatenate(meshes)


def _mesh_volume(mesh: trimesh.Trimesh) -> float:
    """体积 m³。优先 watertight 体积；不闭合时用凸包体积（A2 要求）。"""
    if mesh.is_watertight:
        return float(abs(mesh.volume))
    hull = mesh.convex_hull
    if hull is not None and hull.is_watertight:
        return float(abs(hull.volume))
    # 兜底：包围盒
    return float(np.prod(mesh.extents))


def process_asset(asset_dir: Path, glb_bytes: bytes, vlm: Dict[str, Any],
                  input_image: Optional[bytes] = None) -> Dict[str, Any]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    scene = _load_mesh(glb_bytes)
    raw = _concat(scene)  # 原始姿态（Y-up，不翻转）

    size_cm = vlm.get("size_cm") or [8, 8, 10]
    target = np.array(size_cm, dtype=float) / 100.0
    cur = raw.extents
    scale = float(np.max(target) / max(np.max(cur), 1e-6))
    # 缩放后的平移：XY 居中、底面 y=0（把缩放+平移写成显式 transform 给前端）
    scaled = raw.copy()
    scaled.apply_scale(scale)
    c = scaled.bounds.mean(axis=0)
    offset = np.array([-c[0], -scaled.bounds[0][1], -c[2]])

    # ---- 产物 ----
    if input_image:
        (asset_dir / "input.jpg").write_bytes(input_image)
    # 视觉模型 = 原始 GLB 字节（贴图原样保留）
    (asset_dir / "visual.glb").write_bytes(glb_bytes)
    # 碰撞体 = 缩放后的凸包
    coll = scaled.convex_hull
    if coll is None or len(coll.faces) == 0:
        coll = scaled.bounding_box
    coll.export(asset_dir / "collision.obj", file_type="obj")

    # 质量 = 体积 × 密度（kg/m³）；VLM 可给 material 密度覆盖
    volume_m3 = _mesh_volume(scaled)
    density = float(vlm.get("density_kgm3", 1000.0))
    mass_kg = max(volume_m3 * density, 0.01)

    meta = {
        "id": asset_dir.name,
        "name": vlm.get("name", "物体"),
        "category": vlm.get("category", "unknown"),
        "source": {"type": "lux3d" if vlm.get("lux3dTaskId") else "photo",
                   "image": "input.jpg", "lux3dTaskId": vlm.get("lux3dTaskId")},
        "files": {"visual": "visual.glb", "collision": "collision.obj"},
        "size_m": [round(float(v), 4) for v in scaled.extents],
        # 前端加载 visual.glb 后要应用的变换：先 scale 后 translate
        "transform": {"scale": round(scale, 6),
                      "translate": [round(float(v), 6) for v in offset]},
        "volume_m3": round(volume_m3, 6),
        "density_kgm3": density,
        "mass_kg": round(mass_kg, 4),
        "friction": float(vlm.get("friction", 0.6)),
        "material": vlm.get("material", "unknown"),
        "material_source": vlm.get("material_source", "default"),
        "articulated": None,
    }
    (asset_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
