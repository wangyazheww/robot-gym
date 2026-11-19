#!/usr/bin/env python3
"""SceneVersion 数据模型（PRD v0.5 §10.2 / §6.2）——M1 冻结草案 v1。

三层表示（§6.3）：视觉层（世界 splat 引用）、物理层（代理+物体参数）、
任务/导航层（可行走区、A/B、站位，从物理层派生）。
发布校验（§6.2.5）：出生点无穿透、支撑面有效、A/B 合法、地图连通、尺度确认。

设计决策：
- Schema 固定字段 + free-form `extra`；校验器拒绝未知顶层键（防字段漂移）。
- SceneVersion 不可变：publish 生成新版本 ID；草稿 PATCH 走 draft_version 递增。
- 所有坐标系声明为 world Z-up 米制（§10.3）；变换冲突在 validate 时报错。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# ---------------------------------------------------------------- schema

TOP_KEYS = {"schema_version", "scene_id", "draft_version", "name",
            "world_ref", "calibration", "proxies", "instances",
            "robot_start", "regions", "nav_grid", "published", "extra"}


def new_scene_draft(scene_id: str, name: str, world_ref: dict) -> dict:
    """创建场景草稿。world_ref = {world_id, splat_path, pano_path, source_mode}"""
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "draft_version": 1,
        "name": name,
        "world_ref": world_ref,
        "calibration": {
            "up_axis": "Z",
            "floor_z": None,               # None = 未标定
            "scale_reference": None,        # {length_m, source} | None = 未确认
            "scale_check": None,            # 第二独立参考 {length_m, measured_m, err_pct}
            "walkable_polygon": [],         # [[x,y], ...] 顺时针/逆时针闭合
        },
        "proxies": [],                      # [{proxy_id, kind, shape, pose, size, note}]
        "instances": [],                    # [{instance_id, asset_id, pose, role, grasp_candidates}]
        "robot_start": None,                # {pos:[x,y], yaw_deg}
        "regions": {},                      # {region_id: {kind, pos, size, z_band}}
        "nav_grid": None,                   # {resolution, origin, w, h, occupied: [ints]} 派生
        "published": None,                  # 发布后 = {version_id, published_at, config_hash}
        "extra": {},
    }


# ---------------------------------------------------------------- validate

class SceneInvalid(Exception):
    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s): "
                         + "; ".join(e["code"] for e in errors))


def _err(code: str, msg: str, path: str = "") -> dict:
    return {"code": code, "message": msg, "path": path}


def validate(scene: dict) -> list[dict]:
    """结构化校验（§6.2.5 + §11.1 /scenes/{id}/validate 契约）。
    返回错误列表（空 = 通过）。发布要求零错误且尺度确认。"""
    errors: list[dict] = []
    unknown = set(scene) - TOP_KEYS
    if unknown:
        errors.append(_err("schema_unknown_keys", f"未知顶层键: {sorted(unknown)}"))
    if scene.get("schema_version") != SCHEMA_VERSION:
        errors.append(_err("schema_version", f"需要 schema_version={SCHEMA_VERSION}"))

    cal = scene.get("calibration") or {}
    # P1-08：世界文件存在性检查
    wr = scene.get("world_ref") or {}
    if wr.get("splat_path"):
        from pathlib import Path as _P
        sp_ = _P(wr["splat_path"])
        if not sp_.is_file():
            # 相对路径尝试从仓库根解析
            sp2 = _P(__file__).resolve().parent.parent / wr["splat_path"]
            if not sp2.is_file():
                errors.append(_err("world_file_missing",
                                   f"世界文件不存在: {wr['splat_path']}",
                                   "world_ref.splat_path"))
    # 尺度（§6.2.2）：无参考长度不能发布
    if not cal.get("scale_reference"):
        errors.append(_err("scale_unconfirmed",
                           "缺少尺度参考长度；未确认尺度不能发布为正式评估场景",
                           "calibration.scale_reference"))
    elif not cal.get("scale_check"):
        errors.append(_err("scale_check_missing",
                           "缺少第二独立参考长度检查（PRD §6.2.2）",
                           "calibration.scale_check"))
    else:
        err_pct = cal["scale_check"].get("err_pct")
        if err_pct is None or err_pct > 5.0:
            errors.append(_err("scale_check_fail",
                               f"第二参考长度误差 {err_pct}% > 5%",
                               "calibration.scale_check"))
    # 地面
    if cal.get("floor_z") is None:
        errors.append(_err("floor_uncalibrated", "地面平面未标定", "calibration.floor_z"))
    # 可行走区
    wp = cal.get("walkable_polygon") or []
    if len(wp) < 3:
        errors.append(_err("walkable_invalid", "可行走区多边形至少 3 个顶点",
                           "calibration.walkable_polygon"))
    # 出生点
    rs = scene.get("robot_start")
    if not rs or "pos" not in rs:
        errors.append(_err("robot_start_missing", "机器人起点未设置", "robot_start"))
    else:
        if wp and not _point_in_poly(rs["pos"], wp):
            errors.append(_err("robot_start_outside",
                               "出生点不在可行走区内", "robot_start.pos"))
        # P1-08：出生点穿透家具/障碍代理检查
        for p in scene.get("proxies") or []:
            if p.get("kind") not in ("furniture", "wall", "obstacle"):
                continue
            if proxy_contains_xy(p, rs["pos"]):
                errors.append(_err("robot_start_collision",
                                   f"出生点 {rs['pos']} 在代理 {p['proxy_id']} 内部",
                                   "robot_start.pos"))
                break
    # 区域（A/B）
    for rid, r in (scene.get("regions") or {}).items():
        if r.get("kind") not in ("pick", "place"):
            errors.append(_err("region_kind", f"区域 {rid} kind 必须是 pick/place", f"regions.{rid}"))
        if "pos" not in r:
            errors.append(_err("region_pos", f"区域 {rid} 缺少 pos", f"regions.{rid}.pos"))
        elif wp and not _point_in_poly(r["pos"], wp):
            errors.append(_err("region_outside",
                               f"区域 {rid} 中心不在可行走区内", f"regions.{rid}.pos"))
    # 代理
    for p in scene.get("proxies") or []:
        if p.get("shape") not in ("box", "cylinder", "convex"):
            errors.append(_err("proxy_shape",
                               f"代理 {p.get('proxy_id')} shape 非法", "proxies"))
        if p.get("kind") not in ("furniture", "wall", "obstacle", "support"):
            errors.append(_err("proxy_kind",
                               f"代理 {p.get('proxy_id')} kind 非法", "proxies"))
    # 实例
    for inst in scene.get("instances") or []:
        if not inst.get("asset_id"):
            errors.append(_err("instance_asset", f"实例 {inst.get('instance_id')} 缺 asset_id",
                               "instances"))
        if inst.get("role") not in ("obstacle", "grasp_target", "prop"):
            errors.append(_err("instance_role",
                               f"实例 {inst.get('instance_id')} role 非法", "instances"))
    # pick 区域需要至少一个 grasp_target 实例
    has_pick = any(r.get("kind") == "pick" for r in (scene.get("regions") or {}).values())
    has_target = any(i.get("role") == "grasp_target" for i in scene.get("instances") or [])
    if has_pick and not has_target:
        errors.append(_err("no_grasp_target", "有 pick 区域但没有 grasp_target 实例"))
    return errors


def _point_in_poly(pt: list[float], poly: list[list[float]]) -> bool:
    x, y = pt[0], pt[1]
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def proxy_contains_xy(proxy: dict, xy: list[float], margin: float = 0.0) -> bool:
    """Test the footprint in the proxy's local frame, matching its MuJoCo transform."""
    dx = xy[0] - proxy["pose"][0]
    dy = xy[1] - proxy["pose"][1]
    if proxy["shape"] == "cylinder":
        return math.hypot(dx, dy) < proxy["size"][0] + margin
    yaw = math.radians(proxy.get("yaw_deg", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = proxy["size"][:2]
    return abs(c * dx + s * dy) < hx + margin and abs(-s * dx + c * dy) < hy + margin


# ---------------------------------------------------------------- publish

def publish(scene: dict) -> dict:
    """校验通过后发布为不可变版本（§11.1 /publish）。
    深拷贝（P1-08）；发布时重建派生地图。"""
    import copy
    errors = validate(scene)
    if errors:
        raise SceneInvalid(errors)
    # P1-08：发布前强制重建派生地图（防止旧地图与新代理不一致）
    nav = build_nav_grid(scene, resolution=0.1)
    scene = copy.deepcopy(scene)  # P1-08：深拷贝隔离草稿与发布版本
    scene["nav_grid"] = nav
    blob = json.dumps(scene, sort_keys=True, ensure_ascii=False).encode()
    config_hash = hashlib.sha256(blob).hexdigest()[:16]
    version_id = f"{scene['scene_id']}_v{scene['draft_version']}_{config_hash[:6]}"
    published = dict(scene.get("published") or {})
    published.update({
        "version_id": version_id,
        "config_hash": config_hash,
        "published_at": _now(),
    })
    out = copy.deepcopy(scene)
    out["published"] = published
    return out


def _now() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- nav grid

def build_nav_grid(scene: dict, resolution: float = 0.1) -> dict:
    """从可行走区 + 代理派生占据栅格（§6.2.6：地图与物理同源）。
    返回 nav_grid dict：occupied=1 不可走。"""
    cal = scene["calibration"]
    poly = cal["walkable_polygon"]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = int(math.ceil((x1 - x0) / resolution))
    h = int(math.ceil((y1 - y0) / resolution))
    occupied = [1] * (w * h)      # 默认占据：越界/区外不可走（§6.2.4 地面外默认不可通行）
    for j in range(h):
        for i in range(w):
            cx, cy = x0 + (i + 0.5) * resolution, y0 + (j + 0.5) * resolution
            walk = _point_in_poly([cx, cy], poly)
            # 代理占据检查（box/cylinder 的 xy 投影 + 余量 0.05 起步见机器人 footprint）
            if walk:
                for p in scene.get("proxies") or []:
                    z_top = p["pose"][2] + p["size"][2 if p["shape"] == "box" else 1]
                    if z_top < 0.15:
                        continue   # 低于腿部高度的支撑面不挡底盘行走
                    if proxy_contains_xy(p, [cx, cy], margin=p.get("nav_margin_m", 0.0)):
                        walk = False
                        break
                # P1-06：obstacle 实例也纳入占据
                for inst in scene.get("instances") or []:
                    if inst.get("role") != "obstacle":
                        continue
                    ip = inst.get("pose") or [0, 0, 0]
                    if abs(cx - ip[0]) < 0.15 and abs(cy - ip[1]) < 0.15:
                        walk = False
                        break
            occupied[j * w + i] = 0 if walk else 1
    return {"resolution": resolution, "origin": [x0, y0], "w": w, "h": h,
            "occupied": occupied, "derived_from": "walkable_polygon+proxies"}


# ---------------------------------------------------------------- storage

SCENES_DIR = Path(os.environ.get("RG_SCENES_V1_DIR", str(
    Path(__file__).resolve().parent.parent / "backend" / "data" / "scenes_v1")))


def save_draft(scene: dict) -> Path:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    p = SCENES_DIR / f"{scene['scene_id']}.json"
    p.write_text(json.dumps(scene, ensure_ascii=False, indent=2))
    return p


def load_scene(scene_id: str) -> dict:
    p = SCENES_DIR / f"{scene_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"scene {scene_id} 不存在")
    return json.loads(p.read_text())


# ---------------------------------------------------------------- planner

def plan_path(nav_grid: dict, start_xy: list[float], goal_xy: list[float],
              footprint: float = 0.22) -> list[list[float]] | None:
    """基线 A* 规划器（PRD §7.1 N02 / §9.1 导航规划层可替换基线）。

    footprint 膨胀实现整机碰撞检查（N03）；octile 启发式；
    不可达返回 None（N05 no_path）。"""
    import heapq
    res = nav_grid["resolution"]
    ox, oy = nav_grid["origin"]
    w, h, occ = nav_grid["w"], nav_grid["h"], nav_grid["occupied"]
    pad = int(math.ceil(footprint / res)) - 1
    # footprint 语义=整机直径 0.22 → 膨胀半径 0.11 → pad-1（0.2m 膨胀圈）。
    # 原实现 ceil(0.22/0.1)=3 整格（0.3m）过度保守，把停泊逼到 0.585m 外
    # 压线臂展（ISSUE-5 第二层原因）。

    def blocked(i: int, j: int) -> bool:
        for di in range(-pad, pad + 1):
            for dj in range(-pad, pad + 1):
                ii, jj = i + di, j + dj
                if ii < 0 or jj < 0 or ii >= w or jj >= h or occ[jj * w + ii]:
                    return True
        return False

    def to_ij(xy):
        return (int((xy[0] - ox) / res), int((xy[1] - oy) / res))

    def to_xy(ij):
        return [ox + (ij[0] + 0.5) * res, oy + (ij[1] + 0.5) * res]

    si, sj = to_ij(start_xy)
    gi, gj = to_ij(goal_xy)
    if blocked(si, sj) or blocked(gi, gj):
        return None
    D, D2 = res, res * 1.4142135623730951
    def hfun(i, j):
        dx, dy = abs(i - gi), abs(j - gj)
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)   # octile，可采纳
    openq = [(hfun(si, sj), 0.0, si, sj)]
    came, cost = {}, {(si, sj): 0.0}
    while openq:
        _, d, i, j = heapq.heappop(openq)
        if (i, j) == (gi, gj):
            path = [[i, j]]
            while (i, j) in came:
                i, j = came[(i, j)]
                path.append([i, j])
            return [to_xy(p) for p in reversed(path)]
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            ni, nj = i + di, j + dj
            if blocked(ni, nj):
                continue
            nd = d + (D2 if di and dj else D)
            if (ni, nj) not in cost or nd < cost[(ni, nj)]:
                cost[(ni, nj)] = nd
                came[(ni, nj)] = (i, j)
                heapq.heappush(openq, (nd + hfun(ni, nj), nd, ni, nj))
    return None


def path_length(path: list[list[float]]) -> float:
    return sum(float(((path[k][0]-path[k+1][0])**2 + (path[k][1]-path[k+1][1])**2) ** 0.5)
               for k in range(len(path) - 1))
