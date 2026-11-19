"""FastAPI 入口（PRD §6 API 表）。

开发：cd backend && uvicorn app.main:app --reload --port 8000
生产：python -m app.main   （同时挂载 frontend/dist，单端口演示）
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import asset_pipeline, config, planner, vlm
from .aholo_client import AholoClient, AholoError

app = FastAPI(title="Robot Gym", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
aholo = AholoClient()

# assetId -> {status: queued/generating/processing/ready/failed, meta, error, progress, stage}
_ASSET_JOBS: Dict[str, Dict[str, Any]] = {}
_WORLD_JOBS: Dict[str, Dict[str, Any]] = {}


def _save_upload(data: bytes, dest: Path) -> None:
    dest.write_bytes(data)


# ---------------------------------------------------------------------- #
# 资产：照片 → Lux3D → VLM → 流水线 → meta.json
# ---------------------------------------------------------------------- #

CATEGORY_PRESET = {
    "mug": "preset_mug", "carton": "preset_carton", "toy": "preset_duck",
    "plate": "preset_plate", "block": "preset_block", "bottle": "preset_carton",
}

MATERIAL_FRICTION = {
    "plastic": 0.45, "wood": 0.55, "metal": 0.35, "glass_ceramic": 0.40,
    "rubber": 0.95, "unknown": 0.60,
}


async def _estimate_asset(image_bytes: bytes, filename: str) -> dict:
    try:
        return await vlm.estimate(image_bytes, filename)
    except Exception:
        return vlm.mock_estimate(filename)


async def _run_asset_pipeline(asset_id: str, image_path: Path, preset_id: Optional[str]) -> None:
    job = _ASSET_JOBS[asset_id]
    try:
        job.update(status="generating", stage="Lux3D 生成中", progress=0.05)
        image_bytes = image_path.read_bytes()

        # mock 模式：先跑 VLM 识别，按类别映射到预置资产兜底
        vlm_info = None
        if config.AHOLO_MOCK and not preset_id:
            vlm_info = await _estimate_asset(image_bytes, image_path.name)
            preset_id = CATEGORY_PRESET.get(vlm_info.get("category", ""), "preset_block")

        def on_progress(task_id: str, frac: float, stage: str) -> None:
            job.update(progress=max(job.get("progress", 0), 0.05 + frac * 0.75), stage=stage)

        out = await aholo.img_to_3d(image_bytes, image_path.name, preset_id, on_progress)
        job.update(status="processing", stage="识别与处理中", progress=0.85)

        # 真实模式产物立即落盘（URL ~2h 过期）：ZIP 附带文件 + GLB 都存进资产目录
        if out.get("zip_bytes"):
            (config.ASSETS_DIR / asset_id / "lux3d.zip").write_bytes(out["zip_bytes"])
        else:
            (config.ASSETS_DIR / asset_id / "visual.glb").parent.mkdir(parents=True, exist_ok=True)
            (config.ASSETS_DIR / asset_id / "lux3d_raw.glb").write_bytes(out["glb_bytes"])

        if vlm_info is None:
            vlm_info = await _estimate_asset(image_bytes, image_path.name)
        material = vlm_info.get("material", "unknown")
        if material not in MATERIAL_FRICTION:
            material = "unknown"
        vlm_info["material"] = material
        vlm_info["lux3dTaskId"] = out["task_id"]
        asset_pipeline.process_asset(config.ASSETS_DIR / asset_id, out["glb_bytes"], vlm_info, image_bytes)

        job.update(status="ready", stage="完成", progress=1.0,
                   meta=json.loads((config.ASSETS_DIR / asset_id / "meta.json").read_text()))
    except (AholoError, Exception) as e:  # noqa: BLE001
        job.update(status="failed", stage="失败", error=str(e)[:300])


@app.post("/api/assets")
async def create_asset(
    background: BackgroundTasks,
    image: UploadFile = File(...),
    preset_id: Optional[str] = None,
) -> JSONResponse:
    asset_id = f"a_{uuid.uuid4().hex[:6]}"
    raw_dir = config.ASSETS_DIR / asset_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_path = raw_dir / f"input{Path(image.filename or 'photo.jpg').suffix.lower() or '.jpg'}"
    _save_upload(await image.read(), image_path)
    _ASSET_JOBS[asset_id] = {"id": asset_id, "status": "queued", "stage": "排队中", "progress": 0.0}
    # 传函数+参数：跑在主事件循环上，与全局 AholoClient 共享 AsyncClient（Bug #5）
    background.add_task(_run_asset_pipeline, asset_id, image_path, preset_id)
    return JSONResponse({"id": asset_id, "status": "queued"})


@app.get("/api/assets/{asset_id}")
async def get_asset(asset_id: str) -> Any:
    if asset_id in _ASSET_JOBS:
        return _ASSET_JOBS[asset_id]
    meta_path = config.ASSETS_DIR / asset_id / "meta.json"
    if meta_path.exists():
        return {"id": asset_id, "status": "ready", "progress": 1.0, "stage": "完成",
                "meta": json.loads(meta_path.read_text())}
    raise HTTPException(404, "asset not found")


@app.get("/api/assets")
async def list_assets() -> Any:
    items = []
    for d in sorted(config.ASSETS_DIR.iterdir()):
        meta_path = d / "meta.json"
        if meta_path.exists():
            items.append({"id": d.name, "status": "ready",
                          "meta": json.loads(meta_path.read_text())})
        elif d.name in _ASSET_JOBS:
            items.append(_ASSET_JOBS[d.name])
    items.sort(key=lambda x: 0 if x.get("meta", {}).get("source", {}).get("type") == "preset" else 1)
    return items


# ---------------------------------------------------------------------- #
# 背景：房间照片 → AHOLO World
# ---------------------------------------------------------------------- #

async def _run_world_pipeline(world_id: str, photos: list[Path]) -> None:
    job = _WORLD_JOBS[world_id]
    reconstruction = len(photos) > 1
    wd = config.WORLDS_DIR / world_id
    source_mode = "reconstructed" if reconstruction else "generated"
    try:
        job.update(status="generating", stage="上传重建照片" if reconstruction else "World 生成中",
                   progress=0.1)

        def on_created(remote_id: str) -> None:
            job.update(stage="高质量重建中", worldId=remote_id, started=time.time())
            (wd / "world_pending.json").write_text(json.dumps(
                {"worldId": remote_id, "sourceMode": source_mode}, ensure_ascii=False))

        if reconstruction:
            out = await aholo.world_reconstruct(photos, "Robot Gym 房间", on_created=on_created)
        else:
            job["started"] = time.time()
            out = await aholo.world_generate(photos[0].read_bytes(), photos[0].name)
        pano_rel = splat_rel = None
        if out.get("pano_bytes"):
            (wd / "pano.jpg").write_bytes(out["pano_bytes"])
            pano_rel = f"/files/worlds/{world_id}/pano.jpg"
        if out.get("splat_bytes"):
            suffix = out["splat_extension"]
            (wd / f"splat{suffix}").write_bytes(out["splat_bytes"])
            splat_rel = f"/files/worlds/{world_id}/splat{suffix}"
        # 持久化任务 ID、来源照片与坐标元数据（复查 / 第三步标定用）
        assets = out.get("assets") or {}
        (wd / "world_success.json").write_text(json.dumps({
            "worldId": out.get("world_id") or world_id,
            "taskId": out.get("task_id"),
            "sourcePhoto": photos[0].name,
            "sourcePhotos": [p.name for p in photos],
            "sourceMode": source_mode,
            "assets": assets,
            "upAxis": (assets.get("semanticsMetadata") or {}).get("upAxis", "Z"),
        }, ensure_ascii=False, indent=2))
        job.update(status="ready", progress=1.0,
                   stage="完成" if splat_rel else "完成（无三维产物）",
                   result={"pano": pano_rel, "splat": splat_rel,
                           "worldId": out.get("world_id") or world_id,
                           "upAxis": (assets.get("semanticsMetadata") or {}).get("upAxis", "Z"),
                           "sourceMode": source_mode})
    except Exception as e:  # noqa: BLE001
        job.update(status="failed", stage="失败", error=str(e)[:300])
        (wd / "world_failed.json").write_text(json.dumps(job, ensure_ascii=False))


@app.get("/api/worlds")
async def list_worlds() -> list[dict]:
    completed = []
    for directory in sorted(config.WORLDS_DIR.glob("w_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not any((directory / name).is_file() for name in ("splat.spz", "splat.ply")):
            continue
        try:
            info = await get_world(directory.name)
        except (HTTPException, ValueError):
            continue
        if info["status"] == "ready" and info.get("result", {}).get("splat"):
            completed.append(info)
    return completed


@app.post("/api/worlds")
async def create_world(background: BackgroundTasks, image: list[UploadFile] = File(...)) -> JSONResponse:
    if len(image) != 1 and len(image) < 20:
        raise HTTPException(422, "请上传 1 张生成房间，或至少 20 张同一房间照片做重建。")
    suffixes = [Path(photo.filename or "").suffix.lower() for photo in image]
    if any(suffix not in {".jpg", ".jpeg", ".png", ".webp"} for suffix in suffixes):
        raise HTTPException(422, "仅支持 JPG、PNG 和 WebP 照片。")
    world_id = f"w_{uuid.uuid4().hex[:6]}"
    raw = config.WORLDS_DIR / world_id
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (photo, suffix) in enumerate(zip(image, suffixes)):
        path = raw / (f"input{suffix}" if len(image) == 1 else f"photo_{i+1:03d}{suffix}")
        with path.open("wb") as f:
            while chunk := await photo.read(1024 * 1024):
                f.write(chunk)
        paths.append(path)
    _WORLD_JOBS[world_id] = {"id": world_id, "status": "queued", "stage": "排队中", "progress": 0.0}
    # 传函数+参数（异步任务在响应后的同一事件循环执行）；传协程对象会被 starlette 0.49 丢进线程池分支而报错
    background.add_task(_run_world_pipeline, world_id, paths)
    return JSONResponse({"id": world_id, "status": "queued"})


@app.get("/api/worlds/{world_id}")
async def get_world(world_id: str) -> Any:
    if world_id in _WORLD_JOBS:
        job = dict(_WORLD_JOBS[world_id])
        # 生成中按耗时给出平滑进度（World 全程 8-15 分钟）
        if job.get("status") == "generating" and job.get("started"):
            elapsed = time.time() - job["started"]
            job["progress"] = min(0.9, 0.1 + elapsed / 900.0 * 0.8)
        return job
    if not world_id.startswith("w_") or not world_id[2:].isalnum():
        raise HTTPException(404, "world not found")
    wd = config.WORLDS_DIR / world_id
    splat = next((wd / name for name in ("splat.spz", "splat.ply") if (wd / name).is_file()), None)
    if splat:
        metadata = wd / "world_success.json"
        record = json.loads(metadata.read_text()) if metadata.is_file() else {}
        return {"id": world_id, "status": "ready", "progress": 1.0, "stage": "完成",
                "result": {"pano": f"/files/worlds/{world_id}/pano.jpg" if (wd / "pano.jpg").is_file() else None,
                           "splat": f"/files/worlds/{world_id}/{splat.name}",
                           "worldId": record.get("worldId", world_id),
                           "upAxis": record.get("upAxis", "Z"),
                           "sourceMode": record.get("sourceMode", "generated")}}
    if (wd / "world_failed.json").is_file():
        return json.loads((wd / "world_failed.json").read_text())
    raise HTTPException(404, "world not found")


# ---------------------------------------------------------------------- #
# 规划 / 场景
# ---------------------------------------------------------------------- #

@app.post("/api/plan")
async def api_plan(body: Dict[str, Any]) -> Any:
    scene = body.get("scene") or {}
    instruction = (body.get("instruction") or "").strip()
    if not instruction:
        raise HTTPException(400, "instruction 为空")
    t0 = time.time()
    result = await planner.plan(scene, instruction)
    result["planner_ms"] = int((time.time() - t0) * 1000)
    return result


@app.post("/api/scenes")
async def save_scene(body: Dict[str, Any]) -> Any:
    scene = dict(body)
    scene_id = scene.get("id") or f"s_{uuid.uuid4().hex[:6]}"
    scene["id"] = scene_id
    (config.SCENES_DIR / f"{scene_id}.json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2))
    return {"id": scene_id}




# ---------------------------------------------------------------------- #
# 房间场景版本（v0.5 本轮第三步）：标定 + 碰撞代理 + MuJoCo 验证
# 复用 gym/scene_version.py（纯标准库，importlib 从仓库根加载，backend venv
# 不需要 mujoco；MuJoCo 运行走 spikes/humanoid venv 子进程）
# ---------------------------------------------------------------------- #

import importlib.util as _ilu

_sv_spec = _ilu.spec_from_file_location(
    "gym_scene_version", config.BASE_DIR.parent / "gym" / "scene_version.py")
sv_mod = _ilu.module_from_spec(_sv_spec)
_sv_spec.loader.exec_module(sv_mod)

_REPO = config.BASE_DIR.parent
_ROOM_VERIFY_PYTHON = Path(os.environ.get("ROBOTGYM_SIM_PYTHON",
                                        str(_REPO / "spikes/humanoid/.venv/bin/python")))


def _room_key(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise HTTPException(422, "场景或实体 ID 无效")
    return value


def _room_vector(value: Any, n: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise HTTPException(422, "坐标或尺寸格式无效")
    try:
        values = [float(v) for v in value]
    except (ValueError, TypeError, OverflowError):
        raise HTTPException(422, "坐标或尺寸格式无效")
    if not all(math.isfinite(v) for v in values):
        raise HTTPException(422, "坐标或尺寸必须为有限数字")
    return values


def _room_splat_path(value: str) -> Path:
    if value in ("evidence/S2/splat.spz", "/demo/evidence/S2/splat.spz"):
        return _REPO / "evidence/S2/splat.spz"
    match = re.fullmatch(r"/files/worlds/(w_[a-z0-9]+)/splat\.(spz|ply)", value or "")
    if not match:
        raise HTTPException(422, "请选择已生成的三维房间")
    return config.WORLDS_DIR / match[1] / f"splat.{match[2]}"


def _room_start(scene_id: str) -> dict:
    try:
        return sv_mod.load_scene(_room_key(scene_id))
    except FileNotFoundError:
        raise HTTPException(404, "找不到场景")


def _sv_from_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """页面请求 → SceneVersion 草稿（同一 schema，网页预览/MJCF/导航读同一批代理）。"""
    boxes = body.get("boxes") or []
    proxies = []
    for i, b in enumerate(boxes):
        size = [v / 2 for v in _room_vector(b.get("size"), 3)]
        if min(size) <= 0:
            raise HTTPException(422, "碰撞盒的长、宽、高都要大于零")
        margin = _room_vector([b.get("navMarginM", 0)], 1)[0]
        if not 0 <= margin <= .5:
            raise HTTPException(422, "导航余量须在 0 到 0.5 米之间")
        proxies.append({
            "proxy_id": _room_key(b.get("id") or f"box_{i+1}"),
            "kind": b.get("kind") or "obstacle",
            "shape": "box",
            "pose": _room_vector(b.get("pose"), 3),
            "size": size,
            "yaw_deg": _room_vector([b.get("yawDeg", 0)], 1)[0],
            "nav_margin_m": margin,
            "note": b.get("note") or "",
        })
    cal = body.get("calibration") or {}
    world_id = _room_key(body.get("worldId"))
    splat_url = body.get("splatPath")
    splat_path = _room_splat_path(splat_url)
    if not splat_path.is_file():
        raise HTTPException(422, "房间的三维文件还没就绪")
    sv = sv_mod.new_scene_draft(
        scene_id=_room_key(body.get("sceneId") or f"room_{world_id}"),
        name=str(body.get("name") or f"房间 {body.get('worldId', '')}"),
        world_ref={
            "world_id": world_id,
            "splat_path": str(splat_path),
            "splat_url": splat_url,
            "pano_path": body.get("panoPath") or "",
            "source_mode": body.get("sourceMode") or "generated",
        })
    sv["calibration"] = {
        "up_axis": "Z",
        "floor_z": cal.get("floorZ", 0.0),
        "scale_reference": cal.get("scaleReference"),   # {length_m, source} | None
        "scale_check": cal.get("scaleCheck"),
        "walkable_polygon": [_room_vector(p, 2) for p in cal.get("walkablePolygon") or []],
        "splat_transform": cal.get("splatTransform"),
    }
    tr = sv["calibration"]["splat_transform"]
    if tr:
        tr["position"] = _room_vector(tr.get("position"), 3)
        tr["quaternion"] = _room_vector(tr.get("quaternion"), 4)
        tr["scale"] = _room_vector([tr.get("scale")], 1)[0]
        if not 0.02 <= tr["scale"] <= 50 or abs(sum(v*v for v in tr["quaternion"]) - 1) > .01:
            raise HTTPException(422, "三维场景的变换无效")
    polygon = sv["calibration"]["walkable_polygon"]
    if polygon and max(max(p[i] for p in polygon) - min(p[i] for p in polygon) for i in (0,1)) > 60:
        raise HTTPException(422, "当前只支持最大 60 米的单房间标定")
    sv["proxies"] = proxies
    sv["instances"] = []
    rs = body.get("robotStart") or {}
    sv["robot_start"] = {"pos": _room_vector(rs.get("pos") or [0, 0], 2),
                         "yaw_deg": _room_vector([rs.get("yawDeg", 0)], 1)[0]}
    sv["extra"] = {"source": "room.html", "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    obj = body.get("object")
    if obj:
        asset_id = _room_key(obj.get("assetId"))
        meta_path = config.ASSETS_DIR / asset_id / "meta.json"
        sample_cup = asset_id == "lux3d_mug_demo" and world_id == "3FO4K4RCSR22"
        if not sample_cup and not meta_path.is_file():
            raise HTTPException(422, "物品模型还没就绪")
        meta = ({"size_m": [.061, .1, .061], "category": "mug", "mass_kg": .25}
                if sample_cup else json.loads(meta_path.read_text()))
        width, height, depth = _room_vector(meta.get("size_m"), 3)
        if min(width, height, depth) <= 0:
            raise HTTPException(422, "物品尺寸无效")
        material = obj.get("material", "unknown")
        if material not in MATERIAL_FRICTION:
            raise HTTPException(422, "请选择有效的接触面材质")
        xy = _room_vector(obj.get("xy"), 2)
        support_id = obj.get("supportId")
        support_z = 0.0
        if support_id:
            support = next((p for p in proxies if p["proxy_id"] == support_id and p["kind"] == "support"), None)
            if support is None or not sv_mod.proxy_contains_xy(support, xy):
                raise HTTPException(422, "请把 A 点选在支撑台范围内")
            support_z = support["pose"][2] + support["size"][2]
        material_source = obj.get("materialSource")
        if material_source not in {"vlm", "manual", "default"}:
            material_source = "default"
        physics = {
            "geometry_kind": "mug_like" if meta.get("category") in {"mug", "cup"} else "box_like",
            "size_xyz_m": [width, depth, height],
            "mass_kg": max(0.01, float(meta.get("mass_kg") or 0.2)),
            "friction": MATERIAL_FRICTION[material], "material": material,
            "material_source": material_source, "support_id": support_id,
        }
        if sample_cup and material_source == "default":
            physics = {"geometry_kind": "mug_like", "support_id": support_id}
        sv["instances"].append({
            "instance_id": "room_object", "asset_id": asset_id, "role": "grasp_target",
            "pose": [*xy, support_z + height / 2],
            "extra": physics,
        })
        sv["regions"]["A_pick"] = {"kind": "pick", "pos": xy, "size": [0.30, 0.30]}
    if body.get("place") is not None:
        sv["regions"]["B_place"] = {"kind": "place", "pos": _room_vector(body["place"], 2),
                                     "size": [0.35, 0.35] if obj and sample_cup else [0.25, 0.25]}
    return sv


@app.post("/api/rooms/scenes")
async def save_room_scene(body: Dict[str, Any]) -> Any:
    sv = _sv_from_request(body)
    if not sv["scene_id"].startswith("room_"):
        raise HTTPException(422, "请另存为房间场景")
    # 已存在则递增草稿版本（PATCH 语义；刷新后可找回同一场景的最新版）
    try:
        prev = sv_mod.load_scene(sv["scene_id"])
        sv["draft_version"] = int(prev.get("draft_version", 1)) + 1
    except FileNotFoundError:
        pass
    errors = sv_mod.validate(sv)
    if not sv["calibration"].get("scale_check"):
        errors = [err for err in errors if err["code"] != "scale_check_missing"]
    if errors:
        # 校验失败：拒绝保存（碰撞未就绪），返回结构化错误给页面
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)
    if sv["calibration"].get("scale_check"):
        sv = sv_mod.publish(sv)
    else:
        sv["nav_grid"] = sv_mod.build_nav_grid(sv)
    sv_mod.save_draft(sv)
    return {"ok": True, "sceneId": sv["scene_id"],
            "draftVersion": sv["draft_version"], "errors": []}


@app.get("/api/rooms/scenes")
async def list_room_scenes(worldId: Optional[str] = None) -> Any:
    out = []
    for p in sorted(sv_mod.SCENES_DIR.glob("*.json")):
        try:
            s = json.loads(p.read_text())
        except Exception:
            continue
        if worldId and (s.get("world_ref") or {}).get("world_id") != worldId:
            continue
        out.append({"sceneId": s["scene_id"], "name": s.get("name"),
                    "draftVersion": s.get("draft_version"),
                    "worldId": (s.get("world_ref") or {}).get("world_id")})
    return {"scenes": out}


@app.get("/api/rooms/scenes/{scene_id}")
async def get_room_scene(scene_id: str) -> Any:
    return _room_start(scene_id)


@app.post("/api/rooms/verify")
async def verify_room_scene(body: Dict[str, Any]) -> Any:
    """真实 MuJoCo 碰撞验证：子进程 humanoid venv 跑 gym.room_verify。"""
    scene_id = body.get("sceneId")
    if not scene_id:
        raise HTTPException(400, "sceneId 为空")
    if not _ROOM_VERIFY_PYTHON.is_file():
        raise HTTPException(500, f"humanoid venv 不存在: {_ROOM_VERIFY_PYTHON}")
    _room_start(scene_id)
    drive_m = _room_vector([body.get("driveM", 1.0)], 1)[0]
    if not 0 < drive_m <= 4:
        raise HTTPException(422, "碰撞验证距离须在 0 到 4 米之间")
    cmd = [str(_ROOM_VERIFY_PYTHON), "-m", "gym.room_verify",
           "--scene-id", scene_id, "--drive-m", str(drive_m),
           "--seed", str(body.get("seed", 0))]
    if body.get("withProxies") is False:
        cmd.append("--without-proxies")
    return await _run_room_sim(cmd)


async def _run_room_sim(cmd: list[str]) -> Any:
    if not _ROOM_VERIFY_PYTHON.is_file():
        raise HTTPException(500, "MuJoCo 环境不存在，请配置 ROBOTGYM_SIM_PYTHON")
    import subprocess
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(config.BASE_DIR.parent),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=240.0)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(504, "MuJoCo 验证超时（240s）")
    if proc.returncode != 0:
        raise HTTPException(500, f"验证失败: {err.decode()[-400:]}")
    for line in out.decode().splitlines():
        if line.startswith("RUN_SUMMARY "):
            return JSONResponse(json.loads(line[len("RUN_SUMMARY "):]))
    raise HTTPException(500, f"无摘要输出: {out.decode()[-400:]}")


@app.post("/api/rooms/carry")
async def carry_in_room(body: Dict[str, Any]) -> Any:
    scene_id = body.get("sceneId")
    scene = _room_start(scene_id)
    regions = {r["kind"] for r in scene["regions"].values()}
    if not scene["instances"] or not {"pick", "place"}.issubset(regions):
        raise HTTPException(422, "先保存物品的 A 点和放置的 B 点")
    return await _run_room_sim([str(_ROOM_VERIFY_PYTHON), "-m", "gym.room_carry",
                                "--scene-id", scene_id])


# ---------------------------------------------------------------------- #
# Gym 平台 API：批次列表 / 报告 / 单次运行查询（只读，运行走 CLI 或后台任务）
# ---------------------------------------------------------------------- #

GYM_RUNS_DIR = config.BASE_DIR.parent / "runs"


@app.get("/api/gym/batches")
async def gym_batches() -> Any:
    """列出所有 gym 批次（含报告摘要）。"""
    out = []
    if GYM_RUNS_DIR.exists():
        for d in sorted(GYM_RUNS_DIR.iterdir(), reverse=True):
            rp = d / "report.json"
            if rp.is_file():
                try:
                    r = json.loads(rp.read_text())
                    ov = r.get("overall", {})
                    out.append({
                        "batch_id": d.name,
                        "trials": r.get("protocol", {}).get("trials_total"),
                        "grasp": ov.get("fixed", {}).get("grasp_success", 0)
                                + ov.get("feedback", {}).get("grasp_success", 0),
                        "task": ov.get("fixed", {}).get("task_success", 0)
                                + ov.get("feedback", {}).get("task_success", 0),
                        "conditions": r.get("protocol", {}).get("conditions", []),
                    })
                except Exception:
                    continue
    return {"batches": out}


@app.get("/api/gym/batches/{batch_id}")
async def gym_batch_detail(batch_id: str) -> Any:
    d = GYM_RUNS_DIR / batch_id
    rp = d / "report.json"
    if not rp.is_file():
        raise HTTPException(404, f"batch {batch_id} 不存在")
    return json.loads(rp.read_text())


@app.get("/api/gym/batches/{batch_id}/runs/{run_id}")
async def gym_run_detail(batch_id: str, run_id: str) -> Any:
    f = GYM_RUNS_DIR / batch_id / run_id / "run.json"
    if not f.is_file():
        raise HTTPException(404, f"run {run_id} 不存在")
    return json.loads(f.read_text())


# ---------------------------------------------------------------------- #
# 静态文件：GLB / OBJ / 全景图 / 构建好的前端
# ---------------------------------------------------------------------- #

app.mount("/files", StaticFiles(directory=config.DATA_DIR), name="files")

@app.get("/demo/room.html", include_in_schema=False)
async def room_demo_page():
    return FileResponse(_REPO / "room.html")


@app.get("/demo/room-tools.js", include_in_schema=False)
async def room_demo_tools():
    return FileResponse(_REPO / "room-tools.js", media_type="application/javascript")


@app.get("/demo/evidence/S2/splat.spz", include_in_schema=False)
async def room_demo_sample():
    return FileResponse(_REPO / "evidence/S2/splat.spz", media_type="application/octet-stream")


@app.get("/demo/s2-carry.json", include_in_schema=False)
async def room_demo_carry():
    return FileResponse(_REPO / "evidence/room/s2-carry.json", media_type="application/json")


app.mount("/demo/frontend/public/models", StaticFiles(directory=_REPO / "frontend/public/models"), name="room_models")

if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
