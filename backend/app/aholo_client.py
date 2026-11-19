"""Aholo / Lux3D 客户端 —— 严格按官方 OpenAPI 规范实现（docs/aholo/*.json）。

已核实的事实（2026-11-13，以 docs/aholo/ 为准）：
  网关     https://api.aholo3d.cn（国内）；国际站 https://api.aholo.ai
  鉴权     Header: Authorization: <API_KEY>，不加 "Bearer " 前缀
  Asset 上传（Lux3D 只收 URL，本地文件必须先上传）：
    1) GET  {gw}/asset/v1/token                     → { ousToken, globalDomain, blockSize }
       （token 是一次性的：每次上传都要取新 token，复用报 c=12 'Token已被使用'）
    2) POST {globalDomain}/ous/api/v2/single/upload （multipart: file + md5，Header ous-token-v2: <token>）
       文件 > blockSize 时走 /ous/api/v2/block/upload/init|part
    3) GET  {globalDomain}/ous/api/v2/upload/status → d.status==5 成功，取 d.url
  Lux3D    POST /lux3d/v1/generate/img-to-3d/task/create
           body { img, version: G1|G1-Turbo, faceCount?, outputFormat: ["glb"] }
           响应 { c:"0", d: <taskId:int> }（d 直接是数字！）
           并发上限: c=="GENERATION_CONCURRENCY_LIMIT_EXCEEDED"
  查询     GET /lux3d/v1/generate/task/get?taskid=<int>
           { c:"0", d: { taskId, status, outputs:[{content}] } }
           status: 0 初始化 1 运行中 3 成功 4 失败 6 已取消
           未请求格式 content=="NOT_REQUESTED"（不是 URL）
           结果 URL 2h 过期 → 必须立即下载
  World    POST /world/v1/generations { prompt } 或 { prompt, resources:[{url}] }
  mock 模式（AHOLO_MOCK=1）保留为离线兜底，但不算任何验收项。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from . import config

LOG_PATH = config.BASE_DIR.parent / "logs" / "api_usage.jsonl"


class AholoError(Exception):
    pass


def _log_usage(interface: str, task_id: str, elapsed_s: float, result: str) -> None:
    """记录 API 用量（硬性规则 4）。绝不写 key。"""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "interface": interface,
                "taskId": str(task_id),
                "elapsed_s": round(elapsed_s, 1),
                "result": result,
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


class AholoClient:
    """每个实例持有自己的 httpx client，跨事件循环安全（修复问题 #4）。"""

    def __init__(self, timeout: float = 90.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    # 基础
    # ------------------------------------------------------------------ #

    def _headers(self) -> Dict[str, str]:
        if not config.AHOLO_API_KEY:
            raise AholoError("未配置 AHOLO_API_KEY（backend/.env 或环境变量）")
        return {"Authorization": config.AHOLO_API_KEY}

    @staticmethod
    def _check_envelope(resp: httpx.Response) -> Any:
        """解 Aholo 包装层 { c, m, d }：c=="0" 成功；否则抛业务错误。"""
        r = resp.json()
        c = str(r.get("c", "-1"))
        if c != "0":
            raise AholoError(f"Aholo 业务错误 c={c} m={r.get('m', '')!r}")
        return r.get("d")

    # ------------------------------------------------------------------ #
    # Asset 上传：本地文件 → URL
    # ------------------------------------------------------------------ #

    async def _get_ous_token(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """每次上传取新 token（实测 token 是一次性的：复用会报 c=12 'Token已被使用'）。"""
        r = await self._client.get(f"{config.AHOLO_BASE_URL}/asset/v1/token", headers=headers)
        r.raise_for_status()
        d = r.json()
        if not d.get("ousToken"):
            raise AholoError(f"asset token 响应异常: {str(d)[:200]}")
        self._ous_token = {
            "token": d["ousToken"],
            "domain": d.get("globalDomain", "https://ous-cos.kujiale.com"),
            "blockSize": int(d.get("blockSize", 1048576)),
        }
        return self._ous_token

    async def upload_asset(self, data: bytes, filename: str, mime: str = "application/octet-stream") -> str:
        """上传文件到 OUS，轮询到 status==5，返回可访问 URL。"""
        headers = self._headers()
        token_info = await self._get_ous_token(headers)
        ous_h = {"ous-token-v2": token_info["token"]}
        md5 = hashlib.md5(data).hexdigest()

        if len(data) <= token_info["blockSize"]:
            r = await self._client.post(
                f"{token_info['domain']}/ous/api/v2/single/upload",
                headers=ous_h,
                files={"file": (filename, data, mime)},
                data={"md5": md5},
            )
            r.raise_for_status()
            rd = r.json()
            if str(rd.get("c")) != "0":
                raise AholoError(f"OUS 上传失败: {str(rd)[:300]}")
        else:
            # 分片上传（实测：init/part 都是 form 字段；init{md5,size,name,blocks} part{block,file}）
            n_blocks = (len(data) + token_info["blockSize"] - 1) // token_info["blockSize"]
            r = await self._client.post(
                f"{token_info['domain']}/ous/api/v2/block/upload/init",
                headers=ous_h,
                data={"name": filename, "size": str(len(data)), "md5": md5, "blocks": str(n_blocks)},
            )
            r.raise_for_status()
            rd = r.json()
            if str(rd.get("c")) != "0":
                raise AholoError(f"OUS 分片 init 失败: {str(rd)[:300]}")
            init_d = rd.get("d") or {}
            if not init_d.get("deduplicated"):
                for i in range(n_blocks):
                    chunk = data[i * token_info["blockSize"]:(i + 1) * token_info["blockSize"]]
                    pr = await self._client.post(
                        f"{token_info['domain']}/ous/api/v2/block/upload/part",
                        headers=ous_h,
                        files={"file": (filename, chunk, mime)},
                        data={"block": str(i + 1)},
                    )
                    pr.raise_for_status()
                    pd = pr.json()
                    if str(pd.get("c")) != "0":
                        raise AholoError(f"OUS 分片上传失败 block{i + 1}: {str(pd)[:200]}")
            # 上传完成后轮询 status 收口（无独立 complete 接口）

        # 轮询上传状态直到 status==5
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60.0:
            sr = await self._client.get(f"{token_info['domain']}/ous/api/v2/upload/status", headers=ous_h)
            sr.raise_for_status()
            sd = sr.json()
            if str(sd.get("c")) != "0":
                raise AholoError(f"OUS status 查询失败: {str(sd)[:300]}")
            d = sd.get("d") or {}
            status = d.get("status")
            if status == 5:
                url = d.get("url")
                if not url:
                    raise AholoError(f"OUS 上传成功但无 url: {str(d)[:300]}")
                return url
            if status in (6, 8):
                raise AholoError(f"OUS 上传失败/中止 status={status}: {d.get('errorMsg', '')}")
            await asyncio.sleep(0.5)
        raise AholoError("OUS 上传状态轮询超时（60s）")

    # ------------------------------------------------------------------ #
    # Lux3D 图生 3D
    # ------------------------------------------------------------------ #

    async def img_to_3d(
        self,
        image_bytes: bytes,
        filename: str,
        preset_asset_id: Optional[str] = None,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
    ) -> Dict[str, Any]:
        """返回 { glb_bytes, zip_bytes?, task_id, model }。mock 模式仅供离线演示。"""
        if config.AHOLO_MOCK:
            return await self._mock_img_to_3d(preset_asset_id, on_progress)
        return await self._real_img_to_3d(image_bytes, filename, on_progress)

    async def _mock_img_to_3d(self, preset_asset_id: Optional[str], on_progress) -> Dict[str, Any]:
        task_id = f"mock_{uuid.uuid4().hex[:8]}"
        for i, frac in enumerate((0.1, 0.35, 0.7, 1.0)):
            await asyncio.sleep(1.2)
            if on_progress:
                stage = ("排队中", "生成中", "生成中", "完成")[i]
                on_progress(task_id, frac, stage)
        preset = config.ASSETS_DIR / (preset_asset_id or "preset_mug") / "visual.glb"
        if not preset.exists():
            raise AholoError(f"mock 预置资产缺失: {preset}")
        return {"glb_bytes": preset.read_bytes(), "zip_bytes": None, "task_id": task_id, "model": config.LUX3D_MODEL}

    async def _real_img_to_3d(self, image_bytes: bytes, filename: str, on_progress) -> Dict[str, Any]:
        headers = self._headers()

        # 1. 上传图片拿 URL
        t_upload = time.monotonic()
        if on_progress:
            on_progress("", 0.02, "上传图片")
        image_url = await self.upload_asset(image_bytes, filename, mime="image/jpeg")
        _log_usage("asset.upload", "-", time.monotonic() - t_upload, "ok")

        # 2. 创建 img-to-3d 任务。规范：d 直接是 taskId 数字
        t_create = time.monotonic()
        if on_progress:
            on_progress("", 0.06, "创建任务")
        r = await self._client.post(
            f"{config.AHOLO_BASE_URL}/lux3d/v1/generate/img-to-3d/task/create",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "img": image_url,
                "version": config.LUX3D_MODEL,          # G1 / G1-Turbo
                "faceCount": config.LUX3D_FACE_COUNT,
                "outputFormat": ["glb"],
            },
        )
        r.raise_for_status()
        d = self._check_envelope(r)  # d == taskId (int)
        task_id = str(d)
        _log_usage("lux3d.img_to_3d.create", task_id, time.monotonic() - t_create, "ok")

        # 3. 轮询（统一 task/get；status==3 成功）
        t_poll = time.monotonic()
        outputs = await self._poll_task(task_id, headers, on_progress)
        _log_usage("lux3d.img_to_3d.poll", task_id, time.monotonic() - t_poll, "ok")

        # 4. 立即下载（URL 2h 过期）。NOT_REQUESTED 不是 URL
        if on_progress:
            on_progress(task_id, 0.9, "下载产物")
        glb_bytes: Optional[bytes] = None
        zip_bytes: Optional[bytes] = None
        for out in outputs:
            content = (out or {}).get("content") or ""
            if not content or content == "NOT_REQUESTED" or not content.startswith("http"):
                continue
            low = content.lower().split("?")[0]
            data = await self._download(content)
            if low.endswith(".glb"):
                glb_bytes = data
            elif low.endswith(".zip"):
                zip_bytes = zip_bytes or data
        if glb_bytes is None:
            raise AholoError(f"任务 {task_id} 产物中没有 GLB: {[o.get('content') for o in outputs]}")
        if on_progress:
            on_progress(task_id, 1.0, "完成")
        return {"glb_bytes": glb_bytes, "zip_bytes": zip_bytes, "task_id": task_id, "model": config.LUX3D_MODEL}

    async def _poll_task(self, task_id: str, headers: Dict[str, str],
                         on_progress=None, timeout_s: float = 900.0) -> List[Dict[str, Any]]:
        """轮询 GET /lux3d/v1/generate/task/get?taskid=，直到成功/失败/取消/超时。"""
        t0 = time.monotonic()
        last_status: Any = None
        while True:
            if time.monotonic() - t0 > timeout_s:
                raise AholoError(f"Lux3D 任务 {task_id} 轮询超时（{timeout_s:.0f}s，最后状态 {last_status}）")
            r = await self._client.get(
                f"{config.AHOLO_BASE_URL}/lux3d/v1/generate/task/get",
                headers=headers, params={"taskid": task_id},
            )
            r.raise_for_status()
            try:
                d = self._check_envelope(r)
            except AholoError as e:
                # 任务可能尚未入库（创建后立刻查），短暂重试
                if "c=-1" in str(e) and time.monotonic() - t0 < 30:
                    await asyncio.sleep(2.0)
                    continue
                raise
            status = d.get("status")
            if status != last_status:
                last_status = status
                if on_progress:
                    on_progress(task_id, 0.4, f"Lux3D 状态 {status}")
            if status == 3:
                outputs = d.get("outputs") or []
                if not outputs:
                    raise AholoError(f"任务 {task_id} 成功但 outputs 为空")
                return outputs
            if status == 4:
                raise AholoError(f"Lux3D 任务 {task_id} 失败 status=4: {str(d)[:300]}")
            if status == 6:
                raise AholoError(f"Lux3D 任务 {task_id} 已取消 status=6")
            await asyncio.sleep(3.0)

    async def _download(self, url: str) -> bytes:
        r = await self._client.get(url)
        r.raise_for_status()
        return r.content

    # ------------------------------------------------------------------ #
    # World：文字/图片 → 全景图（V2 用）
    # ------------------------------------------------------------------ #

    async def world_generate(self, image_bytes: Optional[bytes] = None, filename: str = "room.jpg",
                             prompt: str = "") -> Dict[str, Any]:
        if config.AHOLO_MOCK:
            pano = config.WORLDS_DIR / "mock_pano.jpg"
            return {"pano_bytes": pano.read_bytes() if pano.exists() else None,
                    "task_id": f"mock_{uuid.uuid4().hex[:8]}"}
        return await self._real_world(image_bytes, filename, prompt)

    async def _real_world(self, image_bytes: Optional[bytes], filename: str, prompt: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {"prompt": prompt or "一个明亮的客厅，家具齐全，俯视可见桌面"}
        if image_bytes:
            image_url = await self.upload_asset(image_bytes, filename,
                                                mime=mimetypes.guess_type(filename)[0] or "image/jpeg")
            body["resources"] = [{"url": image_url}]
        return await self._submit_world("generations", body)

    async def world_reconstruct(self, photos: list[Path], name: str, on_created=None) -> Dict[str, Any]:
        if len(photos) < 20:
            raise AholoError("多图重建需要至少 20 张同房间照片")
        if config.AHOLO_MOCK:
            raise AholoError("当前为离线模式。使用已生成样例，或配置真实 World 接口。")
        self._headers()
        slots = asyncio.Semaphore(4)

        async def upload(photo: Path) -> dict:
            async with slots:
                url = await self.upload_asset(photo.read_bytes(), photo.name,
                                              mime=mimetypes.guess_type(photo.name)[0] or "image/jpeg")
                return {"url": url, "type": "image"}

        resources = await asyncio.gather(*(upload(photo) for photo in photos))
        return await self._submit_world("reconstructions", {
            "name": name, "resources": resources, "scene": "space", "taskQuality": "high",
        }, on_created=on_created)

    async def _submit_world(self, action: str, body: Dict[str, Any], on_created=None) -> Dict[str, Any]:
        headers = self._headers()
        t0 = time.monotonic()
        r = await self._client.post(
            f"{config.AHOLO_BASE_URL}/world/v1/{action}",
            headers={**headers, "Content-Type": "application/json"},
            json=body,
        )
        if r.is_error:
            try:
                info = r.json()
                reason = info.get("message") or info.get("error") or info.get("status")
            except ValueError:
                reason = None
            raise AholoError(f"World 请求 HTTP {r.status_code}: {reason or '提供方未返回原因'}")
        r.raise_for_status()
        # World 响应无 {c,d} 包装：直接 { worldId } 或 ApiError
        wd = r.json()
        world_id = wd.get("worldId") or wd.get("id")
        if not world_id:
            raise AholoError(f"World 创建失败: {str(wd)[:300]}")
        if on_created is not None:
            on_created(str(world_id))
        return await self.wait_world(str(world_id), action, t0)

    async def wait_world(self, world_id: str, action: str, t0: Optional[float] = None) -> Dict[str, Any]:
        headers = self._headers()
        t0 = time.monotonic() if t0 is None else t0
        deadline_s = 3600 if action == "reconstructions" else 1200
        while True:
            if time.monotonic() - t0 > deadline_s:
                raise AholoError(f"World 任务 {world_id} 轮询超时")
            r = await self._client.get(f"{config.AHOLO_BASE_URL}/world/v1/{world_id}", headers=headers)
            if r.status_code == 404:
                await asyncio.sleep(5.0)
                continue
            r.raise_for_status()
            d = r.json()
            if d.get("code") == 404 or d.get("status") == "NOT_FOUND":
                await asyncio.sleep(5.0)
                continue
            status = d.get("status")
            assets = d.get("assets") or {}
            pano_url = (assets.get("imagery") or {}).get("panoUrl")
            urls = (assets.get("splats") or {}).get("urls") or {}
            splat_url = urls.get("spzPath") or urls.get("plyPath")
            if status in (4, "4", 6, "6", "FAILED", "CANCELLED", "CANCELED"):
                detail = d.get("message") or d.get("error") or "提供方没有返回具体原因"
                raise AholoError(f"World {world_id} {status}: {detail}")
            if splat_url:
                splat = await self._download(splat_url)
                pano = await self._download(pano_url) if pano_url else None
                _log_usage("world." + action, world_id, time.monotonic() - t0, "ok")
                return {"pano_bytes": pano, "splat_bytes": splat,
                        "splat_extension": ".spz" if urls.get("spzPath") else ".ply",
                        "task_id": world_id, "world_id": world_id, "assets": assets}
            await asyncio.sleep(5.0)

    # ------------------------------------------------------------------ #
    # Lux3D 部件拆分 / 关节化（V6 用）
    # ------------------------------------------------------------------ #

    async def part_split(self, glb_bytes: bytes, filename: str = "model.glb") -> str:
        """GLB → 部件拆分（同一账号同时只能一个任务）。返回任务 taskId。"""
        headers = self._headers()
        glb_url = await self.upload_asset(glb_bytes, filename, mime="model/gltf-binary")
        r = await self._client.post(
            f"{config.AHOLO_BASE_URL}/lux3d/v1/part-split/task/create",
            headers={**headers, "Content-Type": "application/json"},
            json={"glbUrl": glb_url},
        )
        r.raise_for_status()
        d = self._check_envelope(r)
        task_id = str(d)
        _log_usage("lux3d.part_split.create", task_id, 0, "ok")
        return task_id

    async def articulation_animation(self, glb_bytes: bytes, prompt: str,
                                     filename: str = "model.glb") -> str:
        """GLB + prompt → 关节化与动画。返回任务 taskId。"""
        headers = self._headers()
        glb_url = await self.upload_asset(glb_bytes, filename, mime="model/gltf-binary")
        r = await self._client.post(
            f"{config.AHOLO_BASE_URL}/lux3d/v1/articulation-animation/task/create",
            headers={**headers, "Content-Type": "application/json"},
            json={"glbUrl": glb_url, "prompt": prompt},
        )
        r.raise_for_status()
        d = self._check_envelope(r)
        task_id = str(d)
        _log_usage("lux3d.articulation.create", task_id, 0, "ok")
        return task_id

    async def wait_task(self, task_id: str) -> List[Dict[str, Any]]:
        """对外暴露的通用等待（V6 等异步任务用）。"""
        return await self._poll_task(task_id, self._headers())

    async def multimodal_to_image(self, image_bytes: bytes, prompt: str,
                                  filename: str = "ref.jpg") -> str:
        """多模态单图生成（B6 材质参考图用）。返回图片 URL（2h 过期，调用方须立即下载）。"""
        headers = self._headers()
        img_url = await self.upload_asset(image_bytes, filename, mime="image/jpeg")
        r = await self._client.post(
            f"{config.AHOLO_BASE_URL}/lux3d/v1/generate/multimodal-to-image/task/create",
            headers={**headers, "Content-Type": "application/json"},
            json={"img": img_url, "prompt": prompt},
        )
        r.raise_for_status()
        d = self._check_envelope(r)
        task_id = str(d)
        outputs = await self._poll_task(task_id, headers)
        content = (outputs[0] or {}).get("content") or ""
        if not content.startswith("http"):
            raise AholoError(f"multimodal-to-image 结果异常: {content[:200]}")
        _log_usage("lux3d.multimodal_to_image", task_id, 0, "ok")
        return content

    async def material_transfer(self, glb_bytes: bytes, ref_image_url: str,
                                filename: str = "model.glb") -> Dict[str, Any]:
        """模型材质重绘（B6 用）。返回 { glb_bytes, task_id }。"""
        headers = self._headers()
        glb_url = await self.upload_asset(glb_bytes, filename, mime="model/gltf-binary")
        r = await self._client.post(
            f"{config.AHOLO_BASE_URL}/lux3d/v1/generate/material-transfer/task/create",
            headers={**headers, "Content-Type": "application/json"},
            json={"glbUrl": glb_url, "refImageUrl": ref_image_url, "outputFormat": ["glb"]},
        )
        r.raise_for_status()
        d = self._check_envelope(r)
        task_id = str(d)
        outputs = await self._poll_task(task_id, headers)
        glb_bytes = None
        for out in outputs:
            content = (out or {}).get("content") or ""
            if content.startswith("http") and content.lower().split("?")[0].endswith(".glb"):
                glb_bytes = await self._download(content)
        if glb_bytes is None:
            raise AholoError(f"material-transfer 任务 {task_id} 无 GLB 产物")
        _log_usage("lux3d.material_transfer", task_id, 0, "ok")
        return {"glb_bytes": glb_bytes, "task_id": task_id}
