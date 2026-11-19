"""全局配置：全部由环境变量驱动，默认 mock 模式可离线跑通全链路。

支持 backend/.env（KEY=VALUE，每行一条，# 注释；已存在的环境变量优先）。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # backend/
DATA_DIR = Path(os.environ.get("RG_DATA_DIR", BASE_DIR / "data"))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:   # 真实环境变量优先于 .env
            os.environ[k] = v


_load_dotenv(BASE_DIR / ".env")

ASSETS_DIR = DATA_DIR / "assets"
WORLDS_DIR = DATA_DIR / "worlds"
SCENES_DIR = DATA_DIR / "scenes"
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

# ---- Aholo / Lux3D ----
# 默认 mock：不上传、不轮询，直接复用预置资产，保证演示稳定。
# 真实模式：AHOLO_MOCK=0，key 写在 backend/.env（AHOLO_API_KEY=...，不加 Bearer 前缀）。
# 官方规范：域名 api.aholo3d.cn；鉴权 key 直接写；图生3D先传 Asset 拿 URL 再创建任务；
#           响应包 {c:"0", d:{...}}；status==3 成功；outputs 按 [ZIP, GLB] 排序。
AHOLO_MOCK = os.environ.get("AHOLO_MOCK", "1") == "1"
AHOLO_API_KEY = os.environ.get("AHOLO_API_KEY", "")
AHOLO_BASE_URL = os.environ.get("AHOLO_BASE_URL", "https://api.aholo3d.cn")
LUX3D_MODEL = os.environ.get("LUX3D_MODEL", "G1-Turbo")  # G1 / G1-Turbo
LUX3D_FACE_COUNT = int(os.environ.get("LUX3D_FACE_COUNT", "300000"))
# 结果 URL 2 小时过期（PRD §9），一切产物必须立即落盘
ARTIFACT_TTL_S = 2 * 3600

# ---- VLM（物体识别 / 尺寸估计）----
VLM_MOCK = os.environ.get("VLM_MOCK", "1") == "1"
# OpenAI 兼容接口（智谱 GLM-4V、GPT-4o、Qwen-VL 均可）
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
VLM_MODEL = os.environ.get("VLM_MODEL", "glm-4v-flash")

# ---- LLM Planner ----
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-4-flash")

for d in (ASSETS_DIR, WORLDS_DIR, SCENES_DIR):
    d.mkdir(parents=True, exist_ok=True)
