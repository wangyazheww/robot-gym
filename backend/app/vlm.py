"""VLM：物体识别 + 真实尺寸/质量估计（PRD F1.3）。

mock 模式：基于文件名关键词给出合理默认值（演示够用、离线稳定）。
real 模式：OpenAI 兼容 vision 接口（GLM-4V / GPT-4o / Qwen-VL），输出 JSON。
"""
from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict

import httpx

from . import config

# mock 兜底表：关键词 → (名称, 类别, 尺寸cm[W,D,H], 质量g, 摩擦系数)
_HEURISTICS = [
    (("杯", "mug", "cup"), ("红色马克杯", "mug", (9, 9, 10), 300, 0.7)),
    (("奶", "milk", "carton"), ("牛奶盒", "carton", (7, 7, 19), 350, 0.6)),
    (("鸭", "duck", "玩具", "toy"), ("小黄鸭", "toy", (8, 8, 9), 80, 0.5)),
    (("盘", "plate"), ("白瓷盘", "plate", (20, 20, 3), 400, 0.8)),
    (("积木", "block"), ("蓝色积木", "block", (5, 5, 5), 60, 0.6)),
]

_SIZE_RE = re.compile(r"(\d{1,3})\s*[xX×*]\s*(\d{1,3})\s*[xX×*]\s*(\d{1,3})")


def mock_estimate(filename: str) -> Dict[str, Any]:
    low = filename.lower()
    for keys, hit in _HEURISTICS:
        if any(k in low for k in keys):
            name, category, size, mass, friction = hit
            return {"name": name, "category": category,
                    "size_cm": list(size), "mass_g": mass, "friction": friction}
    return {"name": "神秘物体", "category": "unknown",
            "size_cm": [8, 8, 10], "mass_g": 200, "friction": 0.6}


async def estimate(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    if config.VLM_MOCK or not config.VLM_API_KEY:
        return mock_estimate(filename)

    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        "看这张物品照片，只输出 JSON："
        '{"name":"带颜色/特征的中文指代名,如 红色马克杯","category":"英文类别",'
        '"size_cm":[宽,深,高],"mass_g":估计质量克数,"friction":0到1,'
        '"material":"主要接触面，只选 plastic/wood/metal/glass_ceramic/rubber/unknown"}'
    )
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{config.VLM_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.VLM_API_KEY}"},
            json={
                "model": config.VLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "temperature": 0.2,
            },
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        data.setdefault("name", "物体")
        data.setdefault("category", "unknown")
        data.setdefault("size_cm", [8, 8, 10])
        data.setdefault("mass_g", 200)
        data.setdefault("friction", 0.6)
        if data.get("material") in {"plastic", "wood", "metal", "glass_ceramic", "rubber"}:
            data["material_source"] = "vlm"
        return data
    except (ValueError, KeyError):
        return mock_estimate(filename)
