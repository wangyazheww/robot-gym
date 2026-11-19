"""mock 全景图生成（AHOLO World 未接入时的占位厨房背景，P0 天空盒用）。

用法: cd backend && python scripts/make_pano.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

W, H = 2048, 1024

# equirect 纵向条带：(v 从 0 顶部到 1 底部) → (RGB, 结束v)
BANDS = [
    ((206, 224, 235), 0.28),   # 天花/上墙
    ((233, 224, 210), 0.52),   # 墙面
    ((196, 178, 160), 0.60),   # 台面背板
    ((172, 160, 150), 0.68),   # 台面
    ((146, 130, 118), 0.74),   # 柜体
    ((188, 170, 150), 1.00),   # 木地板
]


def main() -> None:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    v0 = 0.0
    for rgb, v1 in BANDS:
        y0, y1 = int(v0 * H), int(v1 * H)
        for y in range(y0, y1):
            t = (y - y0) / max(y1 - y0, 1)
            shade = 1.0 - 0.10 * t  # 轻微渐变
            img[y, :] = np.clip(np.array(rgb) * shade, 0, 255)
        v0 = v1
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    # 简单"窗户"和"橱柜竖缝"，避免完全空洞
    for x in range(200, W, 340):
        d.rectangle([x, int(0.30 * H), x + 150, int(0.50 * H)], fill=(215, 232, 240), outline=(180, 190, 200))
    for x in range(120, W, 260):
        d.line([x, int(0.60 * H), x, int(0.72 * H)], fill=(120, 105, 95), width=3)
    rng = np.random.default_rng(7)
    img = np.array(im, dtype=np.int16)
    img += rng.integers(-6, 6, img.shape, dtype=np.int16)
    out = config.WORLDS_DIR / "mock_pano.jpg"
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out, quality=88)
    print(f"  {out}")


if __name__ == "__main__":
    main()
