"""A4 验收：test_planner.py —— 8 条指令，至少 7/8 正确。

含两类必备场景：
  - 提到场景里不存在的物体 → 必须拒绝
  - 用新上传物体的名字指代（名字不在同义词表里）→ 必须正确匹配
有 LLM key 时走真实 LLM（LLM 优先）；没有时回落规则引擎（分数照样统计，
报告里会注明用的是哪条路径）。
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("AHOLO_MOCK", "1")

from app import planner  # noqa: E402
from app import config  # noqa: E402

# 场景：两个常规物体 + 一个"新上传"物体（名字不含在任何同义词表）
SCENE = {
    "objects": [
        {"id": "obj_1", "name": "白色马克杯", "category": "mug"},
        {"id": "obj_2", "name": "白瓷盘", "category": "plate"},
        {"id": "obj_3", "name": "纳米香蕉摆件", "category": "unknown"},  # 新上传，不在同义词表
    ]
}

# (指令, 期望: refused?, 期望 pick 的物体 id, 备注)
CASES = [
    ("把白色马克杯放进盘子里", False, "obj_1", "常规抓放"),
    ("抓起纳米香蕉摆件", False, "obj_3", "新上传物体按名字指代"),
    ("把纳米香蕉摆件放到盘子旁边", False, "obj_3", "新上传物体 + 旁边放置"),
    ("抓起香蕉", False, "obj_3", "模糊名字匹配到香蕉摆件"),
    ("抓起电饭煲", True, None, "场景里不存在 → 拒绝"),
    ("把冰箱打开", True, None, "场景里不存在且无关节物体 → 拒绝"),
    ("把杯子推到桌子中间", False, "obj_1", "推动 + 桌面点"),
    ("打开柜子", True, None, "场景无关节物体 → 拒绝"),
]


def _pick_id(result) -> Optional[str]:
    for s in result.get("steps", []):
        if s.get("skill") in ("pick", "push"):   # push 也操作主物体
            return s["args"].get("object")
    return None


@pytest.mark.parametrize("instruction,expect_refuse,expect_pick,note", CASES)
def test_planner(instruction, expect_refuse, expect_pick, note):
    result = asyncio.run(planner.plan(SCENE, instruction))
    used_llm = "LLM 规划" in (result.get("reasoning") or "")
    ok = bool(result.get("refused")) == expect_refuse
    if not expect_refuse and expect_pick:
        ok = ok and (_pick_id(result) == expect_pick)
    assert ok, f"[{note}] used_llm={used_llm} result={json.dumps(result, ensure_ascii=False)[:300]}"


def test_summary():
    """汇总 8 条结果供报告引用（跑法：pytest tests/test_planner.py::test_summary -s）"""
    n_ok = 0
    detail = []
    for instr, er, ep, note in CASES:
        r = asyncio.run(planner.plan(SCENE, instr))
        ok = bool(r.get("refused")) == er and (er or _pick_id(r) == ep)
        n_ok += ok
        detail.append({"instruction": instr, "ok": ok, "refused": r.get("refused"),
                       "pick": _pick_id(r), "via": "LLM" if "LLM 规划" in (r.get("reasoning") or "") else "rules"})
    mode = "llm(gpt-4o-mini)" if config.LLM_API_KEY else "rules"
    out = {"correct": n_ok, "total": len(CASES), "mode": mode, "detail": detail}
    (BACKEND.parent / "evidence" / "A4").mkdir(parents=True, exist_ok=True)
    (BACKEND.parent / "evidence" / "A4" / "planner_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    assert n_ok >= 7, f"仅 {n_ok}/8 正确（要求 ≥7）"
