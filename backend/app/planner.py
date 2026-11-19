"""Planner（PRD F4）：自然语言 → 技能序列。

规则引擎（默认，离线稳定）：覆盖演示常用句式 —— 把X放进Y / 把X放到Y旁边 /
推X到... / 打开抽屉。物体指代用类别同义词表 + 颜色词消歧。
LLM 模式（配置 LLM_API_KEY 后启用）：OpenAI 兼容 tool calling，输出同一 JSON
结构；任何异常都回落到规则引擎（黑客松现场兜底）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from . import config

# 类别 → 中文同义词（用于从指令中找物体）
SYNONYMS: Dict[str, List[str]] = {
    "mug": ["马克杯", "杯子", "茶杯", "水杯", "杯"],
    "carton": ["牛奶盒", "奶盒", "牛奶", "盒子"],
    "plate": ["盘子", "碟子", "盘", "碟"],
    "toy": ["小黄鸭", "鸭子", "鸭", "玩具"],
    "block": ["积木", "方块", "色块"],
    "bottle": ["瓶子", "瓶"],
}
COLORS = ["红", "蓝", "绿", "黄", "白", "黑", "银"]

def _find_object(scene: Dict[str, Any], text: str, exclude: str = "") -> Optional[str]:
    """在指令文本里找物体 → 返回 assetId。优先：名字含颜色词的最长匹配。"""
    best: tuple[int, Optional[str]] = (0, None)
    for obj in scene.get("objects", []):
        if obj["id"] == exclude:
            continue
        name = obj.get("name", "")
        category = obj.get("category", "unknown")
        cands = SYNONYMS.get(category, []) + [name]
        score = 0
        for w in cands:
            if w and w in text:
                score = max(score, len(w))
        # 颜色消歧：指令里的颜色词与物体名颜色一致加分，名字是别的颜色则减分
        for c in COLORS:
            if c in text:
                if c in name:
                    score += 2
                elif any(w in name for w in COLORS):
                    score -= 1
        if score > best[0]:
            best = (score, obj["id"])
    return best[1]


def _find_place_target(scene: Dict[str, Any], text: str, exclude_obj: str) -> Optional[Dict[str, Any]]:
    obj = _find_object(scene, text, exclude=exclude_obj)
    if obj:
        return {"id": obj}
    if re.search(r"桌子中间|桌中央|桌面中间", text):
        return {"point": "table_center"}
    if re.search(r"桌子左边|桌面左边", text):
        return {"point": "table_left"}
    if re.search(r"桌子右边|桌面右边", text):
        return {"point": "table_right"}
    if re.search(r"桌子(上|面)|桌上", text):
        return {"point": "table_center"}
    m = re.search(r"(.{1,4}?)(旁边|边上|附近)", text)
    if m:
        anchor = _find_object(scene, m.group(1))
        if anchor:
            return {"id": anchor, "offset": 0.15}
    return None


def rule_plan(scene: Dict[str, Any], instruction: str) -> Dict[str, Any]:
    text = instruction.strip()
    objects = {o["id"]: o for o in scene.get("objects", [])}

    # ---- 打开类（关节物体，P2）----
    if "打开" in text or "拉开" in text:
        art = [o for o in objects.values() if o.get("articulated")]
        if art:
            o = art[0]
            return _ok("检测到关节物体，执行打开动作。",
                       [{"skill": "open", "args": {"object": o["id"]}}, {"skill": "home", "args": {}}],
                       {"type": "articulation_opened", "object": o["id"]})
        return _refuse("当前场景里没有可打开的关节物体（P2 彩蛋需要 Lux3D 关节化生成的柜子/抽屉）。")

    # ---- 推类（P1）----
    if "推" in text:
        obj = _find_object(scene, text)
        if not obj:
            return _refuse(f"没有在场景里找到「{text}」里提到的物体。")
        dest = _find_place_target(scene, text, exclude_obj=obj)
        if not dest:
            return _refuse("推到哪里？请说「推到盘子旁边」「推到桌子中间」等。")
        target = dest["id"] if "id" in dest else dest["point"]
        steps = [{"skill": "push", "args": {"object": obj, "target": target}}]
        return _ok(f"识别为推动任务：推 {objects[obj]['name']}。", steps,
                   {"type": "object_moved", "object": obj, "dist_m": 0.1})

    # ---- 抓放类 ----
    obj = _find_object(scene, text)
    if not obj:
        return _refuse(f"场景里没有「{text}」提到的物体，可以先把物品照片上传生成。")

    if re.search(r"抓|拿起|举", text) and not re.search(r"放|进|到|给", text):
        return _ok(f"识别为抓取任务：抓起 {objects[obj]['name']}。",
                   [{"skill": "pick", "args": {"object": obj}}],
                   {"type": "picked", "object": obj})

    dest = _find_place_target(scene, text, exclude_obj=obj)
    if not dest:
        return _refuse("听不懂目标位置，试试「把红色杯子放进盘子里」。")

    label = objects[dest["id"]]["name"] if "id" in dest else "桌面目标点"
    steps = [
        {"skill": "pick", "args": {"object": obj}},
        {"skill": "place", "args": {"target": dest["id"] if "id" in dest else dest["point"],
                                    "offset": dest.get("offset")}},
        {"skill": "home", "args": {}},
    ]
    reasoning = f"用户说「{text}」：先抓起 {objects[obj]['name']}，再放到 {label} 处，最后回到初始位。"
    cond = {"type": "object_in_region",
            "object": obj,
            "region": dest["id"] if "id" in dest else dest["point"]}
    if "offset" in dest:
        cond["offset"] = dest["offset"]
    return _ok(reasoning, steps, cond)


def _ok(reasoning: str, steps: List[Dict[str, Any]], success_condition: Dict[str, Any]) -> Dict[str, Any]:
    return {"reasoning": reasoning, "steps": steps, "success_condition": success_condition, "refused": False}


def _refuse(message: str) -> Dict[str, Any]:
    return {"reasoning": message, "steps": [], "success_condition": None, "refused": True, "message": message}


# ---------------------------------------------------------------------- #
# LLM tool calling（可选）
# ---------------------------------------------------------------------- #

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "execute_task",
        "description": "让机械臂在物理仿真中执行任务",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "一句话推理"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skill": {"type": "string", "enum": ["pick", "place", "push", "open", "home"]},
                            "args": {"type": "object"},
                        },
                        "required": ["skill", "args"],
                    },
                },
                "success_condition": {"type": "object"},
                "refused": {"type": "boolean"},
            },
            "required": ["reasoning", "steps", "refused"],
        },
    },
}]


async def llm_plan(scene: Dict[str, Any], instruction: str) -> Optional[Dict[str, Any]]:
    if not config.LLM_API_KEY:
        return None
    import httpx
    brief = {"objects": [{k: o.get(k) for k in ("id", "name", "category")} for o in scene.get("objects", [])]}
    valid_ids = {o["id"] for o in scene.get("objects", [])} | {"table_center", "table_left", "table_right"}
    sys = (
        "你是机械臂任务规划器。把中文指令转成技能序列 JSON。"
        "可用技能：pick(object=物体id) / place(target=物体id或table_center/table_left/table_right) / "
        "push(object,target) / open(object) / home()。"
        "场景物体（用户会用名字或省略说法指代，如「香蕉」=name 含香蕉的物体，「杯子」=name 含杯子的物体）：\n"
        + json.dumps(brief, ensure_ascii=False) +
        "\n规则："
        "(1) 指令里的物体按名字模糊匹配场景物体的 name（子串即可，如「香蕉」匹配「纳米香蕉摆件」）；"
        "(2) 匹配不到任何场景物体才 refused=true；"
        "(3) 「放到X旁边」= pick(object) 后 place(target=X物体id, offset=0.15)；"
        "(4)「放到盘子/桌面上」= place(target=盘子id 或 table_center)；"
        "(5) 推类：push(object=物体id, target=目标id或table_center)；"
        "(6) place/push 的 target/object 必须是场景物体 id 或 table_center/table_left/table_right，不能是名字；"
        "(7) 成功时输出 {\"reasoning\":一句话,\"steps\":[{\"skill\":...,\"args\":{...}}],"
        "\"success_condition\":{\"type\":\"object_in_region\",\"object\":物体id,\"region\":目标id或点},\"refused\":false}"
        "；拒绝时输出 {\"reasoning\":原因,\"refused\":true,\"steps\":[]}。"
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{config.LLM_BASE_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                json={
                    "model": config.LLM_MODEL,
                    "messages": [{"role": "system", "content": sys},
                                 {"role": "user", "content": instruction}],
                    "tools": _TOOLS,
                    "tool_choice": {"type": "function", "function": {"name": "execute_task"}},
                    "temperature": 0.1,
                },
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        # 校验：任何步骤引用了场景中不存在的物体 id → 整体拒绝（Bug #7：不得静默改 None）
        scene_ids = {o["id"] for o in scene.get("objects", [])}
        for s in args.get("steps", []):
            a = s.get("args", {})
            for key in ("object", "target"):
                v = a.get(key)
                if isinstance(v, str) and v not in valid_ids and v not in scene_ids:
                    return _refuse(
                        f"LLM 规划引用了场景中不存在的物体「{v}」，已拦截（执行层不应收到无效 id）。")
        args.setdefault("refused", False)
        args.setdefault("success_condition", None)
        return args
    except Exception:
        return None


async def plan(scene: Dict[str, Any], instruction: str) -> Dict[str, Any]:
    """LLM 优先（A4 要求）；LLM 不可用/异常时规则引擎兜底。"""
    llm = await llm_plan(scene, instruction)
    if llm is not None:
        if llm.get("refused"):
            llm.setdefault("message", llm.get("reasoning", ""))
            return llm
        if llm.get("steps"):
            llm["reasoning"] = (llm.get("reasoning") or "") + "（LLM 规划）"
            return llm
    return rule_plan(scene, instruction)
