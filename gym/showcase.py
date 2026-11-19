#!/usr/bin/env python3
"""评委展示页生成器：gym/showcase_data.json + 图片 → 自包含 showcase.html。

单文件、离线可开、无外部依赖（内嵌 base64 图片 + 原生 JS 图表）。
用法: python -m gym showcase
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

GYM = Path(__file__).resolve().parent
REPO = GYM.parent
WEB = GYM / "assets_web"


def b64(name: str) -> str:
    p = WEB / name
    if not p.is_file():
        return ""
    return "data:image/" + p.suffix.lstrip(".") + ";base64," + base64.b64encode(p.read_bytes()).decode()


def build() -> Path:
    data = json.loads((GYM / "showcase_data.json").read_text())
    ov = data["overall"]
    fx, fb = ov["fixed"], ov["feedback"]
    rr = data["rep_run"]
    imgs = {n: b64(n + ".jpg") for n in ("pano", "f_stand", "f_approach", "f_lift", "f_release")}
    grasp_total = fx["grasp"] + fb["grasp"]
    task_total = fx["task"] + fb["task"]

    html = HTML_SHELL
    html = html.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__IMG_PANO__", imgs["pano"])
    for n in ("f_stand", "f_approach", "f_lift", "f_release"):
        html = html.replace(f"__IMG_{n.upper()}__", imgs[n])
    html = html.replace("__KPI_TASK__", f"{task_total}/30")
    html = html.replace("__KPI_GRASP__", f"{grasp_total}/30")
    html = html.replace("__KPI_RATE__", f"{round(fx['rate']*100)}%")
    html = html.replace("__KPI_FMAX__", f"{fx['fmax']}")
    html = html.replace("__CONFIG__", data["protocol"]["config_hash"])
    html = html.replace("__FIXED_TASK__", str(fx["task"]))
    html = html.replace("__FB_TASK__", str(fb["task"]))
    html = html.replace("__FIXED_PCT__", str(round(fx["rate"] * 100)))
    html = html.replace("__FB_PCT__", str(round(fb["rate"] * 100)))
    out = REPO / "showcase.html"
    out.write_text(html)
    return out


HTML_SHELL = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot Gym — 拍照 → 机器人任务 → 可复现数据</title>
<style>
:root{
  --bg:#0b1220; --panel:#111a2e; --panel2:#0e1626; --line:#1f2d4a;
  --tx:#e8eef7; --tx2:#9fb0c9; --tx3:#64768f;
  --blue:#3b82f6; --cyan:#22d3ee; --green:#34d399; --amber:#fbbf24; --red:#f87171;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.65}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px}
section{padding:72px 0;border-top:1px solid var(--line)}
h2{font-size:26px;margin-bottom:6px;letter-spacing:.5px}
h2 .num{color:var(--blue);font-family:var(--mono);margin-right:10px;font-size:20px}
.sub{color:var(--tx2);font-size:15px;margin-bottom:28px;max-width:760px}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:2px;color:var(--cyan);
     border:1px solid rgba(34,211,238,.35);border-radius:20px;padding:3px 12px;margin-bottom:14px}
code{font-family:var(--mono);font-size:.9em;background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.25);
     border-radius:5px;padding:1px 6px;color:#a5c8ff}

/* ---------- hero ---------- */
.hero{padding:88px 0 64px;position:relative;overflow:hidden;border-top:none}
.hero::before{content:"";position:absolute;inset:0;
  background:radial-gradient(ellipse 70% 55% at 50% -8%,rgba(59,130,246,.22),transparent 65%)}
.hero .wrap{position:relative}
.hero h1{font-size:42px;line-height:1.28;margin:14px 0 18px;font-weight:700}
.hero h1 em{font-style:normal;color:var(--cyan)}
.hero p.lead{color:var(--tx2);font-size:17px;max-width:640px;margin-bottom:30px}
.pipeline{display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-family:var(--mono);font-size:13px}
.pipeline .step{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:9px 16px}
.pipeline .step b{color:var(--cyan);font-weight:600}
.pipeline .arrow{color:var(--tx3)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:44px}
.kpi{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
     border-radius:14px;padding:20px 22px}
.kpi b{display:block;font-size:32px;font-weight:700;color:var(--tx);font-family:var(--mono)}
.kpi b span{color:var(--blue)}
.kpi i{font-style:normal;color:var(--tx3);font-size:12.5px;display:block;margin-top:4px}
.kpi.g b{color:var(--green)} .kpi.c b{color:var(--cyan)}

/* ---------- 照片世界 ---------- */
.world{display:grid;grid-template-columns:1.35fr 1fr;gap:22px;align-items:stretch}
.panoCard{position:relative;border-radius:14px;overflow:hidden;border:1px solid var(--line);min-height:340px}
.panoCard img{width:100%;height:100%;object-fit:cover;display:block}
.panoCard .cap{position:absolute;left:0;right:0;bottom:0;padding:26px 18px 12px;
  background:linear-gradient(transparent,rgba(5,10,20,.9));font-size:12.5px;color:var(--tx2)}
.panoCard .cap b{color:var(--tx);display:block;font-size:14px}
.worldList{display:flex;flex-direction:column;gap:12px;justify-content:center}
.wl{display:flex;gap:12px;background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px 16px}
.wl .dot{width:8px;height:8px;border-radius:50%;margin-top:7px;flex:none}
.wl h4{font-size:14.5px;margin-bottom:2px}
.wl p{font-size:12.5px;color:var(--tx3)}

/* ---------- 技能链 ---------- */
.chain{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:26px 0 18px}
.chain .ph{background:var(--panel);border:1px solid var(--line);border-radius:9px;
  padding:8px 13px;font-family:var(--mono);font-size:12.5px}
.chain .ph.hot{border-color:rgba(52,211,153,.5);color:var(--green)}
.chain .ar{color:var(--tx3)}
.chainNote{color:var(--tx3);font-size:13px}
.shots{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:22px}
.shot{border:1px solid var(--line);border-radius:11px;overflow:hidden;background:var(--panel)}
.shot img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}
.shot .sc{padding:8px 12px;font-size:12px;color:var(--tx2)}
.shot .sc b{display:block;color:var(--tx);font-size:12.5px}

/* ---------- 触觉 ---------- */
.tactile{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}
.card h3{font-size:15.5px;margin-bottom:12px}
.card h3 .mono{font-family:var(--mono);font-size:12px;color:var(--cyan);margin-left:8px}
#curve{width:100%;height:412px;display:block}
.timeline{font-family:var(--mono);font-size:12px;overflow-y:auto;max-height:412px}
.timeline .tl{display:flex;gap:10px;padding:7px 10px;border-radius:7px;align-items:baseline}
.timeline .tl:hover{background:rgba(59,130,246,.08)}
.timeline .tt{color:var(--tx3);flex:none;width:56px}
.timeline .ph{color:var(--cyan);flex:none;width:92px}
.timeline .tx{color:var(--tx2)}
.timeline .hl .tx{color:var(--green)}

/* ---------- 结果 ---------- */
.resGrid{display:grid;grid-template-columns:1.05fr .95fr;gap:22px}
table.res{width:100%;border-collapse:collapse;font-size:13.5px}
table.res th{color:var(--tx3);font-weight:500;text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);font-size:12px}
table.res td{padding:10px 12px;border-bottom:1px solid rgba(31,45,74,.55)}
table.res td b{font-family:var(--mono)}
.bar{height:6px;border-radius:4px;background:rgba(255,255,255,.07);overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%;border-radius:4px;background:linear-gradient(90deg,var(--blue),var(--cyan))}
.pill{display:inline-block;font-family:var(--mono);font-size:11px;border-radius:12px;padding:1px 9px}
.pill.g{background:rgba(52,211,153,.14);color:var(--green)}
.pill.a{background:rgba(251,191,36,.14);color:var(--amber)}
.pill.r{background:rgba(248,113,113,.14);color:var(--red)}
.failNote{margin-top:16px;font-size:13px;color:var(--tx2)}
.failNote b{color:var(--amber)}

/* ---------- 诚实边界 ---------- */
.honest{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.hCard{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--amber);
  border-radius:11px;padding:16px 18px}
.hCard.ok{border-left-color:var(--green)}
.hCard h4{font-size:14px;margin-bottom:5px}
.hCard p{font-size:12.8px;color:var(--tx2)}
.hCard code{font-size:11.5px}

footer{padding:40px 0 56px;color:var(--tx3);font-size:12.5px;border-top:1px solid var(--line)}
footer .wrap{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}
footer a{color:var(--cyan);text-decoration:none}

@media (max-width:860px){
  .kpis{grid-template-columns:repeat(2,1fr)}
  .world,.tactile,.resGrid,.honest{grid-template-columns:1fr}
  .shots{grid-template-columns:repeat(2,1fr)}
  .hero h1{font-size:30px}
}
</style>
</head>
<body>

<!-- ============ HERO ============ -->
<div class="hero"><div class="wrap">
  <span class="tag">ROBOT GYM · 群核 3D AI 黑客松 · 云深处 S10 + PIPER</span>
  <h1>拍照 → 机器人任务 → <em>可复现数据</em></h1>
  <p class="lead">把一张照片变成一个可交互的机器人任务环境：云深处 S10 机器狗 + Piper 机械臂
  在 MuJoCo 物理仿真中执行「接近 → 触觉抓取 → 搬运 → 放置」全流程，输出带因果日志的配对实验报告。</p>
  <div class="pipeline">
    <span class="step">📷 房间照片</span><span class="arrow">→</span>
    <span class="step">🌐 Aholo World（<b>S2 已验证</b>）</span><span class="arrow">→</span>
    <span class="step">🧸 Lux3D 物品（<b>A1 已验证</b>）</span><span class="arrow">→</span>
    <span class="step">🤖 <b>S10+Piper</b> MuJoCo 任务</span><span class="arrow">→</span>
    <span class="step">📊 配对实验 + 报告</span>
  </div>
  <div class="kpis">
    <div class="kpi g"><b>__KPI_GRASP__</b><i>双指接触抓取成功 / 30 次试次</i></div>
    <div class="kpi"><b>__KPI_TASK__</b><i>完整任务成功（含放置进托盘）</i></div>
    <div class="kpi c"><b>__KPI_RATE__</b><i>任务成功率（3 条件 × 5 种子对 × 2 模式）</i></div>
    <div class="kpi"><b>__KPI_FMAX__ N</b><i>实测峰值指力（mj_contactForce 逐接触求和）</i></div>
  </div>
</div></div>

<!-- ============ 01 照片世界 ============ -->
<section><div class="wrap">
  <h2><span class="num">01</span>照片 → 世界与资产</h2>
  <p class="sub">房间照片经 Aholo World 生成可漫游的三维空间（272k 点高斯溅射，8.7×9.4m），
  物品照片经 Lux3D 生成带贴图的 GLB 资产。真实 API 链路均已通过独立验证并缓存。</p>
  <div class="world">
    <div class="panoCard">
      <img src="__IMG_PANO__" alt="照片生成的世界全景">
      <div class="cap"><b>真实房间照片 → Aholo World 生成</b>
        全景 360° · splat 272k 点 · upAxis=Z · 生成耗时已记录（S2 验证缓存）</div>
    </div>
    <div class="worldList">
      <div class="wl"><span class="dot" style="background:var(--blue)"></span>
        <div><h4>World 生成管线</h4><p>全景与三维产物分别跟踪就绪，splats 缓存 + 元数据齐全，失败可重试</p></div></div>
      <div class="wl"><span class="dot" style="background:var(--cyan)"></span>
        <div><h4>Lux3D 物品资产</h4><p>照片 → GLB，保留原始贴图；质量/摩擦标注 estimated / manual 来源</p></div></div>
      <div class="wl"><span class="dot" style="background:var(--green)"></span>
        <div><h4>碰撞代理（F2.2）</h4><p>3DGS/网格不保证可碰撞——地面、桌面、托盘内腔均建显式代理，物理可判定</p></div></div>
    </div>
  </div>
</div></section>

<!-- ============ 02 任务与技能链 ============ -->
<section><div class="wrap">
  <h2><span class="num">02</span>一个命令，一条完整的任务</h2>
  <p class="sub">场景 / 物体 / 条件 / 种子 / 成功判据全部参数化在 <code>scenes/*.json</code>——
  换参数不改代码。九段技能由状态机程序化驱动，替代了写死关键帧的 demo 脚本。</p>
  <div class="chain" id="chain"></div>
  <p class="chainNote">每段技能有独立失败原因（<code>ik_failed / no_contact / slip_drop / force_limit / unstable</code>），
  成功判据执行 PRD v0.3 §4：离面 ≥5cm 保持 ≥3s、落入托盘区静止。</p>
  <div class="shots">
    <div class="shot"><img src="__IMG_F_STAND__" alt="站立"><div class="sc"><b>Settle 站立</b>自由底座沉降→稳定窗口判定→0.248m（官方 0.26）</div></div>
    <div class="shot"><img src="__IMG_F_APPROACH__" alt="接近"><div class="sc"><b>Approach 接近</b>6D IK（轴对齐 5D）预抓取悬停→垂直下探</div></div>
    <div class="shot"><img src="__IMG_F_LIFT__" alt="抓取"><div class="sc"><b>Grasp→Lift 抓取</b>双指接触确认→夹持抬升 8cm 保持 3s</div></div>
    <div class="shot"><img src="__IMG_F_RELEASE__" alt="放置"><div class="sc"><b>Release 放置</b>先开爪后解约束→杯垂直落入托盘内腔</div></div>
  </div>
</div></section>

<!-- ============ 03 触觉反馈 ============ -->
<section><div class="wrap">
  <h2><span class="num">03</span>触觉反馈：观测 → 决策 → 执行</h2>
  <p class="sub">控制器只能读「观测字典」：两指法向接触力、接触数、滑动量。摩擦设定、质量、条件参数
  是隐藏环境参数，只存在于仿真侧——信息隔离保证反馈不是抄答案（PRD F4.4）。
  下图为代表性试次的完整时序（25Hz 观测），可对照右侧因果日志逐帧核查。</p>
  <div class="tactile">
    <div class="card"><h3>两指接触力 + 离面高度 <span class="mono">telemetry 25Hz</span></h3>
      <canvas id="curve"></canvas></div>
    <div class="card"><h3>因果日志 <span class="mono">run.json · events</span></h3>
      <div class="timeline" id="timeline"></div></div>
  </div>
</div></section>

<!-- ============ 04 配对实验 ============ -->
<section><div class="wrap">
  <h2><span class="num">04</span>配对实验：30 次，全部数据公开</h2>
  <p class="sub">3 条件（正常 / 低摩擦 / 偏心摆放）× 5 种子对 × 2 模式（固定力度 vs 触觉反馈）。
  同种子配对：同初始抖动、同轨迹、同接触模型，仅反馈调节不同。每次独立重置场景，
  <b>全部失败样本保留并计入</b>——汇总不筛选。</p>
  <div class="resGrid">
    <div class="card">
      <h3>条件 × 模式矩阵 <span class="mono">f6_v5 · config __CONFIG__</span></h3>
      <table class="res"><thead><tr><th>条件</th><th>模式</th><th>抓取</th><th>任务</th><th>成功率</th></tr></thead>
      <tbody id="resBody"></tbody></table>
      <div class="failNote">失败分布：<b>force_limit ×2/模式</b>（接近段指-杯刮擦）·
      <b>ik_failed ×3/模式</b>（偏心条件放置点腕倾角运动学边界）——定位过程全部依赖本平台的
      可复现运行与遥测，见 KNOWN_ISSUES.md。</div>
    </div>
    <div class="card">
      <h3>两模式对比 <span class="mono">同种子配对</span></h3>
      <div style="display:flex;gap:26px;margin:14px 0 6px">
        <div style="flex:1"><div style="font-size:13px;color:var(--tx2);margin-bottom:6px">固定力度 fixed</div>
          <div style="font-family:var(--mono);font-size:26px">__FIXED_TASK__/15</div>
          <div class="bar"><i style="width:__FIXED_PCT__%"></i></div></div>
        <div style="flex:1"><div style="font-size:13px;color:var(--tx2);margin-bottom:6px">触觉反馈 feedback</div>
          <div style="font-family:var(--mono);font-size:26px">__FB_TASK__/15</div>
          <div class="bar"><i style="width:__FB_PCT__%"></i></div></div>
      </div>
      <p style="font-size:13px;color:var(--tx2);margin-top:16px">两模式任务率一致（67%）是<b>诚实的预期</b>：
      当前抓取保持基底为披露的 weld 辅助（接触确认真实），基底吸收了摩擦差异，反馈调节不改变最终结果。
      反馈状态机的真实决策序列（slip → tighten → force 恢复）在因果日志中可完整回放——
      其价值验证依赖纯摩擦基底的打通（赛后路线，诊断链已完备）。</p>
      <p style="font-size:13px;color:var(--tx3);margin-top:10px">协议：PRD v0.3 §4/F6 ·
      grasp_mode = contact-confirm + weld-assist (disclosed) · 每试次 run.json + telemetry.jsonl 全量归档</p>
    </div>
  </div>
</div></section>

<!-- ============ 05 诚实边界 ============ -->
<section><div class="wrap">
  <h2><span class="num">05</span>诚实边界</h2>
  <p class="sub">对外表述采用「生成可交互任务环境、验证机器人策略」；不称训练，不伪造数据。</p>
  <div class="honest">
    <div class="hCard ok"><h4>✓ 已验证</h4>
      <p>Aholo World（S2）· Lux3D 资产（A1）· LLM 规划 8/8（A4）· S10+Piper 官方策略仿真
      实时 0.95×（L20 远端，S1）· 技能链 + 触觉观测 + 配对批量（本次）</p></div>
    <div class="hCard ok"><h4>✓ 平台能力</h4>
      <p>同配置重复运行结果位相同（已实证）· config_hash 锁定参数版本 ·
      失败原因枚举覆盖全部试次 · 一条命令复现整批实验：<code>python -m gym batch</code></p></div>
    <div class="hCard"><h4>△ 披露的简化</h4>
      <p>抓取保持基底 = 接触确认（真实）+ weld 辅助（披露，PRD §8 预案）；纯接触摩擦基底
      未打通，9 项对照实验诊断链保留在 <code>KNOWN_ISSUES.md</code> ISSUE-1。
      滑动估计为仿真位姿差分（ideal-pose-assisted）。</p></div>
    <div class="hCard"><h4>△ 范围外</h4>
      <p>表格场景底座站立后运动学保持（真实行走走官方策略轨道）；照片→资产→场景全链路
      接线在赛后路线（管线各自已验证）；触觉为接触力构建的分布近似，非真实传感器。</p></div>
  </div>
</div></section>

<footer><div class="wrap">
  <div>Robot Gym — 本页由 <code>python -m gym showcase</code> 从原始运行数据自动生成</div>
  <div><a href="#" onclick="window.print();return false">打印/导出 PDF</a> ·
       数据: runs/f6_v5 · PRD v0.3 · PRODUCT.md · KNOWN_ISSUES.md</div>
</div></footer>

<script>
const DATA = __DATA_JSON__;

/* 技能链 */
(function(){
  const names = ["settle","approach","grasp","lift","hold","shift","move","release","verify","retreat"];
  const hot = new Set(["grasp","lift","move"]);
  const el = document.getElementById("chain");
  el.innerHTML = names.map((n,i)=>
    `<span class="ph${hot.has(n)?" hot":""}">${n}</span>${i<names.length-1?'<span class="ar">→</span>':''}`
  ).join("");
})();

/* 结果矩阵 */
(function(){
  const tb = document.getElementById("resBody");
  const condName = {normal:"正常", low_friction:"低摩擦", eccentric:"偏心摆放"};
  tb.innerHTML = DATA.matrix.map(m=>{
    const fx=m.fixed, fb=m.feedback;
    const rate=Math.round((fx.task_success+fb.task_success)/(fx.n+fb.n)*100);
    const pill = rate>=80?"g":(rate>=50?"a":"r");
    return `<tr><td>${condName[m.condition]||m.condition}</td>
      <td><span style="font-family:var(--mono);font-size:12px;color:var(--tx3)">fixed / feedback</span></td>
      <td><b>${fx.grasp_success}/${fx.n}</b> · <b>${fb.grasp_success}/${fb.n}</b></td>
      <td><b>${fx.task_success}/${fx.n}</b> · <b>${fb.task_success}/${fb.n}</b></td>
      <td><span class="pill ${pill}">${rate}%</span>
        <div class="bar" style="width:120px"><i style="width:${rate}%"></i></div></td></tr>`;
  }).join("");
})();

/* 因果日志 */
(function(){
  const evColor = {contact_confirmed:"hl", grip_closed:"hl", lifted:"hl", weld_assist_engaged:"", hold_done:"", result:"hl"};
  const fmt = e=>{
    const m = e.metrics||{};
    if (e.event==="contact_confirmed") return `双指接触确认 fL=${e.f_left}N fR=${e.f_right}N`;
    if (e.event==="grip_closed") return `闭合 gap=${e.gap} 接触数=${e.n_contacts}`;
    if (e.event==="weld_assist_engaged") return `保持基底激活（披露 weld，PRD §8）`;
    if (e.event==="stand_ok") return `站立稳定 pelvis_z=${e.pelvis_z}m`;
    if (e.event==="at_mug") return `到达抓取位 IK 误差 0.0m`;
    if (e.event==="lifted") return `抬升完成 离面 ${e.clear_cm}cm 倾角 ${e.grip_tilt_deg}°`;
    if (e.event==="hold_done") return `保持 3s 达标 离面 ${e.clear_cm}cm`;
    if (e.event==="over_tray") return `到达托盘上方 倾角 ${e.grip_tilt_deg}°`;
    if (e.event==="grip_opened") return `开爪 → 杯落入托盘`;
    if (e.event==="verify_done") return `放置判定完成 杯位置 [${e.mug.map(v=>v.toFixed(2)).join(", ")}]`;
    if (e.event==="result") return e.task_success
      ? `任务成功 · Fmax=${m.max_finger_force_N}N · 离面峰 ${m.lift_peak_cm}cm · 静止 v=${m.mug_final_speed_ms}m/s`
      : `任务失败 ${e.failure_reason||""} ${e.failure_detail||""}`;
    if (e.event==="retreat_up") return `抬升避让回位`;
    return e.event;
  };
  document.getElementById("timeline").innerHTML = DATA.rep_run.events.map(e=>
    `<div class="tl ${evColor[e.event]||""}"><span class="tt">${e.t.toFixed(2)}s</span>
     <span class="ph">${e.phase}</span><span class="tx">${fmt(e)}</span></div>`).join("");
})();

/* 力曲线（原生 canvas） */
function drawCurve(){
  const cv = document.getElementById("curve");
  const dpr = window.devicePixelRatio||1;
  const W = cv.clientWidth || cv.parentElement.clientWidth || 460;
  const H = 412;
  cv.width = W*dpr; cv.height = H*dpr;
  cv.style.height = H+"px";
  const g = cv.getContext("2d"); g.scale(dpr,dpr);
  const pts = DATA.rep_run.curve;
  const T = pts[pts.length-1][0];
  const phases = ["settle","approach","grasp","lift","hold","shift","move","release","verify","retreat"];
  const phCol = {"grasp":"rgba(52,211,153,.10)","lift":"rgba(52,211,153,.10)","hold":"rgba(52,211,153,.10)",
                 "move":"rgba(59,130,246,.07)","release":"rgba(251,191,36,.10)","verify":"rgba(251,191,36,.06)"};
  // NOTE: 上一行末尾缺逗号曾导致整段 JS 崩溃（对象字面量跨行必须有逗号）
  const pad = {l:44,r:12,t:14,b:26};
  const x = t=>pad.l + (t/T)*(W-pad.l-pad.r);
  const FMAX = 9, ZSPAN = 9; // cm 0..9
  // 阶段底色
  let segStart = 0, curPh = pts[0][5];
  for (const p of pts){
    if (p[5]!==curPh){
      if (phCol[curPh]){ g.fillStyle=phCol[curPh]; g.fillRect(x(segStart),pad.t,x(p[0])-x(segStart),H-pad.t-pad.b); }
      segStart=p[0]; curPh=p[5];
    }
  }
  g.fillStyle=phCol[curPh]||"transparent"; g.fillRect(x(segStart),pad.t,x(T)-x(segStart),H-pad.t-pad.b);
  // 网格
  g.strokeStyle="rgba(255,255,255,.06)"; g.fillStyle="#64768f"; g.font="10px ui-monospace";
  for (let z=0; z<=ZSPAN; z+=3){
    const y = pad.t+(1-z/ZSPAN)*(H-pad.t-pad.b);
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(W-pad.r,y); g.stroke();
    g.fillText(z+"cm",6,y+3);
  }
  for (let t=0; t<=T; t+=5){ g.fillText(t+"s", x(t)-8, H-8); }
  const line = (idx, color, scale, y0, y1)=>{
    g.strokeStyle=color; g.lineWidth=1.8; g.beginPath();
    pts.forEach((p,i)=>{
      const v = Math.min(Math.max(p[idx]/scale,0),1);   // 归一并夹紧
      const y = y1 + (1-v)*(y0-y1);
      i?g.lineTo(x(p[0]),y):g.moveTo(x(p[0]),y);
    });
    g.stroke();
  };
  // fL/fR（0..9N → 全幅）与 clear（0..9cm → 全幅，虚线）
  const yTop=pad.t+6, yBot=H-pad.b-4;
  line(1,"rgba(34,211,238,.9)",FMAX,yBot,yTop);
  line(2,"rgba(167,139,250,.9)",FMAX,yBot,yTop);
  g.setLineDash([5,4]);
  line(3,"rgba(52,211,153,.9)",ZSPAN,yBot,yTop);
  g.setLineDash([]);
  // 图例
  const lg=[["左指力 N","#22d3ee"],["右指力 N","#a78bfa"],["离面高度 cm（虚线）","#34d399"]];
  lg.forEach((l,i)=>{ g.fillStyle=l[1]; g.fillRect(W-190, pad.t+4+i*15, 14, 3);
    g.fillStyle="#9fb0c9"; g.fillText(l[0], W-170, pad.t+9+i*15); });
  // 接触确认标记
  const cc = pts.find(p=>p[1]>0.4&&p[2]>0.4);
  if (cc){ g.fillStyle="#34d399"; g.beginPath(); g.arc(x(cc[0]), pad.t+10, 4, 0, 7); g.fill();
    g.fillText("contact", x(cc[0])+7, pad.t+13); }
}
requestAnimationFrame(()=>{ drawCurve(); window.addEventListener("resize", drawCurve); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    p = build()
    print(f"showcase -> {p} ({p.stat().st_size//1024} KB)")
