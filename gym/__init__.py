"""Robot Gym — 拍照 → 机器人任务 → 可复现数据。

程序化任务执行平台：TaskSpec 场景参数化 + MuJoCo 物理场景 + 技能状态机
（含触觉反馈抓取）+ 配对实验批量运行 + 报告生成。

用法：
  python -m gym run   --scene tabletop_pick_place --condition normal --mode feedback --seed 0
  python -m gym batch --scene tabletop_pick_place --pairs 5
  python -m gym report --batch <batch_id>
"""
__version__ = "0.4.0"

REPO = None  # 由 gym.paths() 初始化
