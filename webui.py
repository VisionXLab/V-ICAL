"""
Video-CL Web UI 服务端
FastAPI + WebSocket 实时游戏交互
支持 Human Interaction 和 Machine (AI) Interaction 两种模式
"""
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# 导入核心模块
from src.core import GameSession
from src.ai import AIPlayer
from src.api import register_routes, register_websocket

# 导入自定义 Breakout（可选）
CUSTOM_BREAKOUT_AVAILABLE = False
try:
    from new_gym.breakout_wrapper import create_breakout_env, BreakoutGameEnv
    CUSTOM_BREAKOUT_AVAILABLE = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# 游戏列表
# ---------------------------------------------------------------------------
GAMES = {
    # === Toy Text 系列 ===
    "1": ("Taxi-v3", "出租车（网格接送，超机械）", "toy_text"),
    "2": ("CliffWalking-v1", "悬崖漫步（上下左右一步）", "toy_text"),

    # === MiniGrid 系列 ===

    # === Classic Control 系列 ===
    "8": ("CartPole-v1", "倒立摆（连续控制但动作明确）", "classic_control"),
    "9": ("Acrobot-v1", "Acrobot（连续控制）", "classic_control"),

    # === ALE/Atari 系列 ===
    "10": ("ALE/Enduro-v5", "耐力赛（Atari，较慢）", "ale"),
    "11": ("ALE/Freeway-v5", "高速公路（Atari，上下移动）", "ale"),
    "12": ("ALE/Seaquest-v5", "海底任务（Atari，节奏中等）", "ale"),
    "14": ("ALE/Pong-v5", "乒乓球（Atari，实时）", "ale"),
    "15": ("CustomBreakout", "打砖块（可调参数，无惯性，正确掉球处理）", "custom_breakout"),

    # === Classic Control 系列（续）===
    "18": ("MountainCar-v0", "山地车（左右加速到山顶）", "classic_control"),

    # === MiniGrid 系列（续）===
    "21": ("CustomLavaCrossing-v0", "MiniGrid 岩浆穿越（可调参数）", "minigrid"),
    "23": ("CustomUnlockPickup-v0", "MiniGrid 开锁拾取（可调参数）", "minigrid"),
    "24": ("CustomMultiRoom-v0", "MiniGrid 多房间（可调参数）", "minigrid"),

    # === ALE/Atari 系列（续）===
    "26": ("ALE/Frostbite-v5", "冰雪冒险（跳冰块建冰屋）", "ale"),
    "28": ("ALE/Assault-v5", "突击（垂直射击）", "ale"),
    "29": ("ALE/Asterix-v5", "阿斯特里克斯（收集躲避）", "ale"),
    "30": ("ALE/DemonAttack-v5", "恶魔攻击（底部射击）", "ale"),
    "36": ("ALE/MsPacman-v5", "吃豆人（迷宫吃点）", "ale"),
    "37": ("ALE/Alien-v5", "异形（迷宫射击）", "ale"),
    "40": ("ALE/RoadRunner-v5", "走鹃（横版跑酷）", "ale"),

    #Box2D

    "45":("highway-v0","High Way","highway"),

    "101": ("ALE/BattleZone-v5", "战区（第一人称坦克射击）", "ale"),
    "103": ("ALE/Riverraid-v5", "河袭击（喷气机横向射击）", "ale"),
    "104": ("ALE/Turmoil-v5", "混乱（消灭外星人）", "ale"),
    "105": ("ALE/Trondead-v5", "特隆死亡（飞碟对抗）", "ale"),

    #ViZDoom

    #Thirdparty
    '50':("aFlappyBird-v1","Flappy Bird","Thirdparty"),
    '51':("tetris_gymnasium/Tetris","testris","Thirdparty"),

    #Procgen
    '52':("procgen_:procgen-bigfish-v0","bigfish","procgen"),
    '53':("procgen_:procgen-bossfight-v0","bossfight","procgen"),
    '54':("procgen_:procgen-caveflyer-v0","caveflyer","procgen"),
    '58':("procgen_:procgen-dodgeball-v0","dodgeball","procgen"),
    '60':("procgen_:procgen-heist-v0","heist","procgen"),
    '62':("procgen_:procgen-leaper-v0","leaper","procgen"),
    '65':("procgen_:procgen-ninja-v0","ninja","procgen"),

    # === New ALE/Atari (fzq) ===
    "68": ("ALE/Tennis-v5", "网球（接发球空间判断）", "ale"),

    # === Batch 3 — S/A/B tier ALE (fzq) ===
    "69": ("ALE/ChopperCommand-v5", "直升机指挥官（护车队多威胁）", "ale"),
    "70": ("ALE/Berzerk-v5", "狂暴迷宫（机器人+Evil Otto）", "ale"),
}


# ---------------------------------------------------------------------------
# FastAPI App 初始化
# ---------------------------------------------------------------------------

app = FastAPI(title="Video-CL Web UI")

# 静态文件
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# operation 文件夹路径
HUMAN_OP_DIR = Path(__file__).parent / "human_operation"
AI_OP_DIR = Path(__file__).parent / "ai_operation"
CROSSVAL_OP_DIR = Path(__file__).parent / "crossval_operation"
HUMAN_EVAL_DIR = Path(__file__).parent / "human_eval_results"
HUMAN_OP_DIR.mkdir(exist_ok=True)
AI_OP_DIR.mkdir(exist_ok=True)
CROSSVAL_OP_DIR.mkdir(exist_ok=True)

# 全局会话存储
sessions: Dict[str, GameSession] = {}
ai_players: Dict[str, AIPlayer] = {}


# ---------------------------------------------------------------------------
# 注册路由和 WebSocket
# ---------------------------------------------------------------------------

# 注册 REST API 路由
register_routes(
    app=app,
    sessions=sessions,
    ai_players=ai_players,
    games_dict=GAMES,
    static_dir=static_dir,
    human_op_dir=HUMAN_OP_DIR,
    ai_op_dir=AI_OP_DIR,
    crossval_op_dir=CROSSVAL_OP_DIR,
    human_eval_root=HUMAN_EVAL_DIR,
)

# 注册 WebSocket 端点
register_websocket(
    app=app,
    sessions=sessions,
    ai_players=ai_players,
    human_eval_root=HUMAN_EVAL_DIR,
)


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run("webui:app", host=args.host, port=args.port, reload=True)
