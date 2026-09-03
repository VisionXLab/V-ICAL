"""
简洁的游戏环境封装
支持快速反馈和动作空间信息
"""
import gym as gym_origin
import gymnasium as gym
import highway_env
from vizdoom import gymnasium_wrapper
import flappy_bird_env_custom
import procgen_
from pathlib import Path
from tetris_gymnasium.envs.tetris import Tetris
from gymnasium.wrappers import ResizeObservation
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from xsb_wrapper import *
from collections import deque


_LOCAL_PROCGEN_DIR = Path(__file__).resolve().parent / "procgen_"
if Path(procgen_.__file__).resolve().parent != _LOCAL_PROCGEN_DIR:
    raise RuntimeError(
        f"Video-CL requires the repository-local procgen_ source build at {_LOCAL_PROCGEN_DIR}; "
        f"refusing to use {procgen_.__file__}."
    )


def _fl_bfs_dist(desc, nrow, ncol, start_r, start_c, goal_r, goal_c):
    """BFS 计算从 (start_r,start_c) 到 (goal_r,goal_c) 的最短步数，洞(H)不可通行。"""
    if start_r == goal_r and start_c == goal_c:
        return 0
    visited = [[False] * ncol for _ in range(nrow)]
    visited[start_r][start_c] = True
    q = deque([(start_r, start_c, 0)])
    while q:
        r, c, d = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrow and 0 <= nc < ncol and not visited[nr][nc]:
                if desc[nr][nc] == b'H':
                    continue
                if nr == goal_r and nc == goal_c:
                    return d + 1
                visited[nr][nc] = True
                q.append((nr, nc, d + 1))
    return float('inf')  # 理论上不可达


# 导入 ALE 环境以注册命名空间（新版本 gymnasium 需要）
# 这些导入是为了确保 ALE 命名空间被注册
try:
    import ale_py  # 必须先导入 ale_py 来注册 ALE 命名空间
except ImportError:
    pass

try:
    import minigrid  # 必须先导入 minigrid 来注册 MiniGrid 命名空间
except ImportError:
    pass

try:
    import new_gym  # 注册自定义 MiniGrid 环境（CustomLavaCrossing 等）
except ImportError:
    pass

# 自定义 MiniGrid 环境 ID 集合（用于 is_minigrid 检测）
CUSTOM_MINIGRID_IDS = {'CustomLavaCrossing-v0', 'CustomMultiRoom-v0', 'CustomUnlockPickup-v0'}


class DiscretizeAction(gym.ActionWrapper):
    """Wraps a continuous action space into discrete bins."""
    def __init__(self, env, bins):
        super().__init__(env)
        self._bins = np.array(bins, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self._bins))

    def action(self, act):
        return np.array([self._bins[act]], dtype=np.float32)

# 尝试导入 atari 环境模块（如果存在）
# 注意：某些版本的 gymnasium 可能不需要显式导入
try:
    # 尝试不同的导入方式
    try:
        import gymnasium.envs.atari  # type: ignore
    except (ImportError, AttributeError, ModuleNotFoundError):
        try:
            from gymnasium.envs import atari  # type: ignore
        except (ImportError, AttributeError, ModuleNotFoundError):
            pass
except Exception:
    # 忽略所有导入错误，环境创建时会处理
    pass


# 预定义的动作含义映射（用于没有 get_action_meanings 方法的环境）
# 所有名称使用简洁英文，ALE 游戏根据人类视角语义命名（而非内部按钮名）
PREDEFINED_ACTION_MEANINGS = {
    # Toy Text
    "Taxi-v3": ["Down", "Up", "Right", "Left", "Pickup", "Dropoff"],
    "CliffWalking-v1": ["Up", "Right", "Down", "Left"],
    "FrozenLake-v0": ["Left", "Down", "Right", "Up"],
    # MiniGrid
    # Classic Control
    "CartPole-v1": ["Left", "Right"],
    "CartPole-v0": ["Left", "Right"],
    "Acrobot-v1": ["Torque -1", "No Torque", "Torque +1"],
    "MountainCar-v0": ["Accelerate Left", "No Accelerate", "Accelerate Right"],
    # ALE — override native names with human-perspective semantics
    "ALE/Enduro-v5": ["NOOP", "Accelerate", "Right", "Left", "Brake", "Brake+Right", "Brake+Left", "Accelerate+Right", "Accelerate+Left"],
    "ALE/Freeway-v5": ["NOOP", "Up", "Down"],
    "ALE/Seaquest-v5": ["NOOP", "Shoot", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Shoot", "Right+Shoot", "Left+Shoot", "Down+Shoot", "Up+Right+Shoot", "Up+Left+Shoot", "Down+Right+Shoot", "Down+Left+Shoot"],
    "ALE/Pong-v5": ["NOOP", "Fire", "Up", "Down", "Up+Fire", "Down+Fire"],
    "CustomBreakout": ["NOOP", "Fire", "Right", "Left"],
    # === 新增游戏 ===
    # Classic Control (续)
    # MountainCar-v0 已在上方定义(第69行)
    # MiniGrid (续) — 全部共享相同的 7 动作
    "MiniGrid-LavaCrossingS11N5-v0": ["Left", "Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    "MiniGrid-UnlockPickup-v0": ["Left", "Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    "MiniGrid-MultiRoom-N6-v0": ["Left", "Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    # Custom MiniGrid
    "CustomLavaCrossing-v0": ["Left", "Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    "CustomMultiRoom-v0": ["Left", "Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    "CustomUnlockPickup-v0": ["Left", "Right", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    # ALE/Atari (续) — 语义化命名
    "ALE/Frostbite-v5": ["NOOP", "Enter Igloo", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Enter Igloo", "Right+Enter Igloo", "Left+Enter Igloo", "Down+Enter Igloo", "Up+Right+Enter Igloo", "Up+Left+Enter Igloo", "Down+Right+Enter Igloo", "Down+Left+Enter Igloo"],
    "ALE/Assault-v5": ["NOOP", "Shoot", "Up", "Right", "Left", "Right+Shoot", "Left+Shoot"],
    "ALE/Asterix-v5": ["NOOP", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left"],
    "ALE/DemonAttack-v5": ["NOOP", "Shoot", "Right", "Left", "Right+Shoot", "Left+Shoot"],
    "ALE/MsPacman-v5": ["NOOP", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left"],
    "ALE/Alien-v5": ["NOOP", "Shoot", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Shoot", "Right+Shoot", "Left+Shoot", "Down+Shoot", "Up+Right+Shoot", "Up+Left+Shoot", "Down+Right+Shoot", "Down+Left+Shoot"],
    "ALE/RoadRunner-v5": ["NOOP", "Speed Up", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Speed Up", "Right+Speed Up", "Left+Speed Up", "Down+Speed Up", "Up+Right+Speed Up", "Up+Left+Speed Up", "Down+Right+Speed Up", "Down+Left+Speed Up"],

    #Box2D

    #highway
    "highway-v0":["Left","Idle","Right","Faster","Slower"],

    # New ALE games
    "ALE/Tennis-v5": ["NOOP", "Swing", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Swing", "Right+Swing", "Left+Swing", "Down+Swing", "Up+Right+Swing", "Up+Left+Swing", "Down+Right+Swing", "Down+Left+Swing"],

    # Batch 3 — S/A/B tier ALE games
    "ALE/ChopperCommand-v5":["NOOP", "Shoot", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Shoot", "Right+Shoot", "Left+Shoot", "Down+Shoot", "Up+Right+Shoot", "Up+Left+Shoot", "Down+Right+Shoot", "Down+Left+Shoot"],
    "ALE/Berzerk-v5":       ["NOOP", "Shoot", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Shoot", "Right+Shoot", "Left+Shoot", "Down+Shoot", "Up+Right+Shoot", "Up+Left+Shoot", "Down+Right+Shoot", "Down+Left+Shoot"],

    "ALE/BattleZone-v5": ["NOOP", "Fire", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Fire", "Right+Fire", "Left+Fire", "Down+Fire", "Up+Right+Fire", "Up+Left+Fire", "Down+Right+Fire", "Down+Left+Fire"],
    "ALE/Riverraid-v5": ["NOOP", "Fire", "Accelerate", "Right", "Left", "Decelerate", "Accelerate+Right", "Accelerate+Left", "Decelerate+Right", "Decelerate+Left", "Accelerate+Fire", "Right+Fire", "Left+Fire", "Decelerate+Fire", "Accelerate+Right+Fire", "Accelerate+Left+Fire", "Decelerate+Right+Fire", "Decelerate+Left+Fire"],
    "ALE/Turmoil-v5": ["NOOP", "Fire", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Right+Fire", "Left+Fire"],
    "ALE/Trondead-v5": ["NOOP", "Fire", "Up", "Right", "Left", "Down", "Up+Right", "Up+Left", "Down+Right", "Down+Left", "Up+Fire", "Right+Fire", "Left+Fire", "Down+Fire", "Up+Right+Fire", "Up+Left+Fire", "Down+Right+Fire", "Down+Left+Fire"],

    #ViZDoom

    #Flappy-bird
    "aFlappyBird-v1":["NOOP","Flap"],
    "tetris_gymnasium/Tetris":["Left","Right","Down","Clockwise","Counter-clockwise","Drop","Swap","NOOP"],

    #Procgen
    "procgen-bigfish-v0":["Left+Down","Left","Left+Up","Down","NOOP","Up","Right+Down","Right","Right+Up","D","A","W","S","Q","E"],
    "procgen-bossfight-v0":["Left+Down","Left","Left+Up","Down","NOOP","Up","Right+Down","Right","Right+Up","Shoot","A","W","S","Q","E"],
    "procgen-caveflyer-v0":["Left+Down","Left","Left+Up","Down","NOOP","Up","Right+Down","Right","Right+Up","Shoot","A","W","S","Q","E"],
    "procgen-dodgeball-v0":["Left+Down","Left","Left+Up","Down","NOOP","Up","Right+Down","Right","Right+Up","Shoot","A","W","S","Q","E"],
    "procgen-heist-v0":["Left+Down","Left","Left+Up","Down","NOOP","Up","Right+Down","Right","Right+Up","D","A","W","S","Q","E"],
    "procgen-leaper-v0":["Left+Down","Left","Left+Up","Down","NOOP","Up","Right+Down","Right","Right+Up","D","A","W","S","Q","E"],
    "procgen-ninja-v0":["Left+Down","Left","Left+Up","Down","NOOP","Up","Right+Down","Right","Right+Up","Shoot Front","Shoot Up-front","Shoot Up","Dump","Shoot Fornt2","Shoot Front3"],
}


# Simple mode: 仅展示核心动作（减少 AI 选择负担）
ACTION_SIMPLE_IDS = {
    "ALE/Seaquest-v5": [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],      # 18→10: drop diagonals & diagonal+shoot
# 18→10: same pattern
    "ALE/Pong-v5":     [0, 2, 3],                                  # 6→3: drop Fire & combos (Pong auto-serves after delay)
    "ALE/Enduro-v5":   [0, 1, 2, 3, 4],                           # 9→5: drop combos
# 6→5: drop useless Fire
    # 新增游戏 — 18 动作游戏去掉对角线和对角线组合
    "ALE/Frostbite-v5":    [0, 2, 3, 4, 5, 6, 7, 8, 9],  # 9 actions: NOOP + 4 directions + 4 diagonals, no Enter Igloo
# 18→10
# 18→10
# 18→10
    "ALE/Alien-v5":        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],  # 18→10
# 18→10
    "ALE/RoadRunner-v5":   [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],  # 18→10
    "ALE/Riverraid-v5":   [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],  # 18→10: keep core directions + shoot combos
    "ALE/Trondead-v5":    [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],  # 18→10: keep core directions + shoot combos
    # 9 动作游戏去掉对角线
# 9→7: drop diagonals
    "ALE/Asterix-v5":      [0, 1, 2, 3, 4],                      # 9→5: drop diagonals
    "ALE/MsPacman-v5":     [0, 1, 2, 3, 4],                      # 9→5: drop diagonals
    # 14 动作游戏
# 14→8: keep core directions + kick combos


    "highway-v0":[0,1,2,3,4],
    # New ALE 18-action games → 10 (standard pattern: NOOP + Fire + 4 dirs + 4 dir+fire)
    "ALE/Tennis-v5":            [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],
# 6: item use doesn't need directional combos

    # Batch 3 — S/A/B tier ALE games: 18→10 standard pruning for most
    "ALE/ChopperCommand-v5": [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],
    "ALE/Berzerk-v5":        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],
    # VideoPinball: 9→6 drop launch-combos (rare use)

    #ViZDoom

    #Thirdparty
    "aFlappyBird-v1":[0,1],
    "tetris_gymnasium/Tetris":[0,1,3,5,7],

    #procgen
    "procgen_:procgen-bigfish-v0":[1,3,4,5,7],
    "procgen-bossfight-v0":[1,3,4,5,7,9],
    "procgen-caveflyer-v0":[1,3,4,5,7],
    "procgen-dodgeball-v0":[1,3,4,5,7,9],
    "procgen-heist-v0":[1,3,4,5,7],
    "procgen-leaper-v0":[1,3,4,5,7],
    "procgen-ninja-v0":[1,3,4,5,7],
}


@dataclass
class GameState:
    """游戏状态信息"""
    image: np.ndarray  # 当前帧图片 (RGB array)
    reward: float  # 奖励
    terminated: bool  # 是否结束
    truncated: bool  # 是否截断
    info: Dict[str, Any]  # 额外信息
    observation: np.ndarray  # 原始观察值
    ever_scored: bool = False  # 曾经得分/接球标志（持久化）


# Pong RAM 地址映射
PONG_RAM_MAP = {
    'player_y': 51,
    'enemy_y': 50,
    'ball_x': 49,
    'ball_y': 54,
    'player_score': 14,
    'enemy_score': 13,
}

# 游戏 RAM 映射表
GAME_RAM_MAPS = {
    'Pong': PONG_RAM_MAP,
}


# 游戏 lives RAM 地址（用于检测丢球/死亡和设置初始生命）
GAME_LIVES_RAM = {
    'Pong': None,  # Pong 用分数而非 lives
    'Seaquest': 59,
    'Frostbite': 76,   # 6502 addr 0xCC, getRAM index = 0xCC-0x80 = 0x4C = 76, 低 4 位有效
    'Asterix': 83,     # 6502 addr 0xD3, getRAM index = 0xD3-0x80 = 0x53 = 83, 低 4 位有效
# 6502 addr 0xED, getRAM index = 0xED-0x80 = 0x6D = 109, (>>4)&0x7
    'Enduro': None,
    'Freeway': None,
# RAM[5] 存储 lives-1（3条命时值为2）
    'Riverraid': 61,
}

# 某些游戏 RAM 中存储的是 (lives + offset) 而非 lives 本身
# 设置初始生命时需要减去此偏移才能写入正确值
GAME_LIVES_RAM_WRITE_OFFSET = {
# RAM 存储 lives-1，写入时需减1
}

# 需要对 RAM 值做变换的游戏（int=简单mask, tuple=(shift, mask)）
_LIVES_RAM_MASK = {
    'Frostbite': 0x0F,      # val & 0x0F（低 4 位）
    'Asterix': 0x0F,        # val & 0x0F（低 4 位）
# (val >> 4) & 0x07（高 nibble 的低 3 位）
}

# 需要对 RAM 值做变换的游戏（int=简单mask, tuple=(shift, mask)）
_LIVES_RAM_MASK = {
    'Frostbite': 0x0F,      # val & 0x0F（低 4 位）
    'Asterix': 0x0F,        # val & 0x0F（低 4 位）
    'Centipede': (4, 0x07), # (val >> 4) & 0x07（高 nibble 的低 3 位）
}

# 需要按 FIRE 键复活的游戏（死后游戏暂停，必须按 FIRE 才能继续下一条命）
AUTO_RESPAWN_GAMES = {'Seaquest'}

# 游戏名前缀 → xsb_wrapper class 的注册表。create_env() 匹配第一个 startswith 命中的前缀，
# 用 pre_score 调用 wrapper；如果该 wrapper 的 __init__ 不接受 pre_score（TypeError），
# 降级到无参版本。其他异常照常抛出。
WRAPPER_REGISTRY = (
    ("CliffWalking",             cliff_end_wrapper),
    ("ALE/Freeway",              freeway_end_wrapper),
    ("MountainCar",              mountaincar_score_wrapper),
    ("ALE/Assault",              assault_end_wrapper),
    ("ALE/Alien",                alien_end_wrapper),
    ("ALE/RoadRunner",           roadrunner_end_wrapper),
    ("Acrobot",                  acrobot_score_wrapper),
    ("CustomBreakout",           custombreakout_end_wrapper),
    ("highway",                  highway_end_wrapper),
    ("aFlappyBird",              aflappybird_reward_wrapper),
    ("tetris",                   tetris_end_wrapper),
    ("procgen_:procgen-bigfish",   bigfish_end_wrapper),
    ("procgen_:procgen-bossfight", bossfight_end_wrapper),
    ("procgen_:procgen-caveflyer", caveflyer_end_wrapper),
    ("procgen_:procgen-dodgeball", dodgeball_end_wrapper),
    ("procgen_:procgen-heist",     heist_end_wrapper),
    ("procgen_:procgen-leaper",    leaper_end_wrapper),
    ("procgen_:procgen-ninja",     ninja_end_wrapper),
)

# 需要奖励塑形的 ALE 游戏（死亡惩罚 + Seaquest 救人奖励）
_ATARI_SHAPED_GAMES = {'ALE/SpaceInvaders-v5', 'ALE/Seaquest-v5', 'ALE/Asterix-v5', 'ALE/Frostbite-v5'}

# Seaquest RAM 地址
_SEAQUEST_OXYGEN_ADDR = 102  # 氧气值 (0-64)
_SEAQUEST_DIVER_ADDR = 62    # 已救水手数 (0-6)


class GameEnv:
    """游戏环境封装类"""

    def __init__(self, env: gym.Env, action_meanings: list = None, frameskip: int = 1,
                 game_name: str = "", auto_respawn: bool = True, initial_lives: int = None,
                 action_set: str = "simple", partial_obs: bool = False,
                 end_on_life_loss: bool = False, initial_ram: dict = None):
        self.env = env
        self.game_name = game_name
        self.action_space_size = env.action_space.n  # 必须先设置，因为 _get_action_meanings 依赖它
        self.action_meanings = action_meanings or self._get_action_meanings()
        self.frameskip = frameskip  # 存储 frameskip 参数
        self.procgen = False
        if "procgen" in self.game_name:
            self.procgen = True
        self.auto_respawn = auto_respawn  # 死后自动复活（自动按 FIRE 继续下一条命）
        self.initial_lives = initial_lives  # 自定义初始生命数（通过 RAM hack）
        self.action_set = action_set  # "simple" 或 "full"
        self.partial_obs = partial_obs  # MiniGrid: 局部观察（不可见区域涂黑）
        self.end_on_life_loss = end_on_life_loss  # 丢命即终止（不复活，直接 game over）
        self.initial_ram: dict = {int(k): int(v) for k, v in (initial_ram or {}).items()}
        
        # 确定 RAM 映射（用于提取游戏状态文本）
        self.ram_map = None
        for key, ram_map in GAME_RAM_MAPS.items():
            if key.lower() in game_name.lower():
                self.ram_map = ram_map
                break
        
        # 确定 lives RAM 地址和写入偏移
        self._lives_addr = None
        self._game_key = None
        self._lives_write_offset = 0
        for key, addr in GAME_LIVES_RAM.items():
            if key.lower() in game_name.lower():
                self._lives_addr = addr
                self._game_key = key
                self._lives_write_offset = GAME_LIVES_RAM_WRITE_OFFSET.get(key, 0)
                break
        # 判断是否需要 FIRE 来重新发球
        self._needs_fire = any(g.lower() in game_name.lower() for g in AUTO_RESPAWN_GAMES)

        # Seaquest 游戏标志（用于潜在的游戏特定逻辑）
        self._is_seaquest = 'Seaquest' in game_name
        self._last_oxygen = None  # arm 后才设置
        self._last_seaquest_divers = 0  # 上一步的水手数

        self._is_centipede = 'Centipede' in game_name
        self._is_frostbite = 'Frostbite' in game_name
        self._is_riverraid = 'Riverraid' in game_name
        self._is_turmoil = 'Turmoil' in game_name
        self._is_berzerk = game_name == 'ALE/Berzerk-v5'

        self._last_riverraid_ram58 = None
        self._last_turmoil_ram117 = None

        self._death_detection_armed = False  # 跳帧完成后才启用

        # 上一次的 lives（用于检测丢球）
        self._last_lives = None

        # 检测是否为 MiniGrid 环境
        self._is_minigrid = 'MiniGrid' in game_name or game_name in CUSTOM_MINIGRID_IDS

        # MiniGrid 奖励塑形
        self._reward_shaper = None
        self._highlight_mask = None  # fog-of-war 可见格子缓存

        # FrozenLake 自定义得分：追踪本局中离终点最近的 BFS 距离
        self._fl_best_dist = None  # 本局最短 BFS 距离（越小越好）

        # Pong 自定义得分：追踪球位置和比分变化
        self._pong_prev_ball_x = None
        self._pong_prev_enemy_score = None
        self._pong_ball_going_right = False  # 上一帧球是否在向右移动（靠近我方）
        self._ever_scored = False # 曾经得分/接球标志（持久化）
        self._pong_balls_caught = 0  # 本局接球累计数（用于 score，只增不减）

        # Taxi-v3：是否已成功执行过 pickup（action=4 且非非法）
        self._taxi_correct_pickup = False

        # Tennis 接球检测（per-frame 轨迹检测：dy +→- 反转 = 玩家击球）
        # 物理依据：玩家在屏幕下方（ball_y 大），对手在上方（ball_y 小）。
        #   球向玩家飞 = dy>0，球向对手飞 = dy<0。
        #   dy:+→- 反转 = 球被玩家拍击回去（唯一对应玩家击球）。
        # 1981 Activision Tennis ROM 规则：所有击球自动过网、无出界 → dy 反转无歧义。
        # 由于 ALE frameskip 会跳过中间帧（反转可能落在被跳过的帧上），Tennis 创建时
        # 强制 ALE frameskip=1，由 step() 内部 loop frameskip 次以采样每帧 ball_y。
        self._is_tennis = 'Tennis' in game_name
        self._tennis_inner_frameskip = frameskip if self._is_tennis else 1
        self._tennis_last_ball_y = None  # 跨 step 边界保留最后一帧 ball_y

        # RAM 调试监视：设置后每 step 打印指定地址的变化
        # 格式: {addr: label}，例如 {114: "$F2 lives", 62: "$BE unknown"}
        self.ram_debug: dict = {}
        self._ram_debug_prev: dict = {}
        self._ram_debug_step: int = 0

        # initial_lives 延迟写入：某些游戏（如 DemonAttack）在 Step1 ROM 初始化时覆盖 RAM
        # 因此在第一次 step() 结束后再写一次，确保覆盖 ROM 的默认值
        self._initial_lives_pending = False

    def _get_action_meanings(self) -> list:
        """获取动作含义（PREDEFINED 优先于 ALE 原生名称）"""
        # 1. 优先使用 PREDEFINED（我们手动维护的语义正确名称）
        env_id = None
        if hasattr(self.env, 'spec') and self.env.spec is not None:
            env_id = self.env.spec.id

        for candidate in [env_id, self.game_name]:
            if candidate:
                predefined_actions = PREDEFINED_ACTION_MEANINGS.get(candidate)
                if predefined_actions and len(predefined_actions) == self.action_space_size:
                    return predefined_actions

        if env_id:
            # 尝试匹配前缀
            for key, actions in PREDEFINED_ACTION_MEANINGS.items():
                if env_id.startswith(key.replace('-v', '').replace('0', '').replace('1', '').replace('3', '')):
                    if len(actions) == self.action_space_size:
                        return actions

        # 2. 回退到环境原生 get_action_meanings
        if hasattr(self.env, 'get_action_meanings'):
            try:
                return self.env.get_action_meanings()
            except:
                pass

        if hasattr(self.env, 'unwrapped'):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'get_action_meanings'):
                try:
                    return unwrapped.get_action_meanings()
                except:
                    pass

        # 3. 默认返回编号
        return [f"Action_{i}" for i in range(self.action_space_size)]
    
    def get_action_info(self) -> Dict[int, str]:
        """获取动作编号到含义的映射（simple 模式仅返回核心动作）"""
        full = {i: meaning for i, meaning in enumerate(self.action_meanings)}
        env_id = None
        if hasattr(self.env, 'spec') and self.env.spec is not None:
            env_id = self.env.spec.id
        if self.action_set == "simple":
            if self.game_name in ACTION_SIMPLE_IDS :
                ids = ACTION_SIMPLE_IDS[self.game_name]
                return {i: full[i] for i in ids}
            elif env_id in ACTION_SIMPLE_IDS:
                ids=ACTION_SIMPLE_IDS[env_id]
                return {i: full[i] for i in ids}
        return full
    
    def get_ram_info(self) -> Optional[Dict[str, int]]:
        """从 RAM 提取游戏状态信息（仅 ALE 游戏支持）
        
        Returns:
            字典，包含游戏特定的状态信息（如球位置、paddle位置、分数等），
            如果不支持则返回 None
        """
        if self.ram_map is None:
            return None
        
        try:
            ale = self.env.unwrapped.ale
            ram = ale.getRAM()
            info = {}
            for key, addr in self.ram_map.items():
                if isinstance(addr, tuple):
                    # BCD 编码的多字节值（如分数）
                    val = 0
                    for a in addr:
                        val = val * 100 + int(f"{ram[a]:02x}", 10) if ram[a] < 0xa0 else val * 100
                    info[key] = val
                else:
                    info[key] = int(ram[addr])
            return info
        except Exception:
            return None
    
    def get_state_description(self, reward: float = 0.0, total_reward: float = 0.0,
                               step_count: int = 0) -> str:
        """生成给 LLM 看的文本状态描述
        
        Args:
            reward: 本步奖励
            total_reward: 累计奖励
            step_count: 当前步数
        
        Returns:
            结构化的文本描述
        """
        lines = [f"[Step {step_count}]"]
        lines.append(f"Reward: {reward:.1f} | Total: {total_reward:.1f}")
        
        ram_info = self.get_ram_info()
        if ram_info:
            for key, val in ram_info.items():
                lines.append(f"  {key}: {val}")
        
        lines.append(f"Actions: {self.get_action_info()}")
        return "\n".join(lines)
    
    def _get_lives(self) -> Optional[int]:
        """从 ALE API 获取当前 lives，RAM hack 仅作 fallback"""
        # 优先使用 ale.lives() 标准 API
        try:
            val = int(self.env.unwrapped.ale.lives())
            return val
        except Exception:
            pass

        # Fallback: RAM hack（仅当 ale.lives() 不可用时）
        if self._lives_addr is not None:
            try:
                ram = self.env.unwrapped.ale.getRAM()
                val = int(ram[self._lives_addr])
                mask = _LIVES_RAM_MASK.get(self._game_key)
                if mask is not None:
                    if isinstance(mask, tuple):
                        shift, m = mask
                        val = (val >> shift) & m
                    else:
                        val = val & mask
                return val
            except Exception:
                return None
        return None
    
    def _set_lives(self, lives: int):
        """通过 RAM hack 设置 lives（自动应用游戏特定的写入偏移）"""
        if self._lives_addr is None:
            return
        try:
            value = max(0, lives + self._lives_write_offset)
            self.env.unwrapped.ale.setRAM(self._lives_addr, value)
        except Exception:
            pass
    
    def _render(self) -> np.ndarray:
        """渲染当前帧。MiniGrid 在 partial_obs 模式下将不可见区域涂黑"""
        if self.procgen:
            image = self.env.render("rgb_array")
        else:
            image = self.env.render()
        if self._is_minigrid and self.partial_obs and image is not None:
            image = self._apply_fog_of_war(image)
        return image

    def _capture_berzerk_penultimate_frame(self, action: int, pre_state, post_state) -> np.ndarray:
        """重放单个 ALE step 的倒数第二原生帧，并恢复正式 step 的最终状态。"""
        ale_env = self.env.unwrapped
        ale = ale_env.ale
        try:
            ale.restoreSystemState(pre_state)
            ale_action = ale_env._action_set[action]
            for _ in range(self.frameskip - 1):
                ale.act(ale_action, 1.0)
            return np.array(ale.getScreenRGB(), copy=True)
        finally:
            ale.restoreSystemState(post_state)

    def _apply_fog_of_war(self, image: np.ndarray) -> np.ndarray:
        """对 MiniGrid 全局视图应用 fog of war，将 agent 看不到的格子涂黑"""
        uw = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env

        _, vis_mask = uw.gen_obs_grid()

        f_vec = uw.dir_vec
        r_vec = uw.right_vec
        top_left = (
            uw.agent_pos
            + f_vec * (uw.agent_view_size - 1)
            - r_vec * (uw.agent_view_size // 2)
        )

        # highlight_mask[x, y] = True 表示格子 (x, y) 对 agent 可见
        highlight_mask = np.zeros(shape=(uw.width, uw.height), dtype=bool)

        for vis_j in range(uw.agent_view_size):
            for vis_i in range(uw.agent_view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                abs_i, abs_j = int(abs_i), int(abs_j)
                if 0 <= abs_i < uw.width and 0 <= abs_j < uw.height:
                    highlight_mask[abs_i, abs_j] = True

        # 缓存 highlight_mask 供 reward shaper 使用
        self._highlight_mask = highlight_mask

        tile_size = uw.tile_size
        for x in range(uw.width):
            for y in range(uw.height):
                if not highlight_mask[x, y]:
                    image[y * tile_size:(y + 1) * tile_size,
                          x * tile_size:(x + 1) * tile_size] = 0

        return image

    def step(self, action: int) -> GameState:
        """执行动作并返回状态
        
        如果启用了 auto_respawn，当检测到丢命（lives 减少）时，
        会自动发送 FIRE 动作复活，并在 info 中标记 'life_lost': True
        """
        # MiniGrid: 手持物品时 Pickup → 立刻 game over
        # MiniGrid agent 只能携带一个物品，手持物品时 Pickup 会静默失败
        # 将其显式判定为 game over，避免玩家/AI 困惑
        if self._is_minigrid and action == 3:  # pickup
            uw = self.env.unwrapped
            if uw.carrying is not None:
                image = self._render()
                info = {"invalid_pickup": True, "carrying": uw.carrying.type}
                if self._reward_shaper:
                    reward = self._reward_shaper.step(0.0, True, False, info)
                else:
                    reward = 0.0
                return GameState(
                    image=image,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                    observation=None
                )

        # MiniGrid: toggle Box → 立刻 game over
        # Box.toggle 会把 box 替换为 box.contains（通常为 None），导致 box 消失、关卡无法完成
        # 其他非门对象（Ball/Key/Wall/Lava/Goal）的 toggle 只是 no-op（返回 False），不会改变状态
        if self._is_minigrid and action == 5:  # toggle
            uw = self.env.unwrapped
            from minigrid.core.constants import DIR_TO_VEC as MG_DIR_TO_VEC
            fwd_pos = uw.agent_pos + MG_DIR_TO_VEC[uw.agent_dir]
            fwd_cell = uw.grid.get(fwd_pos[0], fwd_pos[1])
            if fwd_cell is not None and fwd_cell.type == "box":
                # toggle box → box 消失 → 判定 game over
                image = self._render()
                info = {"invalid_toggle": True, "toggle_target": "box"}
                if self._reward_shaper:
                    reward = self._reward_shaper.step(0.0, True, False, info)
                else:
                    reward = 0.0
                return GameState(
                    image=image,
                    reward=reward,
                    terminated=True,
                    truncated=False,
                    info=info,
                    observation=None
                )

        # 记录执行前的 lives
        lives_before = self._get_lives()

        berzerk_pre_state = None
        berzerk_penultimate_frame = None
        berzerk_final_frame = None
        if self._is_berzerk and self.frameskip >= 2:
            berzerk_pre_state = self.env.unwrapped.ale.cloneSystemState()

        # Tennis: 按帧 loop（ALE 已设 frameskip=1），同时采样每帧 ball_y 用于击球检测。
        # 其他游戏：单次 env.step（ALE 内部按 self.frameskip 跳帧）。
        tennis_ball_y_trace = None
        if self._is_tennis and self._tennis_inner_frameskip > 1:
            cum_reward = 0.0
            tennis_ball_y_trace = []
            info = {}
            terminated = False
            truncated = False
            for _ in range(self._tennis_inner_frameskip):
                observation, r, terminated, truncated, info = self.env.step(action)
                cum_reward += float(r)
                try:
                    tennis_ball_y_trace.append(int(self.env.unwrapped.ale.getRAM()[15]))
                except Exception:
                    pass
                if terminated or truncated:
                    break
            reward = cum_reward
        elif self.procgen:
            observation, reward, terminated, info = self.env.step(action)
            truncated=False
        else:
            observation, reward, terminated, truncated, info = self.env.step(action)

        # ChopperCommand 的评估 score 按击杀次数计算；这里只记录事件，绝不改写 reward。
        # 清波 bonus 与最后一次击杀在同一正 reward 事件中结算，因此仍只计一次击杀。
        if self.game_name == "ALE/ChopperCommand-v5" and float(reward) > 0:
            info['chopper_kill_count'] = 1

        if berzerk_pre_state is not None:
            ale = self.env.unwrapped.ale
            berzerk_final_frame = np.array(ale.getScreenRGB(), copy=True)
            berzerk_post_state = ale.cloneSystemState()
            berzerk_penultimate_frame = self._capture_berzerk_penultimate_frame(
                action, berzerk_pre_state, berzerk_post_state
            )

        # MiniGrid 胜利信号：terminated 且原始 reward > 0（即拿到目标/物品）
        # 注意：必须在 reward shaper 替换 reward 前捕获
        if self._is_minigrid and terminated and float(reward) > 0:
            info['victory_signal'] = True

        # Taxi-v3 特殊逻辑
        if "Taxi" in self.game_name:
            info['victory_signal'] = False
            # 1. 获胜判定：terminated 且奖励为正 (成功送达 +20)
            if terminated and reward > 0:
                info['victory_signal'] = True

            # 2. 正确 pickup 检测：action=4 且非非法操作
            if action == 4 and reward != -10:
                self._taxi_correct_pickup = True

            # 3. 失败判定：
            # - 如果执行了 dropoff (5)，但没有触发获胜结束，则判定为失败（送错地方）
            # - 或者 pickup (4)/dropoff (5) 导致了非法操作惩罚 (-10)
            if action == 5 and not terminated:
                terminated = True
                info['victory_signal'] = False
            elif action in (4, 5) and reward == -10:
                terminated = True
                info['victory_signal'] = False
                info["taxi_illegal_action"] = True

            # 4. 每步注入 correct_pickup 信号（供 score_config 读）
            info["taxi_correct_pickup"] = self._taxi_correct_pickup

        # initial_lives / initial_ram 延迟写入：ROM 在 Step1 初始化后，在第一次 step 结束后覆盖写入
        if self._initial_lives_pending:
            if self.initial_lives is not None and self._lives_addr is not None:
                self._set_lives(self.initial_lives)
            if self.initial_ram:
                try:
                    for addr, val in self.initial_ram.items():
                        self.env.unwrapped.ale.setRAM(addr, max(0, min(255, val)))
                except Exception:
                    pass
            self._initial_lives_pending = False

        # RAM 调试监视
        if self.ram_debug:
            try:
                _ale = self.env.unwrapped.ale
                _ram = _ale.getRAM()
                self._ram_debug_step += 1
                for _addr, _label in self.ram_debug.items():
                    _cur = int(_ram[_addr])
                    _prev = self._ram_debug_prev.get(_addr)
                    if _prev is None or _cur != _prev:
                        print(f"[RAM][step {self._ram_debug_step:5d}] [{_addr:3d}] {_label}: {_prev} → {_cur}")
                        self._ram_debug_prev[_addr] = _cur
            except Exception:
                pass

        # Seaquest: 氧气钳制 + 浮出全额交付
        if self._is_seaquest:
            try:
                ale = self.env.unwrapped.ale
                ram = ale.getRAM()
                oxy = int(ram[_SEAQUEST_OXYGEN_ADDR])
                divers = int(ram[_SEAQUEST_DIVER_ADDR])

                # 1. 氧气钳制在 63（几乎满，永远不会耗尽）
                # 不能锁 64：ROM 浮出交付依赖氧气 <64 → 64 回充来触发
                if oxy < 63:
                    ale.setRAM(_SEAQUEST_OXYGEN_ADDR, 63)

                # 2. 浮出全额交付：ROM 原版不满 6 人只扣 1 人
                # 我们检测到 ROM 触发了浮出扣人（divers 减少），立即清零
                # 这样 reward_shaper 会看到 N→0，一次性结算全部水手
                if divers < self._last_seaquest_divers and self._last_seaquest_divers > 0:
                    ale.setRAM(_SEAQUEST_DIVER_ADDR, 0)
                    divers = 0

                self._last_seaquest_divers = divers
            except Exception:
                pass

        # 检测丢命（通过 lives 计数器，作为无死亡标志游戏的 fallback）
        lives_after = self._get_lives()
        life_lost = False
        if lives_before is not None and lives_after is not None:
            if lives_after < lives_before:
                life_lost = True
                info['life_lost'] = True
                info['lives_before'] = lives_before
                info['lives_after'] = lives_after

                if self.end_on_life_loss and self._death_detection_armed and not terminated:
                    # 丢命即终止：直接 game over，不复活
                    terminated = True
                elif self.auto_respawn and self._needs_fire and not terminated and not truncated:
                    # auto_respawn: 死后自动按 FIRE 复活
                    if self.procgen:
                        observation, r2, terminated, info2 = self.env.step(1)
                        truncated=terminated
                    else:
                        observation, r2, terminated, truncated, info2 = self.env.step(1)
                    reward += r2
                    # 保存重要的 life_lost 信息，防止被 info2 覆盖
                    life_lost_info = {
                        'life_lost': True,
                        'lives_before': lives_before,
                        'lives_after': lives_after,
                    }
                    info.update(info2)  # 先更新 info2
                    info.update(life_lost_info)  # 再覆盖回重要信息
                    info['auto_respawned'] = True
        
        self._last_lives = lives_after

        # Riverraid 和 Turmoil 特殊 RAM 丢命检测
        if self._is_riverraid:
            try:
                curr_ram58 = int(self.env.unwrapped.ale.getRAM()[58])
                if isinstance(self._last_riverraid_ram58, int) and self._last_riverraid_ram58 != 223 and curr_ram58 == 223:
                    life_lost = True
                    info['life_lost'] = True
                    if self.end_on_life_loss and self._death_detection_armed and not terminated:
                        terminated = True
                self._last_riverraid_ram58 = curr_ram58
            except Exception:
                pass

        if self._is_turmoil:
            try:
                curr_ram117 = int(self.env.unwrapped.ale.getRAM()[117])
                if isinstance(self._last_turmoil_ram117, int) and self._last_turmoil_ram117 == 0 and curr_ram117 == 1:
                    life_lost = True
                    info['life_lost'] = True
                    if self.end_on_life_loss and self._death_detection_armed and not terminated:
                        terminated = True
                self._last_turmoil_ram117 = curr_ram117
            except Exception:
                pass

        # Tennis 等无 lives 游戏：负 reward = 丢球，视为 life_lost
        if not life_lost and self.end_on_life_loss and reward < 0 and not terminated:
            if 'Tennis' in self.game_name:
                life_lost = True
                info['life_lost'] = True
                terminated = True

        # FrozenLake 增量得分：每靠近终点一步 +1，起始为 0 分
        if 'FrozenLake' in self.game_name:
            try:
                uw = self.env.unwrapped
                s = uw.s
                ncol = uw.ncol
                nrow = uw.nrow
                desc = uw.desc
                goal_row, goal_col = nrow - 1, ncol - 1
                curr_row, curr_col = s // ncol, s % ncol
                d_curr = _fl_bfs_dist(desc, nrow, ncol, curr_row, curr_col, goal_row, goal_col)
                if d_curr < self._fl_best_dist:
                    reward = float(self._fl_best_dist - d_curr)
                    self._fl_best_dist = d_curr
                else:
                    reward = 0.0
            except Exception:
                pass  # 异常时保留原始 reward

        # Pong 自定义得分：接球 +1，输一球 -1
        if 'Pong' in self.game_name:
            try:
                ram = self.env.unwrapped.ale.getRAM()
                ball_x = int(ram[PONG_RAM_MAP['ball_x']])
                enemy_score = int(ram[PONG_RAM_MAP['enemy_score']])

                custom_reward = 0.0

                # 检测比分变化：对方得分 -1
                if self._pong_prev_enemy_score is not None:
                    if enemy_score > self._pong_prev_enemy_score:
                        custom_reward -= 1.0

                # 检测接球：球方向从右变左 且 敌方未得分 = 我方成功接球 +1
                if self._pong_prev_ball_x is not None:
                    prev_x = self._pong_prev_ball_x
                    curr_x = ball_x
                    is_normal_move = abs(curr_x - prev_x) < 50
                    if is_normal_move and curr_x != prev_x:
                        curr_going_right = curr_x > prev_x
                        if self._pong_ball_going_right and not curr_going_right:
                            enemy_scored = (self._pong_prev_enemy_score is not None and
                                            enemy_score > self._pong_prev_enemy_score)
                            if not enemy_scored:
                                custom_reward += 1.0
                                self._ever_scored = True # 永久标记本局已接球
                                self._pong_balls_caught += 1
                        self._pong_ball_going_right = curr_going_right
                    elif not is_normal_move:
                        # 大跳跃（发球/重置），清除方向状态，避免跨越发球的虚假接球判定
                        self._pong_ball_going_right = False
                    # 球静止时（curr_x == prev_x）保留上一帧方向状态

                self._pong_prev_ball_x = ball_x
                self._pong_prev_enemy_score = enemy_score
                reward = custom_reward
            except Exception:
                pass  # 异常时保留原始 reward
            info["pong_balls_caught"] = self._pong_balls_caught

        # Tennis 接球检测（per-frame dy 反转）：
        # 把跨 step 的 ball_y 序列拼起来，扫描 dy 从 + 变 - 的位置——
        # 玩家在屏幕下方（ball_y 大），球朝玩家飞 dy>0，被击回 dy<0，反转点 = 击球瞬间。
        # 由于 ROM 物理：(a) 球自动过网、(b) 无出界、(c) 失分立即 terminate（end_on_life_loss），
        # 唯一能让 dy 由 + 变 - 的事件就是玩家击球，无歧义。
        if self._is_tennis and tennis_ball_y_trace:
            # 把跨 step 的最后一帧作为序列开头，保证 step 边界处的反转也能捕获
            trace = []
            if self._tennis_last_ball_y is not None:
                trace.append(self._tennis_last_ball_y)
            trace.extend(tennis_ball_y_trace)

            # 扫描相邻 dy 反转：dy_prev > 0 且 dy_curr < 0
            hit_detected = False
            for i in range(2, len(trace)):
                dy_prev = trace[i - 1] - trace[i - 2]
                dy_curr = trace[i] - trace[i - 1]
                if dy_prev > 0 and dy_curr < 0:
                    hit_detected = True
                    break
            if hit_detected:
                info['tennis_ball_returned'] = True

            self._tennis_last_ball_y = trace[-1]

        image = self._render()

        # MiniGrid 奖励塑形：替换原始稀疏奖励
        if berzerk_penultimate_frame is not None:
            image = np.maximum(berzerk_penultimate_frame, berzerk_final_frame)

        if self._reward_shaper:
            reward = self._reward_shaper.step(float(reward), terminated, truncated, info)

        return GameState(
            image=image,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
            observation=observation,
            ever_scored=self._ever_scored
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[GameState, Dict[str, Any]]:
        """重置环境
        
        Args:
            seed: 随机种子，用于保证初始状态的一致性（可选）
        
        如果设置了 initial_lives，会通过 RAM hack 修改初始生命数
        """
        if self.procgen:
            observation=self.env.reset()
            info={}
        elif seed is not None:
            observation, info = self.env.reset(seed=seed)
        else:
            observation, info = self.env.reset()
        
        # initial_lives / initial_ram 在 reset 时先写一次（对大多数游戏生效）
        # 某些游戏会在 Step1 被 ROM 覆盖，因此只要配置了初始覆盖项，就在 reset 时先 arm pending，
        # 这样即使 skip_initial_steps > 0，第一次 env.step() 后也能立即重新写回。
        has_initial_overrides = (
            (self.initial_lives is not None and self._lives_addr is not None)
            or bool(self.initial_ram)
        )
        self._initial_lives_pending = has_initial_overrides
        if self.initial_lives is not None and self._lives_addr is not None:
            self._set_lives(self.initial_lives)
            info['initial_lives_set'] = self.initial_lives
        if self.initial_ram:
            try:
                for addr, val in self.initial_ram.items():
                    self.env.unwrapped.ale.setRAM(addr, max(0, min(255, val)))
            except Exception:
                pass

        # 记录初始 lives
        self._last_lives = self._get_lives()
        self._ever_scored = False
        self._taxi_correct_pickup = False

        # 记录初始死亡标志（跳帧后可能非零，只检测 0→非零 变化）

        # Pong：重置追踪状态
        if 'Pong' in self.game_name:
            self._pong_prev_ball_x = None
            self._pong_prev_enemy_score = None
            self._pong_ball_going_right = False
            self._pong_balls_caught = 0

        # Tennis：重置接球检测状态
        if self._is_tennis:
            self._tennis_last_ball_y = None

        if self._is_riverraid:
            try:
                self._last_riverraid_ram58 = int(self.env.unwrapped.ale.getRAM()[58])
            except Exception:
                self._last_riverraid_ram58 = None

        if self._is_turmoil:
            try:
                self._last_turmoil_ram117 = int(self.env.unwrapped.ale.getRAM()[117])
            except Exception:
                self._last_turmoil_ram117 = None

        # FrozenLake：重置距离追踪，初始化为起始位置的 BFS 距离（起始分 0）
        if 'FrozenLake' in self.game_name:
            try:
                uw = self.env.unwrapped
                desc = uw.desc
                nrow, ncol = uw.nrow, uw.ncol
                s = uw.s
                start_row, start_col = s // ncol, s % ncol
                goal_r, goal_c = nrow - 1, ncol - 1
                self._fl_best_dist = _fl_bfs_dist(desc, nrow, ncol, start_row, start_col, goal_r, goal_c)
            except Exception:
                self._fl_best_dist = float('inf')


        image = self._render()

        # 奖励塑形初始化
        self._reward_shaper = None
        if self._is_minigrid:
            from src.env.reward_shaper import MiniGridRewardShaper
            self._reward_shaper = MiniGridRewardShaper(self)
        elif self.game_name in _ATARI_SHAPED_GAMES:
            from src.env.reward_shaper import ATARIRewardShaper
            self._reward_shaper = ATARIRewardShaper(self)
        if self._reward_shaper:
            self._reward_shaper.reset()

        # 重置死亡检测状态（跳帧后需调用 arm_death_detection 启用）
        self._death_detection_armed = False

        state = GameState(
            image=image,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
            observation=observation
        )
        return state, info

    def reset_tracking_state(self) -> None:
        """preload 序列结束后调用：清零 episode 跟踪状态，隔离 preload 和 AI 阶段。

        不重置游戏 env 状态（不调用 env.reset()），只清零在 episode 内累积的
        跟踪变量，使 AI 阶段的 score / pass 判定不受 preload 阶段影响。

        注意：调用方在此方法之后再调用 session.score_state.reset(session.env)，
        因为 MiniGrid 的 score_state 需要读取 _reward_shaper.shortest_path，
        而后者在 _reward_shaper.reset() 里以当前位置重新计算。
        """
        self._ever_scored = False
        self._taxi_correct_pickup = False

        if 'Pong' in self.game_name:
            self._pong_prev_ball_x = None
            self._pong_prev_enemy_score = None
            self._pong_ball_going_right = False
            self._pong_balls_caught = 0

        if self._is_tennis:
            self._tennis_last_ball_y = None

        # reward_shaper 重置：清零累积的探索/效率状态，以当前 env 状态为新基准。
        # MiniGrid 会重新从当前 agent_pos 计算 BFS shortest_path，
        # 供后续 score_state.reset() 读取；ATARI shaper 则清零死亡惩罚等中间状态。
        if self._reward_shaper:
            self._reward_shaper.reset()

        game_name_lower=self.game_name.lower()
        if "assault" in game_name_lower or "alien" in game_name_lower:
            self.env.num_steps=1

    def arm_death_detection(self):
        """跳帧完成后调用：读取当前氧气值作为基线，启用即时死亡检测

        Seaquest 特判：开局氧气=0（跳帧后=64），只监控 >0 → 0 的变化。
        """
        if self._is_seaquest:
            try:
                self._last_oxygen = int(self.env.unwrapped.ale.getRAM()[_SEAQUEST_OXYGEN_ADDR])
            except Exception:
                self._last_oxygen = 0

        # 标记 initial_lives / initial_ram 待写入：某些游戏（如 DemonAttack）在 Step1 由 ROM 初始化
        # 需要在第一次 step() 结束后再写，才能覆盖 ROM 的默认值
        if self.initial_lives is not None and self._lives_addr is not None:
            self._initial_lives_pending = True
        if self.initial_ram:
            self._initial_lives_pending = True  # 复用同一个 pending 标志
        self._death_detection_armed = True

    def close(self):
        """关闭环境"""
        self.env.close()


def create_env(name: str, cfg: Optional[Dict[str, Any]] = None) -> GameEnv:
    """创建游戏环境
    
    Args:
        name: 游戏名称，例如 "ALE/Pong-v5", "Taxi-v3", "MiniGrid-Empty-16x16-v0"
        cfg: 配置字典，支持以下参数:
            - frameskip: 环境内部帧跳过数（仅 ALE 游戏支持），
                         1=每个action只执行1帧（反馈最快），
                         4=每个action执行4帧（反馈较慢）。
                         对 LLM 玩建议设为 4（每步有明显变化，减少 API 调用）
            - auto_respawn: 死后是否自动复活（自动按 FIRE 继续下一条命），默认 True
                        开启后 LLM 无需知道 FIRE 机制，丢球后自动续命
            - initial_lives: 自定义初始生命数（通过 RAM hack），默认 None（使用游戏默认值）
                            例如设为 99 可获得 99 条命
            - render_mode: 渲染模式，默认 "rgb_array"
            - resolution: 分辨率 (height, width)，默认 None (原始分辨率)
            - is_slippery: 仅用于 FrozenLake，是否滑，默认 None（使用环境默认）
    Returns:
        GameEnv: 封装后的游戏环境
    """
    if cfg is None:
        cfg = {}
    
    # 解析配置
    frameskip = cfg.get('frameskip', 4)  # 默认4，每步有明显变化，适合 LLM
    render_mode = cfg.get('render_mode', 'rgb_array')
    resolution = cfg.get('resolution', None)
    is_slippery = cfg.get('is_slippery', None)  # 用于 FrozenLake
    map_name = cfg.get('map_name', None)  # 用于 FrozenLake，"4x4" 或 "8x8"
    auto_respawn = cfg.get('auto_respawn', False)  # 死后自动复活（默认关闭，前端按需开启）
    initial_lives = cfg.get('initial_lives', None)  # 自定义初始生命数
    initial_ram = cfg.get('initial_ram', None)      # 自定义初始 RAM：{addr: value, ...}
    end_on_life_loss = cfg.get('end_on_life_loss', False)  # 丢命即终止
    action_set = cfg.get('action_set', 'simple')  # "simple" 或 "full"
    ale_mode = cfg.get('mode', None)  # ALE game mode variant (e.g. Frostbite: 0 or 2)
    ale_difficulty = cfg.get('difficulty', None)  # ALE difficulty (e.g. Seaquest: 0 or 1)
    # MiniGrid 局部观察：默认开启（不可见区域涂黑），可通过前端关闭
    is_minigrid = name.startswith('MiniGrid') or name in CUSTOM_MINIGRID_IDS
    partial_obs = cfg.get('partial_obs', True if is_minigrid else False)
    pre_score=cfg.get("pre_score",0)
    seed=cfg.get("seed",41)
    max_steps=cfg.get("max_steps",1)
    repeat=cfg.get("repeat",1)
    
    # 判断是否为 ALE 游戏
    is_ale_game = name.startswith('ALE/')
    
    last_error = None
    env = None


    if is_ale_game:
        # ALE 游戏：尝试多种名称格式（向后兼容）
        base_name = name.replace('ALE/', '').replace('-v5', '').replace('-v4', '')
        
        env_names_to_try = [
            name,  # 原始名称
            f'ALE/{base_name}-v5',  # 标准格式
            f'ALE/{base_name}-v4',  # 旧版本格式
            base_name,  # 不带前缀
        ]
        
        for env_name in env_names_to_try:
            try:
                # 构建 ALE 环境参数
                # Tennis 强制 ALE frameskip=1：让 GameEnv.step 内部按帧 loop，
                # 这样能拿到每一帧的 RAM[15] 用于检测 dy 反转（接球事件）。
                # 行为等价（GameEnv 内部 loop N 次 ≡ ALE 一次 frameskip=N），只是开销略增。
                ale_frameskip = 1 if 'Tennis' in name else frameskip
                ale_kwargs = {
                    'render_mode': render_mode,
                    'frameskip': ale_frameskip,
                    'repeat_action_probability': 0.0,  # 关闭粘性动作
                }
                # Only pass mode/difficulty if explicitly configured; many ALE games
                # don't support mode=0 (e.g. Berzerk, DemonAttack, BattleZone).
                if ale_mode is not None:
                    ale_kwargs['mode'] = int(ale_mode)
                if ale_difficulty is not None:
                    ale_kwargs['difficulty'] = int(ale_difficulty)
                try:
                    env = gym.make(env_name, **ale_kwargs)
                    # 验证粘性动作是否被禁用
                    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'ale'):
                        rap = env.unwrapped.ale.getFloat("repeat_action_probability")
                        print(f"✅ repeat_action_probability={rap} (粘性动作{'已禁用' if rap == 0 else '未禁用!'})")
                    break
                except TypeError as te:
                    # 如果不支持某些参数，逐步去掉
                    print(f"⚠️ 部分参数不支持，尝试简化: {te}")
                    try:
                        env = gym.make(env_name, render_mode=render_mode, frameskip=ale_frameskip)
                        print(f"⚠️ 警告: 粘性动作可能未禁用！（默认 repeat_action_probability=0.25）")
                        break
                    except TypeError:
                        env = gym.make(env_name, render_mode=render_mode)
                        break
            except Exception as e:
                last_error = e
                continue
        
        if env is None:
            raise RuntimeError(
                f"无法创建 ALE 环境 '{name}'。\n"
                f"尝试了以下名称: {env_names_to_try}\n"
                f"最后错误: {last_error}\n"
                f"提示: 请确保已安装 'gymnasium[atari]' 和 'gymnasium[accept-rom-license]'，"
                f"并且已导入 ale_py 模块。"
            )
    elif "procgen" in name:
        try:
            # 处理特殊参数（如 FrozenLake 的 is_slippery）
            make_kwargs = {'render_mode': render_mode,'rand_seed':seed}
            env = gym_origin.make(name,**make_kwargs)
        except Exception as e:
            last_error = e
            raise RuntimeError(
                f"无法创建环境 '{name}'。\n"
                f"最后错误: {last_error}\n"
                f"提示: 请确保已安装相应的环境包。\n"
                f"  - 基础: pip install gym\n"
                # f"  - MiniGrid: pip install minigrid\n"
                # f"  - Atari: pip install 'gymnasium[atari]' 'gymnasium[accept-rom-license]' ale-py"
            )

    else:
        # 非 ALE 游戏：直接使用 gym.make，不走 ALE fallback
        # 对于非 ALE 游戏，frameskip 参数通常不支持，所以不传
        try:
            # 处理特殊参数（如 FrozenLake 的 is_slippery）
            make_kwargs = {'render_mode': render_mode}
            if 'FrozenLake' in name:
                if is_slippery is not None:
                    make_kwargs['is_slippery'] = is_slippery
                make_kwargs['map_name'] = map_name or '8x8'

            if "carracing" in name.lower():
                make_kwargs["continuous"]=False

            if "highway" in name.lower():
                lanes=cfg.get("lanes_count",4)
                density=cfg.get("vehicles_density",2)
                make_kwargs["config"]={"action":{'type': 'DiscreteMetaAction'},"render_agent":False,"simulation_frequency":5,'policy_frequency':5,"vehicles_count": 100,"lanes_count": lanes,"duration":60,'ego_spacing': 1,'vehicles_density': density,"offscreen_rendering":True}

            if "merge" in name.lower():
                pass

            if "roundabout" in name.lower():
                pass

            if "intersection" in name.lower():
                make_kwargs["config"]={"action":{'type': 'DiscreteAction'},'initial_vehicle_count': 25,"simulation_frequency":5,'policy_frequency':5}

                
            # 自定义 MiniGrid 环境参数转发
            if name == 'CustomLavaCrossing-v0':
                for k in ('size', 'num_crossings'):
                    if k in cfg:
                        make_kwargs[k] = int(cfg[k])
            elif name == 'CustomMultiRoom-v0':
                for k in ('grid_size', 'num_rooms', 'max_room_size'):
                    if k in cfg:
                        make_kwargs[k] = int(cfg[k])
            elif name == 'CustomUnlockPickup-v0':
                for k in ('room_size', 'num_rows', 'num_cols'):
                    if k in cfg:
                        make_kwargs[k] = int(cfg[k])
            
            if "vizdoom" in name.lower():
                make_kwargs["screen_resolution"] = "RES_640X480"
                # make_kwargs["render_hud"] = "true"
                # make_kwargs["render_crosshair"] = "true"
                make_kwargs["render_weapon"] = "true"
                make_kwargs['render_decals'] = "true"
                make_kwargs["render_particles"] = "true"
                make_kwargs["window_visible"] = "true"
                make_kwargs["doom_map"]="map03"


            env = gym.make(name, **make_kwargs)

            if "pendulum" in name.lower():
                env = DiscretizeAction(env, [-2.0, -1.0, 0.0, 1.0, 2.0])
        except Exception as e:
            last_error = e
            raise RuntimeError(
                f"无法创建环境 '{name}'。\n"
                f"最后错误: {last_error}\n"
                f"提示: 请确保已安装相应的环境包。\n"
                f"  - 基础: pip install gymnasium\n"
                f"  - MiniGrid: pip install minigrid\n"
                f"  - Atari: pip install 'gymnasium[atari]' 'gymnasium[accept-rom-license]' ale-py\n"
                f"  - Box2D: pip install swig 'gymnasium[box2d]'"
            )

    # 应用分辨率调整
    if resolution is not None:
        env = ResizeObservation(env, shape=resolution)
    
    # 创建封装
    game_env = GameEnv(env, frameskip=frameskip, game_name=name,
                       auto_respawn=auto_respawn, initial_lives=initial_lives,
                       action_set=action_set, partial_obs=partial_obs,
                       end_on_life_loss=end_on_life_loss, initial_ram=initial_ram)

    # RAM 调试监视（通过 cfg["ram_debug"] 启用）
    # 格式: {addr: label}，例如 {114: "$F2 lives", 62: "$BE unknown"}
    ram_debug = cfg.get("ram_debug", {}) if cfg else {}
    if ram_debug:
        game_env.ram_debug = ram_debug
        print(f"[RAM Debug] 监视地址: {ram_debug}")

    # 打印动作空间信息
    print(f"\n{'='*60}")
    print(f"环境创建成功: {name}")
    print(f"{'='*60}")
    print(f"动作空间大小: {game_env.action_space_size}")
    if partial_obs:
        print(f"[partial_obs] ON (fog of war)")

    if auto_respawn and game_env._needs_fire:
        print(f"[auto_respawn] ON")
    if initial_lives is not None:
        print(f"[initial_lives] {initial_lives} (RAM hack)")
    if initial_ram:
        print(f"[initial_ram] {initial_ram}")
    print(f"\n动作含义:")
    print("-" * 60)
    for i, meaning in enumerate(game_env.action_meanings):
        print(f"  {i}: {meaning}")
    print(f"{'='*60}\n")
    
    for prefix, wrapper_cls in WRAPPER_REGISTRY:
        if name.startswith(prefix):
            if "bossfight" in name.lower() or "heist" in name.lower() or "dodgeball" in name.lower():
                game_env.env=wrapper_cls(game_env.env,seed)
            elif "assault" in name.lower():
                game_env.env=wrapper_cls(game_env.env,max_steps,repeat)
            elif "alien" in name.lower():
                game_env.env=wrapper_cls(game_env.env,pre_score,max_steps,repeat)
            else:
                try:
                    game_env.env = wrapper_cls(game_env.env, pre_score)
                except TypeError:
                    game_env.env = wrapper_cls(game_env.env)
                break


    return game_env


def do_action(env: GameEnv, action: int, repeat: int = 1, 
               action_repeat: Optional[int] = None, noop_fill: bool = False) -> GameState:
    """执行动作
    
    Args:
        env: 游戏环境
        action: 动作编号
        repeat: 总重复执行次数，用于加快游戏速度（跳过多个frame）
               例如 repeat=4 表示总共执行4次，游戏会运行得更快
        action_repeat: 指定动作重复执行的次数（仅在 noop_fill=True 时有效）
                       如果为 None，则等于 repeat（传统模式）
        noop_fill: 是否用 NOOP (action=0) 填充剩余次数
                   如果 True: 先执行 action 共 action_repeat 次，然后用 NOOP 填充剩余次数
                   如果 False: 所有次数都执行 action（传统模式）
    
    Returns:
        GameState: 当前状态（包含累计奖励）
    
    示例:
        # 传统模式：action 执行 4 次
        do_action(env, action=2, repeat=4)
        
        # 新模式：action 执行 1 次，然后用 NOOP 填充 3 次
        do_action(env, action=2, repeat=4, action_repeat=1, noop_fill=True)
    """
    if action < 0 or action >= env.action_space_size:
        raise ValueError(f"动作 {action} 超出范围 [0, {env.action_space_size-1}]")
    
    if repeat < 1:
        raise ValueError("repeat 必须 >= 1")
    
    # 确定实际执行模式
    if noop_fill and action_repeat is not None:
        # 新模式：action 执行 action_repeat 次，NOOP 填充剩余
        action_repeat = max(1, min(repeat, action_repeat))
        noop_count = repeat - action_repeat
    else:
        # 传统模式：所有次数都执行 action
        action_repeat = repeat
        noop_count = 0
    
    total_reward = 0.0
    # 累加 reward_breakdown 的累加型字段（last_state.info 默认只保留最后一帧）
    # ScoreState 现在嗅 `shaped_total - death`，所以必须把这两个累加；
    # raw_reward / original_reward 也累加以保持 batch_viewer / result.json 向后兼容
    cum_raw_reward = 0.0
    cum_original_reward = 0.0
    cum_shaped_total = 0.0
    cum_death = 0.0
    cum_chopper_kills = 0
    has_raw_bd = False
    has_orig_bd = False
    has_shaped_bd = False
    has_death_bd = False
    last_state = None
    merged_info = {}

    def _accum_breakdown(info: dict):
        nonlocal cum_raw_reward, cum_original_reward, cum_shaped_total, cum_death
        nonlocal cum_chopper_kills
        nonlocal has_raw_bd, has_orig_bd, has_shaped_bd, has_death_bd
        bd = info.get('reward_breakdown') or {}
        if 'raw_reward' in bd:
            cum_raw_reward += float(bd['raw_reward'])
            has_raw_bd = True
        if 'original_reward' in bd:
            cum_original_reward += float(bd['original_reward'])
            has_orig_bd = True
        if 'shaped_total' in bd:
            cum_shaped_total += float(bd['shaped_total'])
            has_shaped_bd = True
        if 'death' in bd:
            cum_death += float(bd['death'])
            has_death_bd = True
        cum_chopper_kills += int(info.get('chopper_kill_count', 0))

    # 先执行指定 action
    for _ in range(action_repeat):
        state = env.step(action)
        total_reward += state.reward
        _accum_breakdown(state.info)
        # 合并 info（保留丢球等重要信息）
        if state.info.get('life_lost'):
            merged_info['life_lost'] = True
            merged_info['lives_before'] = state.info.get('lives_before')
            merged_info['lives_after'] = state.info.get('lives_after')
            if state.info.get('auto_respawned'):
                merged_info['auto_respawned'] = True
        if state.info.get('tennis_ball_returned'):
            merged_info['tennis_ball_returned'] = True
        last_state = state

        # 如果游戏结束，提前退出
        if state.terminated or state.truncated:
            last_state.reward = total_reward
            last_state.info.update(merged_info)
            # 把累计字段写回 breakdown，覆盖最后一帧的"per-step"值
            if has_raw_bd or has_orig_bd or has_shaped_bd or has_death_bd:
                bd = last_state.info.setdefault('reward_breakdown', {})
                if has_raw_bd:
                    bd['raw_reward'] = cum_raw_reward
                if has_orig_bd:
                    bd['original_reward'] = cum_original_reward
                if has_shaped_bd:
                    bd['shaped_total'] = cum_shaped_total
                if has_death_bd:
                    bd['death'] = cum_death
            if cum_chopper_kills:
                last_state.info['chopper_kill_count'] = cum_chopper_kills
            return last_state

    # 用 NOOP 填充剩余次数
    for _ in range(noop_count):
        if "highway" in env.game_name.lower():
            state=env.step(1)
        elif "procgen" in env.game_name.lower():
            state=env.step(4)
        else:
            state = env.step(0)  # NOOP
        total_reward += state.reward
        _accum_breakdown(state.info)
        if state.info.get('life_lost'):
            merged_info['life_lost'] = True
            merged_info['lives_before'] = state.info.get('lives_before')
            merged_info['lives_after'] = state.info.get('lives_after')
            if state.info.get('auto_respawned'):
                merged_info['auto_respawned'] = True
        if state.info.get('tennis_ball_returned'):
            merged_info['tennis_ball_returned'] = True
        last_state = state

        # 如果游戏结束，提前退出
        if state.terminated or state.truncated:
            break

    # 更新累计奖励（last_state 一定不为 None，因为 repeat >= 1）
    assert last_state is not None, "执行动作后状态不应为 None"
    last_state.reward = total_reward
    last_state.info.update(merged_info)
    # 把累计字段写回 breakdown
    if has_raw_bd or has_orig_bd or has_shaped_bd or has_death_bd:
        bd = last_state.info.setdefault('reward_breakdown', {})
        if has_raw_bd:
            bd['raw_reward'] = cum_raw_reward
        if has_orig_bd:
            bd['original_reward'] = cum_original_reward
        if has_shaped_bd:
            bd['shaped_total'] = cum_shaped_total
        if has_death_bd:
            bd['death'] = cum_death
    if cum_chopper_kills:
        last_state.info['chopper_kill_count'] = cum_chopper_kills

    return last_state


# 常用游戏配置预设
GAME_CONFIGS = {
    'pong': {
        'frameskip': 1,  # 快速反馈
        'repeat_action': 1,  # 弹球游戏需要精确控制，不重复
    },
    'breakout': {
        'frameskip': 1,
        'repeat_action': 2,  # 可以稍微加快
    },
    'space_invaders': {
        'frameskip': 1,
        'repeat_action': 2,
    },
    'beam_rider': {
        'frameskip': 1,
        'repeat_action': 2,
    },
    'qbert': {
        'frameskip': 1,
        'repeat_action': 1,
    },
}


def get_game_config(game_name: str) -> Dict[str, Any]:
    """获取游戏推荐配置"""
    game_key = game_name.lower().replace('ale/', '').replace('-v5', '').replace('_', '')
    
    for key, config in GAME_CONFIGS.items():
        if key in game_key:
            return config.copy()
    
    # 默认配置
    return {
        'frameskip': 1,
        'repeat_action': 1,
    }
