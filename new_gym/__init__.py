"""
自定义游戏环境包（Breakout + MiniGrid 变体）
"""

from .breakout_env import BreakoutEnv, make_breakout
from .breakout_wrapper import (
    BreakoutGameEnv,
    create_breakout_env,
    do_action,
    GameState,
)

# --- 注册自定义 MiniGrid 环境 ---
from gymnasium.envs.registration import register

register(
    id="CustomLavaCrossing-v0",
    entry_point="new_gym.custom_lava_crossing:CustomLavaCrossingEnv",
    kwargs={"size": 11, "num_crossings": 5},
)

register(
    id="CustomMultiRoom-v0",
    entry_point="new_gym.custom_multiroom:CustomMultiRoomEnv",
    kwargs={"grid_size": 25, "num_rooms": 6, "max_room_size": 10},
)

register(
    id="CustomUnlockPickup-v0",
    entry_point="new_gym.custom_unlock_pickup:CustomUnlockPickupEnv",
    kwargs={"room_size": 6, "num_rows": 1, "num_cols": 2},
)

__all__ = [
    'BreakoutEnv',
    'make_breakout',
    'BreakoutGameEnv',
    'create_breakout_env',
    'do_action',
    'GameState',
]

__version__ = '1.0.0'
