"""
Breakout 环境封装器
提供与 env_wrapper.py 兼容的接口
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .breakout_env import BreakoutEnv, make_breakout


@dataclass
class GameState:
    """游戏状态信息（与 env_wrapper.py 兼容）"""
    image: np.ndarray  # 当前帧图片 (RGB array)
    reward: float  # 奖励
    terminated: bool  # 是否结束
    truncated: bool  # 是否截断
    info: Dict[str, Any]  # 额外信息
    observation: np.ndarray  # 原始观察值


class BreakoutGameEnv:
    """Breakout 游戏环境封装类（与 env_wrapper.GameEnv 接口兼容）"""
    
    def __init__(self, env: BreakoutEnv):
        self.env = env
        self.game_name = "CustomBreakout"
        self.action_space_size = env.action_space.n
        self.frameskip = env.frameskip
        self.auto_respawn = False  # CustomBreakout 不需要 auto_respawn
        self.initial_lives = env.initial_lives
        self._needs_fire = False
        
    def reset(self, seed: Optional[int] = None) -> Tuple[GameState, Dict[str, Any]]:
        """重置环境
        
        Args:
            seed: 随机种子（可选），用于保证初始状态的一致性
        """
        # BreakoutEnv.reset() 支持 seed 参数
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        state = GameState(
            image=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
            observation=obs
        )
        return state, info
    
    def step(self, action: int) -> GameState:
        """执行一步动作"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["pass"]=False
        info["ending"]="timeout"
        if info.get("bounce",0)>0:
            info["pass"]=True
        if terminated:
            if reward>0:
                info["ending"]="victory"
            else:
                info["ending"]="gameover"
        reward*=(100/2)
        info["score"]=reward
        return GameState(
            image=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            observation=obs
        )
    
    def get_action_info(self) -> Dict[int, str]:
        """获取动作信息"""
        meanings = self.env.get_action_meanings()
        return {i: meaning for i, meaning in enumerate(meanings)}
    
    def get_ram_info(self) -> Dict[str, Any]:
        """获取 RAM 信息（CustomBreakout 直接从环境获取）"""
        # CustomBreakout 不使用 RAM，而是直接从环境状态获取
        if hasattr(self.env, 'paddle_x') and hasattr(self.env, 'ball_x'):
            return {
                'paddle_x': int(self.env.paddle_x),
                'ball_x': int(self.env.ball_x),
                'ball_y': int(self.env.ball_y),
                'lives': self.env.lives,
                'score': self.env.score,
            }
        return {}
    
    def reset_tracking_state(self) -> None:
        """No-op：与 env_wrapper.GameEnv.reset_tracking_state 的接口保持一致。

        tools/batch.py 在 replay preload 序列后无条件调此方法（ee3931e, 2026-06-13）。
        GameEnv 里清的是 wrapper 层累积字段（_ever_scored / _pong_balls_caught /
        _taxi_correct_pickup / _tennis_last_ball_y / _reward_shaper），
        CustomBreakout 一个都没有；preload→AI 阶段的分数隔离由上游 batch.py 里
        session.score_state.reset() + session.episode_reward=0.0 完成，此处不用做事。
        """
        pass

    def close(self):
        """关闭环境"""
        self.env.close()


def create_breakout_env(cfg: Optional[Dict[str, Any]] = None) -> BreakoutGameEnv:
    """创建 Breakout 环境
    
    Args:
        cfg: 配置字典，支持以下参数:
            - paddle_width: 木板宽度（像素），默认 20
            - paddle_height: 木板高度（像素），默认 6
            - paddle_speed: 木板移动速度（像素/帧），默认 4
            - ball_radius: 小球半径（像素），默认 3
            - ball_speed_x: 小球水平速度（像素/帧），默认 2.0
            - ball_speed_y: 小球垂直速度（像素/帧），默认 2.0
            - ball_speed: 小球速度（同时设置 x 和 y），默认 2.0
            - brick_rows: 砖块行数，默认 6
            - brick_cols: 砖块列数，默认 10
            - brick_height: 砖块高度（像素），默认 8
            - brick_area_top_offset: 砖块区域距离顶部的偏移（像素），默认 10
            - initial_lives: 初始生命数，默认 5
            - launch_angle_mode: 发球角度模式 ("random"/"fixed"/"fixed_angle")，默认 "random"
            - launch_angle: 固定发球角度（弧度），默认 0.3
            - screen_width: 屏幕宽度，默认 160
            - screen_height: 屏幕高度，默认 210
            - frameskip: 帧跳过数（每个 action 执行多少帧），默认 4
            - render_mode: 渲染模式，默认 "rgb_array"
            - max_episode_steps: 最大步数，默认 10000
    
    Returns:
        BreakoutGameEnv: 封装后的游戏环境
    """
    if cfg is None:
        cfg = {}
    
    # 提取参数
    paddle_width = cfg.get('paddle_width', 20)
    paddle_height = cfg.get('paddle_height', 6)
    paddle_speed = cfg.get('paddle_speed', 4)
    ball_radius = cfg.get('ball_radius', 3)
    
    # 小球速度（支持单独设置或统一设置）
    ball_speed = cfg.get('ball_speed', 2.0)
    ball_speed_x = cfg.get('ball_speed_x', ball_speed)
    ball_speed_y = cfg.get('ball_speed_y', ball_speed)
    
    brick_rows = cfg.get('brick_rows', 6)
    brick_cols = cfg.get('brick_cols', 10)
    brick_height = cfg.get('brick_height', 8)
    brick_area_top_offset = cfg.get('brick_area_top_offset', 10)
    initial_lives = cfg.get('initial_lives', 5)
    launch_angle_mode = cfg.get('launch_angle_mode', 'random')
    launch_angle = cfg.get('launch_angle', 0.3)
    screen_width = cfg.get('screen_width', 160)
    screen_height = cfg.get('screen_height', 210)
    frameskip = cfg.get('frameskip', 4)
    render_mode = cfg.get('render_mode', 'rgb_array')
    max_episode_steps = cfg.get('max_episode_steps', 10000)
    
    # 创建环境
    env = BreakoutEnv(
        paddle_width=paddle_width,
        paddle_height=paddle_height,
        paddle_speed=paddle_speed,
        ball_radius=ball_radius,
        ball_speed_x=ball_speed_x,
        ball_speed_y=ball_speed_y,
        brick_rows=brick_rows,
        brick_cols=brick_cols,
        brick_height=brick_height,
        brick_area_top_offset=brick_area_top_offset,
        initial_lives=initial_lives,
        launch_angle_mode=launch_angle_mode,
        launch_angle=launch_angle,
        screen_width=screen_width,
        screen_height=screen_height,
        frameskip=frameskip,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    
    return BreakoutGameEnv(env)


def do_action(env: BreakoutGameEnv, action: int, repeat: int = 1) -> GameState:
    """执行动作（支持重复执行）
    
    Args:
        env: 游戏环境
        action: 动作编号
        repeat: 重复执行次数（注意：环境内部已有 frameskip）
    
    Returns:
        GameState: 最后一步的游戏状态
    """
    state = None
    for _ in range(repeat):
        state = env.step(action)
        if state.terminated or state.truncated:
            break
    return state
