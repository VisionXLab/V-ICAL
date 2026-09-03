"""
自定义 Breakout 游戏环境
解决原版 ALE/Breakout-v5 的问题：
1. 支持自定义参数（小球速度、木板长度等）
2. 小球掉出屏幕时正确结束游戏（减少生命值）
3. 修复木板惯性问题（noop 时立即停止）
4. 保留 frameskip 等参数支持
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import cv2


class BreakoutEnv(gym.Env):
    """自定义 Breakout 环境"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        # 游戏参数
        paddle_width: int = 20,  # 木板宽度（像素）
        paddle_height: int = 6,  # 木板高度（像素）
        paddle_speed: int = 4,  # 木板移动速度（像素/帧）
        ball_radius: int = 3,  # 小球半径（像素）
        ball_speed_x: float = 2.0,  # 小球水平速度（像素/帧）
        ball_speed_y: float = 2.0,  # 小球垂直速度（像素/帧）
        brick_rows: int = 6,  # 砖块行数
        brick_cols: int = 10,  # 砖块列数
        brick_height: int = 8,  # 砖块高度（像素）
        brick_area_top_offset: int = 10,  # 砖块区域距离顶部的偏移（像素，越大砖块越靠下）
        initial_lives: int = 5,  # 初始生命数
        launch_angle_mode: str = "random",  # 发球角度模式: "random"=随机, "fixed"=固定, "fixed_angle"=指定角度
        launch_angle: float = 0.3,  # 固定发球角度（弧度），仅在 launch_angle_mode="fixed_angle" 时使用
        # 环境参数
        screen_width: int = 160,  # 屏幕宽度
        screen_height: int = 210,  # 屏幕高度
        frameskip: int = 4,  # 帧跳过数（每个 action 执行多少帧）
        render_mode: Optional[str] = "rgb_array",
        # 游戏机制
        max_episode_steps: int = 10000,  # 最大步数
    ):
        super().__init__()
        
        # 保存参数
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.paddle_speed = paddle_speed
        self.ball_radius = ball_radius
        self.ball_speed_x_init = ball_speed_x
        self.ball_speed_y_init = ball_speed_y
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols
        self.brick_height = brick_height
        self.brick_area_top_offset = brick_area_top_offset
        self.initial_lives = initial_lives
        self.launch_angle_mode = launch_angle_mode
        self.launch_angle = launch_angle
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.frameskip = frameskip
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # 定义动作空间：0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        self.action_space = spaces.Discrete(4)
        
        # 定义观察空间：RGB 图像
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(screen_height, screen_width, 3),
            dtype=np.uint8
        )
        
        # 游戏区域定义
        self.play_area_top = 30  # 游戏区域顶部（留给分数显示）
        self.play_area_bottom = screen_height - 10  # 游戏区域底部
        self.paddle_y = self.play_area_bottom - 15  # 木板 Y 位置
        
        # 砖块区域
        self.brick_area_top = self.play_area_top + brick_area_top_offset
        self.brick_width = (screen_width - 10) // brick_cols
        
        # 游戏状态
        self.paddle_x = 0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.bricks = None  # 砖块状态矩阵
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.ball_launched = False  # 小球是否已发射
        
        # 颜色定义
        self.COLOR_BG = (0, 0, 0)  # 背景色
        self.COLOR_PADDLE = (200, 72, 72)  # 木板颜色
        self.COLOR_BALL = (236, 236, 236)  # 小球颜色
        self.COLOR_TEXT = (236, 236, 236)  # 文字颜色
        # 砖块颜色（不同行不同颜色）
        self.BRICK_COLORS = [
            (200, 72, 72),    # 红色
            (198, 108, 58),   # 橙色
            (180, 122, 48),   # 黄色
            (162, 162, 42),   # 绿色
            (72, 160, 72),    # 青色
            (66, 72, 200),    # 蓝色
        ]

        self.bounce=0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置游戏状态
        self.lives = self.initial_lives
        self.score = 0
        self.steps = 0
        self.bounce = 0
        self.paddle_x = self.screen_width // 2 - self.paddle_width // 2
        
        # 初始化砖块（全部存在）
        self.bricks = np.ones((self.brick_rows, self.brick_cols), dtype=bool)
        
        # 重置小球（放在木板上方，未发射）
        self._reset_ball()
        
        # 渲染初始帧
        observation = self._render_frame()
        info = self._get_info()
        
        return observation, info
    
    def _reset_ball(self):
        """重置小球位置（放在木板上方中央，等待发射）"""
        self.ball_launched = False
        self.ball_x = float(self.paddle_x + self.paddle_width // 2)
        self.ball_y = float(self.paddle_y - self.ball_radius - 1)
        self.ball_vx = 0.0
        self.ball_vy = 0.0
    
    def _launch_ball(self):
        """发射小球"""
        if not self.ball_launched:
            self.ball_launched = True
            
            # 根据发球模式确定角度
            if self.launch_angle_mode == "random":
                # 随机发射角度（向上），避免完全垂直
                # 范围：-0.5 到 -0.2 和 0.2 到 0.5（排除 -0.2 到 0.2 的接近垂直区域）
                if self.np_random.random() < 0.5:
                    angle = self.np_random.uniform(-0.5, -0.2)  # 左侧
                else:
                    angle = self.np_random.uniform(0.2, 0.5)   # 右侧
            elif self.launch_angle_mode == "fixed":
                # 固定角度（默认向右上 0.3 弧度，约 17 度）
                angle = 0.3
            elif self.launch_angle_mode == "fixed_angle":
                # 使用指定的固定角度
                angle = self.launch_angle
            else:
                # 默认随机
                if self.np_random.random() < 0.5:
                    angle = self.np_random.uniform(-0.5, -0.2)
                else:
                    angle = self.np_random.uniform(0.2, 0.5)
            
            self.ball_vx = self.ball_speed_x_init * np.sin(angle)
            self.ball_vy = -self.ball_speed_y_init * np.cos(angle)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        total_reward = 0.0
        terminated = False
        truncated = False
        
        # 根据 frameskip 执行多帧
        for _ in range(self.frameskip):
            # 执行单帧
            reward, term = self._step_single_frame(action)
            total_reward += reward
            
            if term:
                terminated = True
                break
        
        self.steps += 1
        
        # 检查是否超过最大步数
        if self.steps >= self.max_episode_steps:
            truncated = True
        
        # 渲染当前帧
        observation = self._render_frame()
        info = self._get_info()
        
        return observation, total_reward, terminated, truncated, info
    
    def _step_single_frame(self, action: int) -> Tuple[float, bool]:
        """执行单帧更新
        
        Returns:
            reward: 本帧获得的奖励
            terminated: 游戏是否结束
        """
        reward = 0.0
        terminated = False
        
        # 处理动作（修复惯性问题：只在有动作时移动）
        if action == 1:  # FIRE - 发射小球
            self._launch_ball()
        elif action == 2:  # RIGHT
            self.paddle_x = min(
                self.screen_width - self.paddle_width,
                self.paddle_x + self.paddle_speed
            )
            # 如果小球未发射，跟随木板移动
            if not self.ball_launched:
                self.ball_x = float(self.paddle_x + self.paddle_width // 2)
        elif action == 3:  # LEFT
            self.paddle_x = max(0, self.paddle_x - self.paddle_speed)
            # 如果小球未发射，跟随木板移动
            if not self.ball_launched:
                self.ball_x = float(self.paddle_x + self.paddle_width // 2)
        # action == 0 (NOOP) 时不移动，木板立即停止
        
        # 更新小球位置（仅当已发射时）
        if self.ball_launched:
            self.ball_x += self.ball_vx
            self.ball_y += self.ball_vy
            
            # 检测碰撞
            collision_reward = self._check_collisions()
            reward += collision_reward
            
            # 检查小球是否掉出屏幕（修复问题2）
            if self.ball_y > self.play_area_bottom + self.ball_radius:
                self.lives -= 1
                if self.lives <= 0:
                    terminated = True
                else:
                    # 重置小球
                    self._reset_ball()
        
        # 检查是否所有砖块都被打完
        if not self.bricks.any():
            terminated = True
        
        return reward, terminated
    
    def _check_collisions(self) -> float:
        """检测碰撞并更新小球速度
        
        Returns:
            reward: 碰撞获得的奖励
        """
        reward = 0.0
        
        # 1. 检测左右墙壁碰撞
        if self.ball_x - self.ball_radius <= 0:
            self.ball_x = float(self.ball_radius)
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + self.ball_radius >= self.screen_width:
            self.ball_x = float(self.screen_width - self.ball_radius)
            self.ball_vx = -abs(self.ball_vx)
        
        # 2. 检测顶部墙壁碰撞
        if self.ball_y - self.ball_radius <= self.play_area_top:
            self.ball_y = float(self.play_area_top + self.ball_radius)
            self.ball_vy = abs(self.ball_vy)
        
        # 3. 检测木板碰撞
        if (self.ball_vy > 0 and  # 小球向下运动
            self.paddle_y - self.ball_radius <= self.ball_y <= self.paddle_y + self.paddle_height + self.ball_radius and
            self.paddle_x - self.ball_radius <= self.ball_x <= self.paddle_x + self.paddle_width + self.ball_radius):
            self.bounce+=1
            # 反弹
            self.ball_y = float(self.paddle_y - self.ball_radius)
            self.ball_vy = -abs(self.ball_vy)
            
            # 根据击球位置调整水平速度（增加游戏策略性）
            hit_pos = (self.ball_x - self.paddle_x) / self.paddle_width  # 0到1
            self.ball_vx = self.ball_speed_x_init * (hit_pos - 0.5) * 2  # -1到1
        
        # 4. 检测砖块碰撞
        brick_reward = self._check_brick_collision()
        reward += brick_reward
        
        return reward
    
    def _check_brick_collision(self) -> float:
        """检测砖块碰撞
        
        Returns:
            reward: 打砖块获得的奖励
        """
        reward = 0.0
        
        # 计算小球在砖块网格中的位置
        ball_grid_x = int((self.ball_x - 5) / self.brick_width)
        ball_grid_y = int((self.ball_y - self.brick_area_top) / self.brick_height)
        
        # 检查周围的砖块
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                grid_y = ball_grid_y + dy
                grid_x = ball_grid_x + dx
                
                # 检查是否在有效范围内
                if (0 <= grid_y < self.brick_rows and
                    0 <= grid_x < self.brick_cols and
                    self.bricks[grid_y, grid_x]):
                    
                    # 计算砖块的边界
                    brick_left = 5 + grid_x * self.brick_width
                    brick_right = brick_left + self.brick_width - 2
                    brick_top = self.brick_area_top + grid_y * self.brick_height
                    brick_bottom = brick_top + self.brick_height - 2
                    
                    # 检测碰撞
                    if (brick_left - self.ball_radius <= self.ball_x <= brick_right + self.ball_radius and
                        brick_top - self.ball_radius <= self.ball_y <= brick_bottom + self.ball_radius):
                        
                        # 移除砖块
                        self.bricks[grid_y, grid_x] = False
                        
                        # 奖励（不同行不同分数）
                        # points = (self.brick_rows - grid_y) * 1
                        points=1
                        reward += points
                        self.score += points
                        
                        # 反弹（根据速度方向判断碰撞类型）
                        # 使用速度方向而非位置偏移来判断，更准确
                        # if abs(self.ball_vx) > abs(self.ball_vy):
                        # if self.ball_vx>0:
                        #     position_base=brick_left
                        # else:
                        position_base=brick_left if self.ball_vx>0 else brick_right
                        if brick_top<=(self.ball_y-(self.ball_x-position_base)/self.ball_vx*self.ball_vy)<=brick_bottom:
                            ## this method is wrong(水平速度更大 -> 左右碰撞)!!!, you should compute whether the ball enters the brick from vertical line of the brick or horizental line
                            brick_center_x = (brick_left + brick_right) / 2
                            if self.ball_x > brick_center_x:
                                # 小球在砖块右侧
                                self.ball_x = float(brick_right + self.ball_radius)
                            else:
                                # 小球在砖块左侧
                                self.ball_x = float(brick_left - self.ball_radius)
                            self.ball_vx = -self.ball_vx
                        else:
                            # 垂直速度更大 -> 上下碰撞
                            if self.ball_vy < 0:
                                # 小球向上运动，碰到砖块底部，反弹向下
                                self.ball_y = float(brick_bottom + self.ball_radius)
                                self.ball_vy = abs(self.ball_vy)  # 变为正值（向下）
                            else:
                                # 小球向下运动，碰到砖块顶部，反弹向上
                                self.ball_y = float(brick_top - self.ball_radius)
                                self.ball_vy = -abs(self.ball_vy)  # 变为负值（向上）
                        
                        # 只处理一个砖块碰撞
                        return reward
        
        return reward
    
    def _render_frame(self) -> np.ndarray:
        """渲染当前帧"""
        # 创建画布
        canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        canvas[:] = self.COLOR_BG
        
        # 1. 绘制分数和生命值
        self._draw_text(canvas, f"SCORE: {self.score}", 5, 5)
        self._draw_text(canvas, f"LIVES: {self.lives}", 5, 15)
        
        # 2. 绘制砖块
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if self.bricks[row, col]:
                    x = 5 + col * self.brick_width
                    y = self.brick_area_top + row * self.brick_height
                    color = self.BRICK_COLORS[row % len(self.BRICK_COLORS)]
                    cv2.rectangle(
                        canvas,
                        (x, y),
                        (x + self.brick_width - 2, y + self.brick_height - 2),
                        color,
                        -1
                    )
        
        # 3. 绘制木板
        cv2.rectangle(
            canvas,
            (int(self.paddle_x), int(self.paddle_y)),
            (int(self.paddle_x + self.paddle_width), int(self.paddle_y + self.paddle_height)),
            self.COLOR_PADDLE,
            -1
        )
        
        # 4. 绘制小球
        cv2.circle(
            canvas,
            (int(self.ball_x), int(self.ball_y)),
            self.ball_radius,
            self.COLOR_BALL,
            -1
        )
        
        return canvas
    
    def _draw_text(self, canvas: np.ndarray, text: str, x: int, y: int):
        """在画布上绘制文字"""
        cv2.putText(
            canvas,
            text,
            (x, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            "lives": self.lives,
            "score": self.score,
            "paddle_x": self.paddle_x,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_launched": self.ball_launched,
            "bricks_remaining": int(self.bricks.sum()),
            "bounce": self.bounce,
        }
    
    def render(self):
        """渲染环境（用于兼容 gym 接口）"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def close(self):
        """关闭环境"""
        pass
    
    def get_action_meanings(self):
        """获取动作含义（兼容 ALE 接口）"""
        return ["NOOP", "Fire", "Right", "Left"]


# 便捷函数：创建环境
def make_breakout(
    paddle_width: int = 20,
    paddle_speed: int = 4,
    ball_speed: float = 2.0,
    initial_lives: int = 5,
    frameskip: int = 4,
    **kwargs
) -> BreakoutEnv:
    """创建 Breakout 环境的便捷函数
    
    Args:
        paddle_width: 木板宽度（像素）
        paddle_speed: 木板移动速度（像素/帧）
        ball_speed: 小球速度（像素/帧）
        initial_lives: 初始生命数
        frameskip: 帧跳过数
        **kwargs: 其他参数传递给 BreakoutEnv
    
    Returns:
        BreakoutEnv 实例
    """
    return BreakoutEnv(
        paddle_width=paddle_width,
        paddle_speed=paddle_speed,
        ball_speed_x=ball_speed,
        ball_speed_y=ball_speed,
        initial_lives=initial_lives,
        frameskip=frameskip,
        **kwargs
    )
