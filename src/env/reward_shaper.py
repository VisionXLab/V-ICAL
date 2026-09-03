"""游戏奖励塑形模块

MiniGrid: 替换稀疏奖励为四组分密集奖励（探索/终点/效率/死亡）
ALE/Atari: 保留原始密集奖励，追加死亡惩罚和 Seaquest 专用救人/浮出奖励
"""

from collections import deque
from typing import Any, Callable, Dict, Optional, Set, Tuple
import numpy as np


# 每个 MiniGrid 游戏的 BFS 策略
GAME_BFS_STRATEGY = {
    "MiniGrid-Empty-16x16-v0":              "simple",
    "MiniGrid-FourRooms-v0":                "doors_only",
    "MiniGrid-Dynamic-Obstacles-16x16-v0":  "simple",
    "MiniGrid-LavaGapS7-v0":               "simple",
    "MiniGrid-LavaCrossingS11N5-v0":        "simple",
    "MiniGrid-DoorKey-16x16-v0":           "key_door_goal",
    "MiniGrid-Unlock-v0":                   "key_door",
    "MiniGrid-UnlockPickup-v0":            "key_door_pickup",
    "MiniGrid-MultiRoom-N6-v0":            "doors_only",
    # Custom MiniGrid
    "CustomLavaCrossing-v0":               "simple",
    "CustomMultiRoom-v0":                  "doors_only",
    "CustomUnlockPickup-v0":               "key_door_pickup",
    "MiniGrid-KeyCorridorS6R3-v0":         "key_door_pickup",
}

# 方向 → 前进向量: 0=right, 1=down, 2=left, 3=up
DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


class MiniGridRewardShaper:
    """MiniGrid 奖励塑形器"""

    def __init__(self, game_env):
        """
        Args:
            game_env: GameEnv 实例（来自 env_wrapper.py）
        """
        self.game_env = game_env
        self.game_name = game_env.game_name

        # 权重（×100 放大，提高前端可读性）
        self.w_explore = 10
        self.w_goal = 100
        self.w_efficiency = 70
        self.w_death = -50

        # Episode 状态（reset 时初始化）
        self.explored_cells: Set[Tuple[int, int]] = set()
        self.total_passable: int = 0
        self.shortest_path: int = 0
        self.last_exploration_ratio: float = 0.0

    def reset(self):
        """Episode 开始时调用，计算初始状态"""
        uw = self.game_env.env.unwrapped

        self.total_passable = self._count_passable_cells()
        self.shortest_path = self._compute_shortest_path()

        # 初始化探索集合（加入初始可见格子）
        self.explored_cells = set()
        highlight_mask = self.game_env._highlight_mask
        if highlight_mask is not None:
            self._update_exploration(highlight_mask)
        # 设为 0：第一步会把初始视野也算作探索奖励
        self.last_exploration_ratio = 0.0

        print(f"[RewardShaper] Reset: game={self.game_name}, "
              f"passable={self.total_passable}, "
              f"shortest_path={self.shortest_path}, "
              f"initial_explored={len(self.explored_cells)}")

    def step(self, raw_reward: float, terminated: bool, truncated: bool,
             info: Dict[str, Any]) -> float:
        """每步调用，返回 shaped reward"""
        # 1. 更新探索
        highlight_mask = self.game_env._highlight_mask
        if highlight_mask is not None:
            self._update_exploration(highlight_mask)

        new_ratio = len(self.explored_cells) / max(self.total_passable, 1)
        explore_delta = self.w_explore * (new_ratio - self.last_exploration_ratio)
        self.last_exploration_ratio = new_ratio

        # 2. 判断是否到达目标
        goal_reached = terminated and raw_reward > 0

        # 3. 目标奖励
        goal_bonus = self.w_goal if goal_reached else 0.0

        # 4. 效率 combo
        efficiency_bonus = 0.0
        if goal_reached and self.shortest_path > 0:
            actual_steps = self.game_env.env.unwrapped.step_count
            excess_ratio = (actual_steps - self.shortest_path) / self.shortest_path
            efficiency_bonus = self.w_efficiency * max(0.0, 1.0 - excess_ratio)

        # 5. 死亡惩罚（terminated 但未到达目标 = 死亡）
        death_penalty = 0.0
        if terminated and not goal_reached:
            death_penalty = self.w_death

        shaped = explore_delta + goal_bonus + efficiency_bonus + death_penalty

        # 6. 写入 breakdown 供调试
        info['reward_breakdown'] = {
            'exploration': round(explore_delta, 4),
            'exploration_total': round(new_ratio * self.w_explore, 4),
            'goal': round(goal_bonus, 4),
            'efficiency': round(efficiency_bonus, 4),
            'death': round(death_penalty, 4),
            'shaped_total': round(shaped, 4),
            'original_reward': round(raw_reward, 4),
            'explored_cells': len(self.explored_cells),
            'total_passable': self.total_passable,
            'shortest_path': self.shortest_path,
        }

        return shaped

    # ─── 探索追踪 ───

    def _update_exploration(self, highlight_mask: np.ndarray):
        """从 fog-of-war 的 highlight_mask 缓存中更新已探索格子"""
        uw = self.game_env.env.unwrapped
        for x in range(uw.width):
            for y in range(uw.height):
                if highlight_mask[x, y]:
                    self.explored_cells.add((x, y))

    # ─── 可通行格子统计 ───

    def _count_passable_cells(self) -> int:
        """统计可通行格子总数（探索奖励的分母）"""
        uw = self.game_env.env.unwrapped
        count = 0
        for x in range(uw.width):
            for y in range(uw.height):
                cell = uw.grid.get(x, y)
                if cell is None:
                    count += 1
                elif cell.type in ("goal", "key", "door", "ball"):
                    count += 1
                # Wall, Lava: 不计入
        return max(count, 1)

    # ─── BFS 最短路径 ───

    def _compute_shortest_path(self) -> int:
        """根据游戏类型计算 BFS 最短路径"""
        strategy = GAME_BFS_STRATEGY.get(self.game_name, "simple")
        try:
            if strategy == "simple":
                return self._bfs_simple()
            elif strategy == "doors_only":
                return self._bfs_doors_only()
            elif strategy == "key_door_goal":
                return self._bfs_key_door_goal()
            elif strategy == "key_door":
                return self._bfs_key_door()
            elif strategy == "key_door_pickup":
                return self._bfs_key_door_pickup()
            else:
                return self._bfs_simple()
        except Exception as e:
            uw = self.game_env.env.unwrapped
            fallback = getattr(uw, 'max_steps', 100) // 2
            print(f"[RewardShaper] WARNING: BFS failed ({e}), fallback={fallback}")
            return fallback

    def _bfs(self, start_pos: Tuple[int, int], start_dir: int,
             goal_pos: Tuple[int, int],
             passable_fn: Callable[[int, int], bool]) -> int:
        """(x, y, dir) 三维 BFS，返回最少操作步数（含转向）

        Args:
            start_pos: (x, y) 起点
            start_dir: 起始方向 (0-3)
            goal_pos: (x, y) 终点（任意方向到达即可）
            passable_fn: (x, y) -> bool 判断格子是否可通行

        Returns:
            最少步数，不可达返回 float('inf')
        """
        sx, sy = start_pos
        gx, gy = goal_pos

        # 起点就是终点
        if (sx, sy) == (gx, gy):
            return 0

        queue = deque()
        queue.append((sx, sy, start_dir, 0))
        visited = set()
        visited.add((sx, sy, start_dir))

        while queue:
            x, y, d, dist = queue.popleft()

            # 邻居 1: turn left
            new_d = (d + 3) % 4
            if (x, y, new_d) not in visited:
                visited.add((x, y, new_d))
                queue.append((x, y, new_d, dist + 1))

            # 邻居 2: turn right
            new_d = (d + 1) % 4
            if (x, y, new_d) not in visited:
                visited.add((x, y, new_d))
                queue.append((x, y, new_d, dist + 1))

            # 邻居 3: forward
            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy
            if passable_fn(nx, ny) and (nx, ny, d) not in visited:
                if (nx, ny) == (gx, gy):
                    return dist + 1
                visited.add((nx, ny, d))
                queue.append((nx, ny, d, dist + 1))

        return float('inf')

    def _bfs_to_adjacent(self, start_pos: Tuple[int, int], start_dir: int,
                         target_pos: Tuple[int, int],
                         passable_fn: Callable[[int, int], bool]) -> Tuple[int, int]:
        """BFS 到 target 的相邻格（面对 target），返回 (步数, 最终方向)

        agent 不能走到 target 格上（key/door 等不可通行），
        所以需要走到 target 的邻格并面朝 target。

        Returns:
            (最少步数, 面朝 target 时的方向)，不可达返回 (inf, -1)
        """
        tx, ty = target_pos
        sx, sy = start_pos

        # 目标：站在 target 的邻格，面朝 target
        # 即状态 (tx-dx, ty-dy, dir) 其中 DIR_TO_VEC[dir] = (dx, dy)
        goal_states = set()
        for d, (dx, dy) in DIR_TO_VEC.items():
            adj_x, adj_y = tx - dx, ty - dy
            if passable_fn(adj_x, adj_y) or (adj_x, adj_y) == (sx, sy):
                goal_states.add((adj_x, adj_y, d))

        if not goal_states:
            return float('inf'), -1

        queue = deque()
        queue.append((sx, sy, start_dir, 0))
        visited = set()
        visited.add((sx, sy, start_dir))

        # 检查起点是否已是目标状态
        if (sx, sy, start_dir) in goal_states:
            return 0, start_dir

        while queue:
            x, y, d, dist = queue.popleft()

            # turn left
            new_d = (d + 3) % 4
            state = (x, y, new_d)
            if state not in visited:
                if state in goal_states:
                    return dist + 1, new_d
                visited.add(state)
                queue.append((x, y, new_d, dist + 1))

            # turn right
            new_d = (d + 1) % 4
            state = (x, y, new_d)
            if state not in visited:
                if state in goal_states:
                    return dist + 1, new_d
                visited.add(state)
                queue.append((x, y, new_d, dist + 1))

            # forward
            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy
            if passable_fn(nx, ny) and (nx, ny, d) not in visited:
                state = (nx, ny, d)
                if state in goal_states:
                    return dist + 1, d
                visited.add(state)
                queue.append((nx, ny, d, dist + 1))

        return float('inf'), -1

    # ─── 策略实现 ───

    def _make_passable_fn(self, allow_doors: bool = False,
                          allow_unlocked_doors: bool = False,
                          extra_passable: Optional[Set[Tuple[int, int]]] = None):
        """创建可通行判定函数"""
        uw = self.game_env.env.unwrapped
        width, height = uw.width, uw.height
        grid = uw.grid
        _extra = extra_passable or set()

        def fn(x: int, y: int) -> bool:
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
            if (x, y) in _extra:
                return True
            cell = grid.get(x, y)
            if cell is None:
                return True
            if cell.type == "goal":
                return True
            if cell.type == "door":
                if cell.is_open or allow_doors:
                    return True
                if allow_unlocked_doors and not cell.is_locked:
                    return True
            return False

        return fn

    def _find_objects(self) -> Dict[str, list]:
        """扫描 grid 找各种对象位置"""
        uw = self.game_env.env.unwrapped
        objects = {"goal": [], "key": [], "door": [], "ball": [], "box": []}
        for x in range(uw.width):
            for y in range(uw.height):
                cell = uw.grid.get(x, y)
                if cell is not None and cell.type in objects:
                    objects[cell.type].append({
                        "pos": (x, y),
                        "color": getattr(cell, "color", None),
                        "is_open": getattr(cell, "is_open", None),
                        "is_locked": getattr(cell, "is_locked", None),
                    })
        return objects

    def _bfs_simple(self) -> int:
        """简单导航：BFS(start → goal)"""
        uw = self.game_env.env.unwrapped
        objects = self._find_objects()

        if not objects["goal"]:
            return getattr(uw, 'max_steps', 100) // 2

        goal_pos = objects["goal"][0]["pos"]
        start_pos = tuple(uw.agent_pos)
        start_dir = uw.agent_dir

        passable_fn = self._make_passable_fn()
        result = self._bfs(start_pos, start_dir, goal_pos, passable_fn)

        if result == float('inf'):
            return getattr(uw, 'max_steps', 100) // 2
        return result

    def _bfs_doors_only(self) -> int:
        """FourRooms / MultiRoom：门视为可通行，每扇门 +1 步 toggle"""
        uw = self.game_env.env.unwrapped
        objects = self._find_objects()

        if not objects["goal"]:
            return getattr(uw, 'max_steps', 100) // 2

        goal_pos = objects["goal"][0]["pos"]
        start_pos = tuple(uw.agent_pos)
        start_dir = uw.agent_dir

        # 门视为可通行（allow_doors=True）
        passable_fn = self._make_passable_fn(allow_doors=True)
        base_dist = self._bfs(start_pos, start_dir, goal_pos, passable_fn)

        if base_dist == float('inf'):
            return getattr(uw, 'max_steps', 100) // 2

        # 统计路径上需要打开的关闭门数量（近似：所有关闭门 +1）
        closed_doors = sum(1 for d in objects["door"] if not d["is_open"])
        return base_dist + closed_doors

    def _bfs_key_door_goal(self) -> int:
        """DoorKey / KeyCorridor：key → door → goal 分段 BFS"""
        uw = self.game_env.env.unwrapped
        objects = self._find_objects()

        if not objects["key"] or not objects["door"] or not objects["goal"]:
            return getattr(uw, 'max_steps', 100) // 2

        # 找匹配颜色的 key-door 对
        key_info = objects["key"][0]
        door_info = None
        for d in objects["door"]:
            if d.get("is_locked") and d.get("color") == key_info.get("color"):
                door_info = d
                break
        if door_info is None:
            door_info = objects["door"][0]

        key_pos = key_info["pos"]
        door_pos = door_info["pos"]
        goal_pos = objects["goal"][0]["pos"]
        start_pos = tuple(uw.agent_pos)
        start_dir = uw.agent_dir

        # 允许穿过未锁的门（KeyCorridor 等有多扇未锁门）
        passable_fn = self._make_passable_fn(allow_unlocked_doors=True)

        # Phase 1: start → key 的邻格（面朝 key）
        d1, dir1 = self._bfs_to_adjacent(start_pos, start_dir, key_pos, passable_fn)
        # +1 步 pickup
        total = d1 + 1

        # Phase 2: key 邻格 → door 的邻格（面朝 door）
        # key 已拾取，key 所在格现在可通行
        passable_fn2 = self._make_passable_fn(extra_passable={key_pos}, allow_unlocked_doors=True)
        # 从面朝 key 的位置出发，先转身（方向未知，取最优）
        if dir1 >= 0:
            adj_x = key_pos[0] - DIR_TO_VEC[dir1][0]
            adj_y = key_pos[1] - DIR_TO_VEC[dir1][1]
            d2, dir2 = self._bfs_to_adjacent((adj_x, adj_y), dir1, door_pos, passable_fn2)
        else:
            d2, dir2 = float('inf'), -1
        # +1 步 toggle
        total += d2 + 1

        # Phase 3: door → goal（门已打开）
        passable_fn3 = self._make_passable_fn(extra_passable={key_pos, door_pos}, allow_unlocked_doors=True)
        if dir2 >= 0:
            adj_x = door_pos[0] - DIR_TO_VEC[dir2][0]
            adj_y = door_pos[1] - DIR_TO_VEC[dir2][1]
            d3 = self._bfs((adj_x, adj_y), dir2, goal_pos, passable_fn3)
        else:
            d3 = float('inf')
        total += d3

        # 路径上穿过的关闭未锁门，每扇 +1 步 toggle
        closed_unlocked = sum(1 for d in objects["door"]
                              if not d["is_open"] and not d.get("is_locked"))
        total += closed_unlocked

        if total >= float('inf'):
            return getattr(uw, 'max_steps', 100) // 2
        return total

    def _bfs_key_door(self) -> int:
        """Unlock：key → door（toggle 即完成）"""
        uw = self.game_env.env.unwrapped
        objects = self._find_objects()

        if not objects["key"] or not objects["door"]:
            return getattr(uw, 'max_steps', 100) // 2

        key_info = objects["key"][0]
        door_info = None
        for d in objects["door"]:
            if d.get("is_locked") and d.get("color") == key_info.get("color"):
                door_info = d
                break
        if door_info is None:
            door_info = objects["door"][0]

        key_pos = key_info["pos"]
        door_pos = door_info["pos"]
        start_pos = tuple(uw.agent_pos)
        start_dir = uw.agent_dir

        passable_fn = self._make_passable_fn()

        # Phase 1: start → key 邻格
        d1, dir1 = self._bfs_to_adjacent(start_pos, start_dir, key_pos, passable_fn)
        total = d1 + 1  # +1 pickup

        # Phase 2: key 邻格 → door 邻格
        passable_fn2 = self._make_passable_fn(extra_passable={key_pos})
        if dir1 >= 0:
            adj_x = key_pos[0] - DIR_TO_VEC[dir1][0]
            adj_y = key_pos[1] - DIR_TO_VEC[dir1][1]
            d2, dir2 = self._bfs_to_adjacent((adj_x, adj_y), dir1, door_pos, passable_fn2)
        else:
            d2 = float('inf')
        total += d2 + 1  # +1 toggle

        if total >= float('inf'):
            return getattr(uw, 'max_steps', 100) // 2
        return total

    def _bfs_key_door_pickup(self) -> int:
        """UnlockPickup：key → door → target object"""
        uw = self.game_env.env.unwrapped
        objects = self._find_objects()

        if not objects["key"] or not objects["door"]:
            return getattr(uw, 'max_steps', 100) // 2

        key_info = objects["key"][0]
        door_info = None
        for d in objects["door"]:
            if d.get("is_locked") and d.get("color") == key_info.get("color"):
                door_info = d
                break
        if door_info is None:
            door_info = objects["door"][0]

        # target = box (the object to pick up in the next room)
        target_list = objects.get("box", []) or objects.get("ball", [])
        if not target_list:
            # 如果没有明确 target，fallback 到 goal
            target_list = objects.get("goal", [])
        if not target_list:
            return getattr(uw, 'max_steps', 100) // 2

        key_pos = key_info["pos"]
        door_pos = door_info["pos"]
        target_pos = target_list[0]["pos"]
        start_pos = tuple(uw.agent_pos)
        start_dir = uw.agent_dir

        # 允许穿过未锁的门（KeyCorridor 等有多扇未锁门）
        passable_fn = self._make_passable_fn(allow_unlocked_doors=True)

        # Phase 1: start → key
        d1, dir1 = self._bfs_to_adjacent(start_pos, start_dir, key_pos, passable_fn)
        total = d1 + 1  # pickup

        # Phase 2: key → door
        passable_fn2 = self._make_passable_fn(extra_passable={key_pos}, allow_unlocked_doors=True)
        if dir1 >= 0:
            adj_x = key_pos[0] - DIR_TO_VEC[dir1][0]
            adj_y = key_pos[1] - DIR_TO_VEC[dir1][1]
            d2, dir2 = self._bfs_to_adjacent((adj_x, adj_y), dir1, door_pos, passable_fn2)
        else:
            d2, dir2 = float('inf'), -1
        total += d2 + 1  # toggle

        # Phase 3: door → target
        passable_fn3 = self._make_passable_fn(extra_passable={key_pos, door_pos}, allow_unlocked_doors=True)
        if dir2 >= 0:
            adj_x = door_pos[0] - DIR_TO_VEC[dir2][0]
            adj_y = door_pos[1] - DIR_TO_VEC[dir2][1]
            d3, dir3 = self._bfs_to_adjacent((adj_x, adj_y), dir2, target_pos, passable_fn3)
        else:
            d3 = float('inf')
        total += d3 + 1  # pickup target

        # 路径上穿过的关闭未锁门，每扇 +1 步 toggle
        closed_unlocked = sum(1 for d in objects["door"]
                              if not d["is_open"] and not d.get("is_locked"))
        total += closed_unlocked

        if total >= float('inf'):
            return getattr(uw, 'max_steps', 100) // 2
        return total


# ═══════════════════════════════════════════════════════════════════════════
# ALE/Atari 奖励塑形
# ═══════════════════════════════════════════════════════════════════════════

# 需要奖励塑形的 ALE 游戏
ATARI_SHAPED_GAMES = {
    'ALE/SpaceInvaders-v5',
    'ALE/Seaquest-v5',
    'ALE/Asterix-v5',
    'ALE/Frostbite-v5',
}

# Per-game 死亡惩罚（未列出的用默认 -100）
ATARI_DEATH_PENALTIES = {
    'ALE/Seaquest-v5': -10000,
    'ALE/Asterix-v5': -10000,
    'ALE/Frostbite-v5': -10000,
}

# Seaquest RAM 地址
_SEAQUEST_RAM = {
    'oxygen': 102,      # 0-64, 64=满, 水下递减, 水面回升
    'diver_count': 62,  # 0-6, 接触水手 +1, 浮出后归零
    'lives': 59,        # 生命数
}


class ATARIRewardShaper:
    """ALE 游戏奖励塑形器（保留原始奖励 + 追加组分）

    通用：死亡惩罚 -100
    Seaquest 专用：水手即时奖励 +30/人，浮出送人奖励 +50×人数
    """

    def __init__(self, game_env):
        self.game_env = game_env
        self.game_name = game_env.game_name
        self.w_death = ATARI_DEATH_PENALTIES.get(self.game_name, -100)

        # Seaquest 专用配置
        self._is_seaquest = 'Seaquest' in game_env.game_name
        self.w_diver_rescue = 30   # 每接触一个水手的即时奖励
        self.w_surface_delivery = 50  # 浮出时每个水手的奖励

        # Episode 状态
        self._last_diver_count = 0
        self._last_oxygen = 0

    def _get_ram(self):
        """获取 ALE RAM"""
        try:
            return self.game_env.env.unwrapped.ale.getRAM()
        except Exception:
            return None

    def _get_seaquest_state(self):
        """读取 Seaquest 专用 RAM 状态"""
        ram = self._get_ram()
        if ram is None:
            return 0, 0
        diver_count = int(ram[_SEAQUEST_RAM['diver_count']])
        oxygen = int(ram[_SEAQUEST_RAM['oxygen']])
        return diver_count, oxygen

    def reset(self):
        """Episode 开始时调用"""
        if self._is_seaquest:
            self._last_diver_count, self._last_oxygen = self._get_seaquest_state()

        print(f"[RewardShaper] Reset: game={self.game_name}, "
              f"type=ATARI, death_penalty={self.w_death}"
              + (f", initial_divers={self._last_diver_count}, "
                 f"initial_oxygen={self._last_oxygen}"
                 if self._is_seaquest else ""))

    def step(self, raw_reward: float, terminated: bool, truncated: bool,
             info: Dict[str, Any]) -> float:
        """每步调用，返回 shaped reward = raw + bonuses + penalty"""
        shaped = raw_reward
        diver_bonus = 0.0
        surface_bonus = 0.0
        death_penalty = 0.0

        # Seaquest 专用奖励
        if self._is_seaquest:
            diver_count, oxygen = self._get_seaquest_state()

            # 1. 水手即时奖励（diver_count 增加 → 接触了水手）
            if diver_count > self._last_diver_count:
                new_divers = diver_count - self._last_diver_count
                diver_bonus = self.w_diver_rescue * new_divers

            # 2. 浮出送人奖励（diver_count 减少 → 浮出交付了水手）
            if diver_count < self._last_diver_count and self._last_diver_count > 0:
                delivered = self._last_diver_count - diver_count
                surface_bonus = self.w_surface_delivery * delivered

            self._last_diver_count = diver_count
            self._last_oxygen = oxygen
            shaped += diver_bonus + surface_bonus

        # 死亡惩罚（terminated = 游戏结束，无论原因）
        if terminated:
            death_penalty = self.w_death
            shaped += death_penalty

        # 写入 breakdown 供调试
        breakdown = {
            'raw_reward': round(raw_reward, 4),
            'death': round(death_penalty, 4),
            'shaped_total': round(shaped, 4),
        }
        if self._is_seaquest:
            breakdown.update({
                'diver_bonus': round(diver_bonus, 4),
                'surface_bonus': round(surface_bonus, 4),
                'diver_count': self._last_diver_count,
                'oxygen': self._last_oxygen,
            })
        info['reward_breakdown'] = breakdown

        return shaped
