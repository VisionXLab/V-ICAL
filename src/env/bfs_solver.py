"""MiniGrid BFS 最优解求解器

从 tmp/gen_human_ops.py 提取的核心 BFS 函数，用于计算 MiniGrid 环境的最短路径动作序列。
"""

from collections import deque
from src.env.reward_shaper import DIR_TO_VEC, GAME_BFS_STRATEGY


# ── BFS 核心 ──

def bfs_with_path(start_pos, start_dir, goal_pos, passable_fn):
    """(x,y,dir) BFS，返回 (距离, 动作列表)。动作: 0=left, 1=right, 2=forward"""
    sx, sy = start_pos
    gx, gy = goal_pos
    if (sx, sy) == (gx, gy):
        return 0, []

    queue = deque()
    queue.append((sx, sy, start_dir, 0))
    visited = {(sx, sy, start_dir): None}

    while queue:
        x, y, d, dist = queue.popleft()

        neighbors = [
            ((x, y, (d + 3) % 4), 0),  # turn left
            ((x, y, (d + 1) % 4), 1),  # turn right
        ]
        dx, dy = DIR_TO_VEC[d]
        nx, ny = x + dx, y + dy
        if passable_fn(nx, ny):
            neighbors.append(((nx, ny, d), 2))

        for state, action in neighbors:
            if state not in visited:
                visited[state] = ((x, y, d), action)
                if state[0] == gx and state[1] == gy and action == 2:
                    path = [action]
                    cur = (x, y, d)
                    while visited[cur] is not None:
                        parent, act = visited[cur]
                        path.append(act)
                        cur = parent
                    path.reverse()
                    return dist + 1, path
                queue.append((*state, dist + 1))

    return float('inf'), []


def bfs_to_adjacent_with_path(start_pos, start_dir, target_pos, passable_fn):
    """BFS到target邻格（面对target），返回 (距离, 动作列表, 最终方向)"""
    tx, ty = target_pos
    sx, sy = start_pos

    goal_states = set()
    for d, (dx, dy) in DIR_TO_VEC.items():
        adj_x, adj_y = tx - dx, ty - dy
        if passable_fn(adj_x, adj_y) or (adj_x, adj_y) == (sx, sy):
            goal_states.add((adj_x, adj_y, d))

    if not goal_states:
        return float('inf'), [], -1

    if (sx, sy, start_dir) in goal_states:
        return 0, [], start_dir

    queue = deque()
    queue.append((sx, sy, start_dir, 0))
    visited = {(sx, sy, start_dir): None}

    while queue:
        x, y, d, dist = queue.popleft()

        neighbors = [
            ((x, y, (d + 3) % 4), 0),
            ((x, y, (d + 1) % 4), 1),
        ]
        dx, dy = DIR_TO_VEC[d]
        nx, ny = x + dx, y + dy
        if passable_fn(nx, ny):
            neighbors.append(((nx, ny, d), 2))

        for state, action in neighbors:
            if state not in visited:
                visited[state] = ((x, y, d), action)
                if state in goal_states:
                    path = [action]
                    cur = (x, y, d)
                    while visited[cur] is not None:
                        parent, act = visited[cur]
                        path.append(act)
                        cur = parent
                    path.reverse()
                    return dist + 1, path, state[2]
                queue.append((*state, dist + 1))

    return float('inf'), [], -1


# ── 辅助函数 ──

def make_passable_fn(uw, allow_doors=False, allow_unlocked_doors=False, extra_passable=None):
    width, height = uw.width, uw.height
    grid = uw.grid
    _extra = extra_passable or set()

    def fn(x, y):
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


def find_objects(uw):
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


# ── 策略调度 ──

def get_shortest_path_actions(env):
    """获取当前环境的 BFS 最短路径动作序列。

    Parameters
    ----------
    env : GameEnv
        已 reset 的 GameEnv 实例

    Returns
    -------
    list[int]
        动作序列（0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle）
    """
    uw = env.env.unwrapped
    game_name = env.game_name
    strategy = GAME_BFS_STRATEGY.get(game_name, "simple")
    objects = find_objects(uw)
    start_pos = tuple(uw.agent_pos)
    start_dir = uw.agent_dir

    if strategy == "simple":
        if not objects["goal"]:
            return []
        goal_pos = objects["goal"][0]["pos"]
        pfn = make_passable_fn(uw)
        _, actions = bfs_with_path(start_pos, start_dir, goal_pos, pfn)
        return actions

    elif strategy == "doors_only":
        if not objects["goal"]:
            return []
        goal_pos = objects["goal"][0]["pos"]
        pfn = make_passable_fn(uw, allow_doors=True)
        _, raw_actions = bfs_with_path(start_pos, start_dir, goal_pos, pfn)
        return raw_actions

    elif strategy in ("key_door_goal", "key_door", "key_door_pickup"):
        if not objects["key"] or not objects["door"]:
            return []

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
        actions = []

        # Phase 1: start → key (adjacent, facing key)
        pfn = make_passable_fn(uw, allow_unlocked_doors=True)
        _, p1, dir1 = bfs_to_adjacent_with_path(start_pos, start_dir, key_pos, pfn)
        actions.extend(p1)
        actions.append(3)  # pickup

        if dir1 < 0:
            return []

        # Phase 2: key → door (adjacent, facing door)
        adj_x = key_pos[0] - DIR_TO_VEC[dir1][0]
        adj_y = key_pos[1] - DIR_TO_VEC[dir1][1]
        pfn2 = make_passable_fn(uw, extra_passable={key_pos}, allow_unlocked_doors=True)
        _, p2, dir2 = bfs_to_adjacent_with_path((adj_x, adj_y), dir1, door_pos, pfn2)
        actions.extend(p2)
        actions.append(5)  # toggle (open door)

        if strategy == "key_door":
            return actions

        # key_door_pickup: drop key, then go to target
        if strategy == "key_door_pickup":
            actions.extend([1, 1, 4, 1, 1])  # turn 180, drop, turn 180

        # Phase 3: door → goal/target
        if strategy == "key_door_goal":
            if not objects["goal"]:
                return []
            target_pos = objects["goal"][0]["pos"]
        else:  # key_door_pickup
            target_list = objects.get("box", []) or objects.get("ball", [])
            if not target_list:
                target_list = objects.get("goal", [])
            if not target_list:
                return []
            target_pos = target_list[0]["pos"]

        if dir2 >= 0:
            adj_x = door_pos[0] - DIR_TO_VEC[dir2][0]
            adj_y = door_pos[1] - DIR_TO_VEC[dir2][1]
            pfn3 = make_passable_fn(uw, extra_passable={key_pos, door_pos}, allow_unlocked_doors=True)

            if strategy == "key_door_pickup":
                _, p3, _ = bfs_to_adjacent_with_path((adj_x, adj_y), dir2, target_pos, pfn3)
                actions.extend(p3)
                actions.append(3)  # pickup target
            else:
                _, p3 = bfs_with_path((adj_x, adj_y), dir2, target_pos, pfn3)
                actions.extend(p3)
        return actions

    return []


def compute_and_execute_optimal(game_name, config, seed):
    """计算 BFS 最优解并执行，返回帧、动作、奖励等数据。

    Parameters
    ----------
    game_name : str
        游戏 ID
    config : dict
        游戏配置（含自定义参数）
    seed : int
        随机种子

    Returns
    -------
    dict or None
        成功时返回 {frames_jpeg, action_history, reward_history, action_info,
                     episode_reward, ending_type, total_steps}；
        BFS 无解时返回 None。
    """
    from env_wrapper import create_env, PREDEFINED_ACTION_MEANINGS
    from src.utils.helpers import encode_frame_jpeg

    # 创建环境并 reset
    cfg = dict(config)
    cfg["partial_obs"] = cfg.get("partial_obs", True)
    env = create_env(game_name, cfg)
    state0, _ = env.reset(seed=seed)

    # 计算 BFS 最优路径
    actions_plan = get_shortest_path_actions(env)
    env.close()

    if not actions_plan:
        return None

    # 重新创建环境执行并收集帧
    env = create_env(game_name, cfg)
    state0, _ = env.reset(seed=seed)
    uw = env.env.unwrapped

    frames_jpeg = [encode_frame_jpeg(state0.image)]
    action_history = []
    reward_history = []
    episode_reward = 0.0

    # 动作名映射
    meanings = PREDEFINED_ACTION_MEANINGS.get(game_name, [])
    action_info = {i: name for i, name in enumerate(meanings)}

    # 执行动作（forward 前自动 toggle 关闭的门）
    s = None
    for action in actions_plan:
        if action == 2:  # forward
            dx, dy = DIR_TO_VEC[uw.agent_dir]
            nx, ny = uw.agent_pos[0] + dx, uw.agent_pos[1] + dy
            cell = uw.grid.get(nx, ny)
            if cell is not None and cell.type == "door" and not cell.is_open:
                s = env.step(5)
                frames_jpeg.append(encode_frame_jpeg(s.image))
                action_history.append(5)
                reward_history.append(s.reward)
                episode_reward += s.reward
                if s.terminated or s.truncated:
                    break

        s = env.step(action)
        frames_jpeg.append(encode_frame_jpeg(s.image))
        action_history.append(action)
        reward_history.append(s.reward)
        episode_reward += s.reward

        if s.terminated or s.truncated:
            break

    if s is None:
        env.close()
        return None

    won = s.terminated and s.reward > 0
    ending_type = "victory" if won else ("time_out" if s.truncated else "game_over")
    pass_type = bool(won)
    env.close()

    return {
        "frames_jpeg": frames_jpeg,
        "action_history": action_history,
        "reward_history": reward_history,
        "action_info": action_info,
        "episode_reward": episode_reward,
        "ending_type": ending_type,
        "pass_type": pass_type,
        "total_steps": len(action_history),
    }
