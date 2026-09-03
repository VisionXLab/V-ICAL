"""游戏会话管理"""
import asyncio
import copy
import random
from typing import Any, Dict, List, Optional

from env_wrapper import create_env, do_action
from src.utils import encode_frame_jpeg, frame_to_base64, safe_json
from src.utils.score_config import ScoreState, compute_score

# 导入自定义 Breakout
CUSTOM_BREAKOUT_AVAILABLE = False
try:
    from new_gym.breakout_wrapper import create_breakout_env
    CUSTOM_BREAKOUT_AVAILABLE = True
except Exception:
    pass


# 每个游戏 pass=True 的判定规则：
#   "timeout"                            → 撑到 max_steps 即通过（生存类）
#   "victory"                            → 拿到胜利信号即通过（目标类）
#   {"event": "<name>", "min_count": N}  → episode 内累积事件 ≥ N 次即通过
#   未列出的游戏 → pass 永远为 False
# 注：死亡（life_lost）会让 "timeout"/"victory" 规则 pass=False；event 规则不受 life_lost 影响。
GAME_PASS_RULES = {
    "ALE/Asterix-v5": "timeout",
    "ALE/Seaquest-v5": "timeout",
    "ALE/ChopperCommand-v5": "timeout",
    "CustomLavaCrossing-v0": "victory",
    "CustomMultiRoom-v0": "victory",
    "CustomUnlockPickup-v0": "victory",
    "ALE/DemonAttack-v5": "timeout",
    "ALE/MsPacman-v5": {"type": "timeout", "min_score": 25},
    "ALE/BattleZone-v5": "timeout",
    "ALE/Riverraid-v5": "timeout",
    "ALE/Turmoil-v5": "timeout",
    "ALE/Trondead-v5": "timeout",
    "CartPole-v1": "timeout",
    "Taxi-v3": "victory",
    "ALE/Enduro-v5": "",
    "ALE/Pong-v5": "",
    # 事件计数型：必须真的触发指定事件 ≥ N 次
    # or_timeout=True：事件未达标时，撑到 timeout 也算 pass
    "ALE/Frostbite-v5": {"event": "frostbite_ice_jump",     "min_count": 2},
    "ALE/Tennis-v5":    {"event": "tennis_ball_returned",   "min_count": 1},
    "ALE/Berzerk-v5":   {"event": "berzerk_robot_killed",   "min_count": 3, "or_timeout": True},
}


class GameSession:
    """管理单个游戏会话"""

    def __init__(self, session_id: str, game_name: str, config: dict):
        self.session_id = session_id
        self.game_name = game_name
        self.config = config
        self.env: Optional[Any] = None  # GameEnv or BreakoutGameEnv
        self.frames: List[bytes] = []  # JPEG bytes
        self.step_count: int = 0
        self.episode_reward: float = 0.0
        self.action_info: Dict[int, str] = {}
        self.is_game_over: bool = False
        self.last_reward: float = 0.0
        self.last_info: dict = {}
        # 撤回支持
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self._snapshots: List[Any] = []  # env state snapshots for undo
        # 序列回放边界：前 _seq_action_count 个动作属于预加载序列，不属于 AI/人类操作
        self._seq_action_count: int = 0
        # 事件计数（per-game pass 规则使用）
        self.event_counts: Dict[str, int] = {}
        # Score 跟踪状态：累加 raw_episode_reward / MiniGrid 里程碑 / 门统计
        # （reward_shaper 改 reward 给 AI 看，score_state 不改 reward，只算评估指标）
        self.score_state: ScoreState = ScoreState(game_name)
        self.procgen=False
        if "procgen" in self.game_name:
            self.procgen=True

    async def initialize(self):
        """创建 env 并 reset（在线程中执行避免阻塞）"""
        def _create():
            cfg = dict(self.config)
            cfg["render_mode"] = "rgb_array"
            repeat = cfg.get("repeat", 1)
            action_repeat = cfg.pop("action_repeat", None)
            noop_fill = cfg.pop("noop_fill", False)
            seed=cfg.get("seed",None)
            if seed is None:
                seed = random.randint(0, 2**31 - 1)
            max_steps = cfg.get("max_steps", None)
            max_score = cfg.pop("max_score", None)
            end_on_life_loss = cfg.get("end_on_life_loss", False)  # 不 pop，传给 create_env
            skip_initial_steps = cfg.pop("skip_initial_steps", 0)
            skip_initial_action = cfg.pop("skip_initial_action", 0)

            if self.game_name == "CustomBreakout":
                if not CUSTOM_BREAKOUT_AVAILABLE:
                    raise RuntimeError("CustomBreakout 不可用，请检查 new_gym 目录")
                env = create_breakout_env(cfg)
            else:
                env = create_env(self.game_name, cfg)

            # reset
            if seed is not None:
                state, info = env.reset(seed=seed)
            else:
                state, info = env.reset()

            # Skip initial steps (execute action to skip loading/idle/serve screens)
            if self.procgen:
                # procgen: NOOP is action 4 (not 0); skip_initial_action does not apply
                for _ in range(skip_initial_steps):
                    state = env.step(4)
                    if state.terminated or state.truncated:
                        break
            else:
                for _ in range(skip_initial_steps):
                    state = env.step(skip_initial_action)
                    if state.terminated or state.truncated:
                        break
            # After skipping, state is a GameState from env.step() with .image
            if skip_initial_steps > 0:
                info = state.info

            # 跳帧完成后启用即时死亡检测（读取当前 RAM 作为基线）
            if hasattr(env, "arm_death_detection"):
                env.arm_death_detection()

            return env, state, info, repeat, action_repeat, noop_fill, seed, max_steps, max_score, end_on_life_loss, skip_initial_steps, skip_initial_action

        result = await asyncio.to_thread(_create)
        env, state, info, repeat, action_repeat, noop_fill, seed, max_steps, max_score, end_on_life_loss, skip_initial_steps, skip_initial_action = result
        self.env = env
        self.config["_repeat"] = repeat
        self.config["_action_repeat"] = action_repeat
        self.config["_noop_fill"] = noop_fill
        self.config["_seed"] = seed
        self.config["_max_steps"] = max_steps
        self.config["_max_score"] = max_score
        self.config["_end_on_life_loss"] = end_on_life_loss
        self.config["_skip_initial_steps"] = skip_initial_steps
        self.config["_skip_initial_action"] = skip_initial_action
        print(f"[GameSession] max_steps={max_steps}, end_on_life_loss={end_on_life_loss}, skip_initial_steps={skip_initial_steps}")
        self.action_info = env.get_action_info()

        # 保存初始帧
        jpeg = encode_frame_jpeg(state.image)
        self.frames.append(jpeg)
        self.step_count = 0
        self.episode_reward = 0.0
        self.is_game_over = False
        self.action_history = []
        self.reward_history = []
        self.event_counts = {}
        # ScoreState 必须在 env 完全 reset（含 reward_shaper.reset）后初始化，
        # 才能读到 shortest_path 与 grid 上的门
        self.score_state.reset(self.env)
        self._snapshots = [self._save_snapshot()]

        return frame_to_base64(jpeg)

    # -- env 快照 save/restore --
    def _get_unwrapped(self):
        """获取最底层的 gymnasium env（去掉所有 wrapper）"""
        gym_env = self.env.env  # GameEnv.env → gymnasium env (可能有 wrapper 链)
        if hasattr(gym_env, 'unwrapped'):
            return gym_env.unwrapped
        return gym_env

    # 需要保存的状态字段（按 env 类型）—— 排除渲染资源
    _STATE_FIELDS = {
        # Toy Text: 核心状态就是 s (整数)
        "TaxiEnv": ["s", "lastaction"],
        "CliffWalkingEnv": ["s", "lastaction"],
        "FrozenLakeEnv": ["s", "lastaction"],
        "BlackjackEnv": ["player", "dealer", "natural"],
        # Classic Control
        "CartPoleEnv": ["state", "steps_beyond_terminated"],
        "AcrobotEnv": ["state"],
        # MiniGrid
        "MiniGridEnv": ["agent_pos", "agent_dir", "grid", "carrying", "step_count", "mission"],
    }

    def _save_snapshot(self):
        """保存当前 env 状态快照（只保存游戏逻辑状态，不碰渲染资源）"""
        uw = self._get_unwrapped()

        # ALE: 专用 API
        if hasattr(uw, 'ale'):
            return ("ale", uw.ale.cloneSystemState())

        # CustomBreakout
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'paddle_x'):
            try:
                return ("deepcopy", copy.deepcopy(self.env.env))
            except Exception:
                pass

        # 按类名查找需要保存的字段
        class_name = type(uw).__name__
        for key, fields in self._STATE_FIELDS.items():
            if key in class_name:
                state = {}
                for f in fields:
                    if hasattr(uw, f):
                        val = getattr(uw, f)
                        # numpy array 需要 copy，其他直接存
                        if hasattr(val, 'copy'):
                            state[f] = val.copy()
                        else:
                            try:
                                state[f] = copy.deepcopy(val)
                            except Exception:
                                state[f] = val
                return ("fields", state)

        # 通用 fallback: 逐字段 deepcopy，跳过不可复制的
        state = {}
        skip_types = (type(None),)  # placeholder
        for k, v in uw.__dict__.items():
            # 跳过明显的渲染/资源字段
            if any(s in k.lower() for s in [
                'window', 'clock', 'screen', 'surf', 'render',
                'viewer', 'display', 'font', 'img', 'image',
                'texture', 'canvas', 'pygame', '_monitor',
            ]):
                continue
            # 跳过 spaces（不变的）
            if k in ('observation_space', 'action_space', 'spec', 'metadata',
                     'render_mode', 'reward_range', 'np_random'):
                continue
            try:
                if hasattr(v, 'copy'):
                    state[k] = v.copy()
                else:
                    state[k] = copy.deepcopy(v)
            except Exception:
                continue

        if state:
            return ("fields", state)

        return None

    def _restore_snapshot(self, snapshot):
        """恢复 env 到之前的快照"""
        if snapshot is None:
            return
        kind, data = snapshot

        if kind == "ale":
            uw = self._get_unwrapped()
            uw.ale.restoreSystemState(data)
        elif kind == "fields":
            uw = self._get_unwrapped()
            for k, v in data.items():
                # 恢复时必须 copy，否则后续 step 修改 list/array 会污染 snapshot
                if isinstance(v, list):
                    setattr(uw, k, v.copy())
                elif hasattr(v, 'copy'):
                    setattr(uw, k, v.copy())
                else:
                    setattr(uw, k, copy.deepcopy(v))

        elif kind == "deepcopy":
            if hasattr(self.env, 'env'):
                self.env.env = data

    async def reset(self):
        seed = self.config.get("_seed")
        skip_initial_steps = self.config.get("_skip_initial_steps", 0)
        skip_initial_action = self.config.get("_skip_initial_action", 0)

        def _reset():
            if seed is not None:
                state, info = self.env.reset(seed=seed)
            else:
                state, info = self.env.reset()

            # Skip initial steps on reset too
            if self.procgen:
                # procgen: NOOP is action 4 (not 0); skip_initial_action does not apply
                for _ in range(skip_initial_steps):
                    state = self.env.step(4)
                    if state.terminated or state.truncated:
                        break
            else:
                for _ in range(skip_initial_steps):
                    state = self.env.step(skip_initial_action)
                    if state.terminated or state.truncated:
                        break
            if skip_initial_steps > 0:
                info = state.info

            # 跳帧完成后启用即时死亡检测（如果环境支持）
            if hasattr(self.env, "arm_death_detection"):
                arm_fn = getattr(self.env, "arm_death_detection")
                if callable(arm_fn):
                    arm_fn()

            return state, info

        state, info = await asyncio.to_thread(_reset)
        jpeg = encode_frame_jpeg(state.image)
        # 清空所有历史，从头开始
        self.frames = [jpeg]
        self.step_count = 0
        self.episode_reward = 0.0
        self.is_game_over = False
        self.last_reward = 0.0
        self.last_info = info
        self.action_history = []
        self.reward_history = []
        self.event_counts = {}
        self.score_state.reset(self.env)
        self._snapshots = [self._save_snapshot()]
        self._seq_action_count = 0
        return frame_to_base64(jpeg)

    async def step(self, action: int):
        # 游戏已结束，拒绝执行
        if self.is_game_over:
            return frame_to_base64(self.frames[-1])

        repeat = self.config.get("_repeat", 1)
        action_repeat = self.config.get("_action_repeat")
        noop_fill = self.config.get("_noop_fill", False)

        def _step():
            if self.game_name == "CustomBreakout":
                from new_gym.breakout_wrapper import do_action as breakout_do_action
                return breakout_do_action(self.env, action, repeat=repeat)
            return do_action(self.env, action, repeat=repeat,
                             action_repeat=action_repeat, noop_fill=noop_fill)

        state = await asyncio.to_thread(_step)
        jpeg = encode_frame_jpeg(state.image)
        self.frames.append(jpeg)
        self.step_count += 1
        self.episode_reward += state.reward
        self.last_reward = state.reward
        self.is_game_over = state.terminated or state.truncated
        self.last_info = state.info
        self.last_info["terminated"] = state.terminated
        self.last_info["truncated"] = state.truncated
        try:
            self.last_info["ever_scored"] = state.ever_scored
        except AttributeError:
            pass

        # 累计事件计数（写入 self.last_info["event_counts"]）
        # 必须在 last_info 已经填好 life_lost / tennis_ball_returned 等字段之后调用
        self._update_event_counts(state)

        # Score 跟踪：累加 raw_episode_reward + MiniGrid 状态机
        # 必须在 reward_shaper 已写入 reward_breakdown 之后调用（env.step 内已完成）
        try:
            self.score_state.step(self.env, self.last_info, state.reward)
            self.last_info.update(self.score_state.to_info_dict())
        except Exception as e:
            print(f"[GameSession] WARN: score_state.step failed: {e}")

        # 实时 score 注入：把 compute_score 结果写到 info["score"]，供前端无脑显示
        # - REWARD_AS_SCORE_GAMES 18 个 → 等价 cumulative reward（与 xsb_wrapper 写法一致）
        # - SCORE_RULES 9 个 → 归一化 [0,100]
        # - 其它游戏 → compute_score 返回 None，不写 info["score"]，前端显示 "-"
        try:
            score_val = compute_score(
                self.game_name,
                float(self.episode_reward),
                info=self.last_info,
                step_count=self.step_count,
                max_steps=self.config.get("_max_steps"),
                max_score=self.config.get("_max_score"),
            )
            if score_val is not None:
                self.last_info["score"] = score_val
        except Exception as e:
            print(f"[GameSession] WARN: compute_score failed: {e}")

        # Check max_steps limit
        max_steps = self.config.get("_max_steps")
        timeout_hit = bool(max_steps and self.step_count >= max_steps)
        if timeout_hit:
            self.is_game_over = True
            self.last_info["truncated"] = True

        # end_on_life_loss 已下沉到 env_wrapper：life_lost 时直接 terminated=True
        # session 层只需检查 state.terminated（上面已处理）

        # 计算 ending / pass（仅在游戏结束时设置）
        if self.is_game_over and self.game_name in GAME_PASS_RULES:
            life_lost = bool(self.last_info.get("life_lost", False))
            victory = bool(self.last_info.get("victory_signal", False))

            # ending 优先级：死亡 > 胜利 > 超时 > 其它结束
            if state.terminated and life_lost:
                ending = "gameover"
            elif state.terminated and victory:
                ending = "victory"
            elif state.terminated:
                ending = "gameover"
            elif timeout_hit or state.truncated:
                ending = "timeout"
            else:
                ending = "gameover"

            rule = GAME_PASS_RULES.get(self.game_name)
            # event 规则：累计 ≥ min_count 即通过（不受 life_lost 影响，事件已发生就算数）
            if isinstance(rule, dict) and "event" in rule:
                passed = self.event_counts.get(rule["event"], 0) >= rule["min_count"]
                # or_timeout：事件未达标但撑到 timeout 也算过
                if not passed and rule.get("or_timeout") and ending == "timeout":
                    passed = True
            elif self.game_name == "ALE/Enduro-v5":
                passed = self.episode_reward > 0
            elif self.game_name == "ALE/Pong-v5":
                passed = bool(self.last_info.get("ever_scored"))
            elif life_lost:
                passed = False
            elif ending == "timeout" and (
                rule == "timeout"
                or (isinstance(rule, dict) and rule.get("type") == "timeout")
            ):
                passed = True
                if isinstance(rule, dict):
                    min_score = rule.get("min_score")
                    if min_score is not None:
                        passed = self.last_info.get("score", 0) >= min_score
            elif rule == "victory" and ending == "victory":
                passed = True
            else:
                passed = False

            self.last_info["ending"] = ending
            self.last_info["pass"] = passed

        self.action_history.append(action)
        self.reward_history.append(state.reward)
        self._snapshots.append(self._save_snapshot())
        return frame_to_base64(jpeg)

    def _update_event_counts(self, state):
        """每步检测特定游戏事件并累计。
        结果同步进 self.last_info["event_counts"]，供前端 / batch / pass 判定使用。

        通用路径：GAME_PASS_RULES 注册了 event 规则的游戏，wrapper 在 info[event_name]
        发 bool 信号（True 即"这一步发生了事件"），这里负责累加。

        特殊路径：Frostbite 的 ice_jump 基于 reward==10.0 检测且 life_lost 时归零，
        不走通用路径。
        """
        spec = GAME_PASS_RULES.get(self.game_name, {})
        if not (isinstance(spec, dict) and "event" in spec):
            return  # 非 event 规则的游戏跳过

        event = spec["event"]

        if self.game_name == "ALE/Frostbite-v5":
            # Frostbite ice_jump：单条命累计；life_lost 但还能复活时归零
            life_lost = bool(self.last_info.get("life_lost", False))
            if float(state.reward) == 10.0 and not state.terminated:
                self.event_counts[event] = self.event_counts.get(event, 0) + 1
            elif life_lost and not state.terminated:
                self.event_counts[event] = 0
        elif self.game_name == "ALE/Berzerk-v5":
            # Berzerk 只能靠射杀机器人得分，任意正分步即视为射死怪物
            if float(state.reward) > 0:
                self.event_counts[event] = self.event_counts.get(event, 0) + 1
        else:
            # 通用路径：wrapper 发 info[event] bool 信号，累加
            if self.last_info.get(event):
                self.event_counts[event] = self.event_counts.get(event, 0) + 1

        self.last_info["event_counts"] = dict(self.event_counts)

    async def undo(self):
        """撤回上一步：恢复 env 快照"""
        # 需要至少 2 个快照（当前 + 可回退的上一步）
        if len(self._snapshots) <= 1:
            return None

        def _restore():
            self._snapshots.pop()  # 弹出当前状态
            prev = self._snapshots[-1]  # 取上一个状态
            self._restore_snapshot(prev)

        await asyncio.to_thread(_restore)

        self.frames.pop()
        removed_reward = self.reward_history.pop()
        self.action_history.pop()
        self.step_count -= 1
        self.episode_reward -= removed_reward
        self.is_game_over = False
        self.last_reward = self.reward_history[-1] if self.reward_history else 0.0
        self.last_info = {}
        self.event_counts = {}  # undo 后清零（保守：用户重新触发事件来 pass）
        # score_state 同样保守清零：精确回滚里程碑/门状态机成本太高，重新触发即可
        self.score_state.reset(self.env)

        return frame_to_base64(self.frames[-1])

    def get_frame_base64(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.frames):
            return frame_to_base64(self.frames[index])
        return None

    def get_state_info(self) -> dict:
        ram_info = None
        if self.env is not None:
            ram_info = self.env.get_ram_info()
        return safe_json({
            "step": self.step_count,
            "reward": self.last_reward,
            "cumulative_reward": self.episode_reward,
            "is_game_over": self.is_game_over,
            "total_frames": len(self.frames),
            "action_info": self.action_info,
            "ram_info": ram_info,
            "info": self.last_info or {},
            "action_history": self.action_history,
        })

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
