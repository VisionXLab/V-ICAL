"""游戏 score 计算配置。

`reward_shaper.py` 与 `score_config.py` 的关系：
- reward_shaper 给 AI 看的训练信号——其中 Seaquest 的 +30/+50（diver/surface）和
  ALE 死亡惩罚 -10000 等数字是**人工校准过的"真实分"**，不是 ROM 原生 reward。
- score 嗅取 **`shaped_total - death`**（剔除死亡惩罚后的 shaper 输出），
  即 AI 表现的真实分。**不再使用** ROM 原生 raw_reward（时机/数值不可靠）。

两类规则：
- `REWARD_AS_SCORE_GAMES`：episode_reward 即 score，**保留原始量纲不归一**（这批
  游戏 reward 本身就是原始分，没有 shaping）。
- `SCORE_RULES`：9 个自定义游戏，**归一化到 [0, 100]**。每条规则用 `type` 字段
  分发到具体公式：
    step_ratio          —— Tennis：存活步数 / max_steps × 100
    ratio_capped        —— Asterix / Berzerk / Seaquest 等：
                            min(reward_or_raw / max, 1) × 100，死亡 cap 50
    event_count_capped  —— ChopperCommand / Frostbite 等：
                            min(event × per, cap)，可选死亡 cap
    minigrid_efficiency —— LavaCrossing：胜利时 shortest_path / step × 100，否则 0
    minigrid_doors_eff  —— MultiRoom：opened/total × 50 + 效率 × 50
    milestone_set       —— UnlockPickup：4 个有序里程碑累加（最高 100）

调用：`compute_score(game_id, episode_reward, *, info=None, step_count=None, max_steps=None)`。
其中 `info` 期望含 `ScoreState.to_info_dict()` 注入的字段（详见 game_session.py）。

`pass` 与 `score` 完全独立——可能 pass=True 但 score 很低（撑过 max_steps 没拿分），
也可能 score 高但 pass=False（活下来但事件没触发）。
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Reward 直传（保留原值，不归一化到 100）
# ─────────────────────────────────────────────────────────────────────────────
REWARD_AS_SCORE_GAMES: set[str] = {
    # Classic control
    "Acrobot-v1",
    "MountainCar-v0",
    "CliffWalking-v1",
    # Toy / third-party
    "aFlappyBird-v1",
    "CustomBreakout",
    "highway-v0",
    "tetris_gymnasium/Tetris",
    # ALE（reward 已是原始 Atari 分，无 shaping）
    "ALE/Alien-v5",
    "ALE/Assault-v5",
    "ALE/Freeway-v5",
    "ALE/RoadRunner-v5",
    # Procgen
    "procgen_:procgen-bigfish-v0",
    "procgen_:procgen-bossfight-v0",
    "procgen_:procgen-caveflyer-v0",
    "procgen_:procgen-dodgeball-v0",
    "procgen_:procgen-heist-v0",
    "procgen_:procgen-leaper-v0",
    "procgen_:procgen-ninja-v0",
}

# ─────────────────────────────────────────────────────────────────────────────
# 归一化到 [0, 100] 的 9 个游戏
# - source="raw"  → 用 info["raw_episode_reward"]（语义：shaper 输出扣死亡惩罚；
#                   Seaquest 含 diver/surface bonus，其它 ATARI 游戏只是 raw 不含 death）
# - source="env"  → 用入参 episode_reward（这批游戏没 shaping，reward 就是原始分）
# - death_cap     → ending=="gameover" 时 score = min(score, death_cap)
# ─────────────────────────────────────────────────────────────────────────────
SCORE_RULES: dict[str, dict] = {
    # ── BattleZone：score_hit（首次击中步数）× 0.7 + score_survive（存活步数比）× 0.3 ──
    # optimal_hit_step=25：在 25 步内首次击中得 score_hit=100；
    # 25<s≤50：score_hit = 75 + 25*(50-s)/25；未击中：score_hit=0
    "ALE/BattleZone-v5": {"type": "battlezone", "optimal_hit_step": 25},

    # ── Pong：按接球数归一（上限 3 球，每球 33.3 分）──
    # 不惩罚丢球；count_field 每接球 +1，try/except 内保证不减
    "ALE/Pong-v5": {
        "type": "event_count_capped",
        "count_field": "pong_balls_caught",
        "per_event": 100.0 / 3.0,
        "cap": 100,
    },

    # ── Tennis：按存活步数比例。撑到 max_steps（或胜利早达）→ 100 ──
    "ALE/Tennis-v5": {"type": "step_ratio"},

    # Taxi：三档里程碑得分（0 / 30 / 60-100），见 compute_score taxi_milestone 分支，max_score 由外部 cfg-max-score 传入
    "Taxi-v3": {"type": "taxi_milestone"},

    # ── ALE 累计分按比例归一，死亡 cap 50 ──
    "ALE/Asterix-v5":        {"type": "ratio_capped", "source": "raw", "max_score": 300,  "death_cap": 50},
    # ChopperCommand：原始 reward 仅用于 AI 反馈；评估按击杀事件统一计 10 分，忽略清波 bonus
    "ALE/ChopperCommand-v5": {
        "type": "event_count_capped",
        "count_field": "chopper_kill_total",
        "per_event": 10,
        "cap": 100,
        "death_cap": 50,
    },
    "ALE/Berzerk-v5":        {"type": "ratio_capped", "source": "env", "max_score": 600,  "death_cap": 50},
    "ALE/Seaquest-v5":       {"type": "ratio_capped", "source": "raw", "max_score": 500,  "death_cap": 50},
    # RiverRaid：桥 500→50 已在 ScoreState.step 重塑
    "ALE/Riverraid-v5":      {"type": "ratio_capped", "source": "raw", "max_score": 400,    "death_cap": 50},
    # Turmoil：吃光点 800→160 已在 ScoreState.step 重塑
    "ALE/Turmoil-v5":      {"type": "ratio_capped", "source": "raw", "max_score": 200,    "death_cap": 50},
    "ALE/DemonAttack-v5":  {"type": "ratio_capped", "source": "env", "max_score": 60,     "death_cap": 50},
    "ALE/MsPacman-v5":     {"type": "ratio_capped", "source": "env", "max_score": 800,    "death_cap": 50},
    "ALE/Trondead-v5":     {"type": "ratio_capped", "source": "env", "max_score": 30,     "death_cap": 50},

    # ── 死亡无惩罚：CartPole 倒杆不压分，Enduro 不会死亡（cap 100 等价无惩罚）──
    "CartPole-v1":         {"type": "ratio_capped", "source": "env", "max_score": 60,     "death_cap": 100},
    "ALE/Enduro-v5":       {"type": "ratio_capped", "source": "env", "max_score": 20,     "death_cap": 100},

    
    
    # ── Frostbite：按 ice_jump 事件数归一（10 分/个，cap 100，无死亡惩罚） ──
    # count_field 指向 ScoreState 的全局单调计数器，而非 event_counts（后者单条命会归零）
    # 不设 death_cap：死亡不压分，score 纯按累计跳冰数（与其它 ratio_capped 游戏不同）
    "ALE/Frostbite-v5": {
        "type": "event_count_capped",
        "count_field": "frostbite_ice_jump_total",
        "per_event": 10,
        "cap": 100,
    },

    # ── MiniGrid 自定义 ──
    "CustomLavaCrossing-v0": {"type": "minigrid_efficiency"},
    "CustomMultiRoom-v0":    {"type": "minigrid_doors_eff", "door_max": 50, "eff_max": 50},
    "CustomUnlockPickup-v0": {
        "type": "milestone_set",
        # 里程碑 → 分数。pick_key/open_door/drop_key_after_open 各 20，pick_box 40
        # 状态机由 ScoreState._update_unlock_pickup_milestones 维护：
        #   pick_key  ：第一次 carrying.type=='key'
        #   open_door ：任意 door.is_open=True（UnlockPickup 只有 1 扇锁门）
        #   drop_key_after_open：曾拿过 key + 门已开 + 当前 carrying != key
        #   pick_box  ：carrying.type=='box'（=游戏胜利）
        "milestones": {
            "pick_key": 20,
            "open_door": 20,
            "drop_key_after_open": 20,
            "pick_box": 40,
        },
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# ScoreState：跨 step 的 score 跟踪状态
# ═════════════════════════════════════════════════════════════════════════════

class ScoreState:
    """跟踪 score 计算需要的跨 step 状态。

    GameSession：reset() 调 self.reset(env)；step() 调 self.step(env, info, step_reward)；
    随后把 self.to_info_dict() 合并进 last_info，供 compute_score 读。
    """

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.raw_episode_reward: float = 0.0
        self.chopper_kills: int = 0
        # Frostbite 专用：score 用的"跳新白冰"全局累计计数器。
        # 与 game_session.event_counts["frostbite_ice_jump"]（单条命、life_lost 归零、
        # 给 pass 判定用）**完全独立**——这个只增不减、无视 life_lost / 关卡转场。
        self.frostbite_ice_jumps: int = 0
        # BattleZone：记录首次击中敌人的步数（None=未击中）
        self.battlezone_first_hit_step: Optional[int] = None
        self._battlezone_step: int = 0
        # 只记录 AI 阶段**新触发**的里程碑 / 开门；baseline 已达成的不计入
        self.milestones: dict[str, bool] = {}
        self.opened_doors: set[Tuple[int, int]] = set()
        self.total_doors: int = 0
        self.shortest_path: int = 0
        # UnlockPickup 状态机内部
        self._key_picked: bool = False
        self._door_opened: bool = False
        # Baseline：reset 当下已达成的状态（preload 阶段产物 / webui 起手为空）
        # 用于隔离"AI 阶段实际贡献"与"前置场景白送"
        self._baseline_opened: set[Tuple[int, int]] = set()
        self._baseline_milestones: set[str] = set()

    def reset(self, env_wrapper: Any) -> None:
        """env reset 后调用：清零累计 + 从 reward_shaper / grid 取静态信息。

        eval_batch / eval_crossval 在 preload 之后**再调一次** reset，
        这样 preload 阶段已开门 / 已捡钥匙等不算进 AI 的 score。
        """
        self.raw_episode_reward = 0.0
        self.chopper_kills = 0
        self.frostbite_ice_jumps = 0
        self.battlezone_first_hit_step = None
        self._battlezone_step = 0
        self.milestones = {}
        self.opened_doors = set()
        self.total_doors = 0
        self.shortest_path = 0
        self._key_picked = False
        self._door_opened = False
        self._baseline_opened = set()
        self._baseline_milestones = set()

        # MiniGrid 类游戏：从 reward_shaper 读 BFS 最短路；扫 grid 统计门总数 + baseline
        if self.game_id not in (
            "CustomLavaCrossing-v0",
            "CustomMultiRoom-v0",
            "CustomUnlockPickup-v0",
        ):
            return

        rs = getattr(env_wrapper, "_reward_shaper", None)
        sp = getattr(rs, "shortest_path", None)
        if isinstance(sp, (int, float)) and sp > 0 and sp != float("inf"):
            self.shortest_path = int(sp)

        try:
            uw = env_wrapper.env.unwrapped
            if not hasattr(uw, "grid"):
                return
            # 统计总门数 + baseline 已开门
            for x in range(uw.width):
                for y in range(uw.height):
                    cell = uw.grid.get(x, y)
                    if cell is not None and cell.type == "door":
                        self.total_doors += 1
                        if getattr(cell, "is_open", False):
                            self._baseline_opened.add((x, y))
            # UnlockPickup baseline 里程碑
            if self.game_id == "CustomUnlockPickup-v0":
                carrying = getattr(uw, "carrying", None)
                ctype = getattr(carrying, "type", None) if carrying is not None else None
                if ctype == "key":
                    self._key_picked = True
                    self._baseline_milestones.add("pick_key")
                if self._baseline_opened:
                    self._door_opened = True
                    self._baseline_milestones.add("open_door")
                if self._key_picked and self._door_opened and ctype != "key":
                    self._baseline_milestones.add("drop_key_after_open")
                if ctype == "box":
                    self._baseline_milestones.add("pick_box")
        except Exception:
            pass

    def step(self, env_wrapper: Any, info: dict, step_reward: float) -> None:
        """每步调用：累加 raw_episode_reward + 更新 MiniGrid 状态。

        注：字段名沿用 `raw_episode_reward` 是历史名字，**语义**是「AI 实际收到的
        shaper 输出，剔除死亡惩罚」——也就是 reward_shaper 里被人工校准过的"真实分"
        （Seaquest 20 杀人 / 30 救人 / +50 浮出），不是 ROM 原生 env.step() 输出
        （ROM 原生奖励时机不可靠，已不再使用）。
        """
        breakdown = info.get("reward_breakdown") or {}
        if "shaped_total" in breakdown:
            # ATARI shaper（含 Seaquest 专用 diver/surface bonus）→ 扣死亡惩罚
            self.raw_episode_reward += (
                float(breakdown["shaped_total"]) - float(breakdown.get("death", 0.0))
            )
        elif "original_reward" in breakdown:
            # MiniGrid shaper 写入（MiniGrid score 不读此字段，保留兼容性）
            self.raw_episode_reward += float(breakdown["original_reward"])
        else:
            # 无 shaper：step_reward 即真实分
            raw = float(step_reward)
            if self.game_id == "ALE/Riverraid-v5" and raw >= 500:
                # 桥给恰好 500 分；同 step 摧毁其他目标的得分保留原值
                raw = 50.0 + (raw - 500.0)
            elif self.game_id == "ALE/Turmoil-v5" and raw >= 800:
                # 吃光点给恰好 800 分；同 step 其他得分保留原值
                raw = 160.0 + (raw - 800.0)
            self.raw_episode_reward += raw

        if self.game_id == "ALE/ChopperCommand-v5":
            self.chopper_kills += int(info.get("chopper_kill_count", 0))

        # Frostbite：跳一块**新白冰** ROM 给 +10；跳**蓝冰（已点亮）给 0**，
        # 所以用 raw==10 检测天然不会重复计数已点亮的冰块。
        # 与 event_counts 的单条命计数不同：这里只增不减，life_lost / 关卡转场
        # （整行冰块变蓝后重置为白）都不清零——后者正是前端 score 先涨后跌的根因。
        if self.game_id == "ALE/Frostbite-v5":
            # raw_reward 不含死亡惩罚（死亡步 shaped 会是 10-10000，用 raw 更稳）。
            # repeat>1 时一次 do_action 的 raw 已逐帧累加（如跨 2 块新冰 = 20），
            # 故按 +10 的份数计，避免漏计；蓝冰 raw==0 自然算 0 份。
            # Frostbite 单帧得分恒为 0 或 10（每块白冰统一 10 分），无大额误算风险。
            ice_signal = breakdown.get("raw_reward", step_reward)
            if float(ice_signal) > 0:
                self.frostbite_ice_jumps += int(round(float(ice_signal) / 10.0))

        # BattleZone：追踪步数计数器 + 首次击中步数
        # 无 shaper，step_reward > 0 即击中/摧毁敌人
        if self.game_id == "ALE/BattleZone-v5":
            self._battlezone_step += 1
            if self.battlezone_first_hit_step is None and float(step_reward) > 0:
                self.battlezone_first_hit_step = self._battlezone_step

        # 2. MiniGrid 门 & 里程碑
        if self.game_id not in (
            "CustomLavaCrossing-v0",
            "CustomMultiRoom-v0",
            "CustomUnlockPickup-v0",
        ):
            return

        try:
            uw = env_wrapper.env.unwrapped
        except Exception:
            return
        if not hasattr(uw, "grid"):
            return

        # 扫描已打开的门（一旦 is_open 过即记入；剔除 baseline 已开的）
        for x in range(uw.width):
            for y in range(uw.height):
                cell = uw.grid.get(x, y)
                if cell is not None and cell.type == "door" and getattr(cell, "is_open", False):
                    if (x, y) not in self._baseline_opened:
                        self.opened_doors.add((x, y))

        if self.game_id == "CustomUnlockPickup-v0":
            self._update_unlock_pickup_milestones(uw)

    def _update_unlock_pickup_milestones(self, uw: Any) -> None:
        carrying = getattr(uw, "carrying", None)
        carrying_type = getattr(carrying, "type", None) if carrying is not None else None

        # 状态机翻 True 的瞬间记入 milestones；baseline 已达成的不算分
        # 1. pick_key
        if carrying_type == "key" and not self._key_picked:
            self._key_picked = True
            if "pick_key" not in self._baseline_milestones:
                self.milestones["pick_key"] = True

        # 2. open_door（UnlockPickup 只有一扇锁门，任何 is_open 即触发）
        if not self._door_opened:
            for x in range(uw.width):
                for y in range(uw.height):
                    cell = uw.grid.get(x, y)
                    if cell is not None and cell.type == "door" and getattr(cell, "is_open", False):
                        self._door_opened = True
                        if "open_door" not in self._baseline_milestones:
                            self.milestones["open_door"] = True
                        break
                if self._door_opened:
                    break

        # 3. drop_key_after_open：拿过 key + 门已开 + 当前没拿 key
        if (
            self._key_picked
            and self._door_opened
            and carrying_type != "key"
            and not self.milestones.get("drop_key_after_open")
            and "drop_key_after_open" not in self._baseline_milestones
        ):
            self.milestones["drop_key_after_open"] = True

        # 4. pick_box
        if carrying_type == "box" and "pick_box" not in self._baseline_milestones:
            self.milestones["pick_box"] = True

    def to_info_dict(self) -> dict:
        """返回要塞进 last_info 的字段；compute_score 从这里读。"""
        d: dict[str, Any] = {"raw_episode_reward": float(self.raw_episode_reward)}
        if self.game_id == "ALE/ChopperCommand-v5":
            d["chopper_kill_total"] = self.chopper_kills
        if self.game_id == "ALE/Frostbite-v5":
            d["frostbite_ice_jump_total"] = self.frostbite_ice_jumps
        if self.game_id == "ALE/BattleZone-v5":
            d["battlezone_first_hit_step"] = self.battlezone_first_hit_step
        if self.game_id in (
            "CustomLavaCrossing-v0",
            "CustomMultiRoom-v0",
            "CustomUnlockPickup-v0",
        ):
            d["shortest_path"] = self.shortest_path
        if self.game_id == "CustomMultiRoom-v0":
            d["opened_doors_count"] = len(self.opened_doors)
            d["total_doors"] = self.total_doors
        if self.game_id == "CustomUnlockPickup-v0":
            d["milestones"] = dict(self.milestones)
        return d


# ═════════════════════════════════════════════════════════════════════════════
# compute_score —— 总入口
# ═════════════════════════════════════════════════════════════════════════════

def compute_score(
    game_id: str,
    episode_reward: float,
    *,
    info: Optional[dict] = None,
    step_count: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_score: Optional[float] = None,
) -> Optional[float]:
    """计算游戏 score。

    - `REWARD_AS_SCORE_GAMES` 中：返回 episode_reward（不归一化）
    - `SCORE_RULES` 中：返回归一化 [0, 100]
    - 其他：返回 None
    - 缺失必要字段（如自定义规则需要的 info / step_count / max_steps）：返回 None
    """
    if game_id in REWARD_AS_SCORE_GAMES:
        try:
            return float(episode_reward)
        except (TypeError, ValueError):
            return None

    rule = SCORE_RULES.get(game_id)
    if rule is None:
        return None

    info = info or {}
    is_gameover = info.get("ending") == "gameover"
    rtype = rule["type"]

    if rtype == "step_ratio":
        if step_count is None or not max_steps:
            return None
        return min(float(step_count) / float(max_steps), 1.0) * 100.0

    if rtype == "ratio_capped":
        if rule.get("source") == "raw":
            val = info.get("raw_episode_reward")
            if val is None:
                return None
        else:
            val = episode_reward
        max_score_field = max_score if max_score is not None else None
        effective_max = max_score_field if max_score_field is not None else rule["max_score"]
        try:
            score = min(float(val) / float(effective_max), 1.0) * 100.0
        except (TypeError, ValueError):
            return None
        score = max(score, 0.0)
        if is_gameover and "death_cap" in rule:
            score = min(score, float(rule["death_cap"]))
        return score

    if rtype == "event_count_capped":
        count_field = rule.get("count_field")
        if count_field:
            # 优先读 ScoreState 注入的全局单调计数器（不受 life_lost 归零影响）
            n = int(info.get(count_field, 0))
        else:
            ec = info.get("event_counts") or {}
            n = int(ec.get(rule["event"], 0))
        score = float(min(n * rule["per_event"], rule["cap"]))
        if is_gameover and "death_cap" in rule:
            score = min(score, float(rule["death_cap"]))
        return score

    if rtype == "minigrid_efficiency":
        victory = bool(info.get("victory_signal"))
        if not victory or not step_count:
            return 0.0
        sp = info.get("shortest_path")
        if not sp or sp <= 0:
            return None
        return min(float(sp) / float(step_count), 1.0) * 100.0

    if rtype == "minigrid_doors_eff":
        opened = int(info.get("opened_doors_count", 0))
        total = int(info.get("total_doors") or 0)
        door_score = min(opened / total, 1.0) * rule["door_max"] if total > 0 else 0.0
        eff_score = 0.0
        if bool(info.get("victory_signal")) and step_count:
            sp = info.get("shortest_path")
            if sp and sp > 0:
                eff_score = min(float(sp) / float(step_count), 1.0) * rule["eff_max"]
        return min(door_score + eff_score, 100.0)

    if rtype == "milestone_set":
        ms = info.get("milestones") or {}
        total = 0
        for name, pts in rule["milestones"].items():
            if ms.get(name):
                total += pts
        return float(min(total, 100))

    if rtype == "battlezone":
        opt = float(rule.get("optimal_hit_step", 25))
        ms = float(max_steps) if max_steps else 50.0
        s = info.get("battlezone_first_hit_step")
        # score_hit
        if s is None:
            score_hit = 0.0
        elif float(s) <= opt:
            score_hit = 100.0
        else:
            score_hit = 75.0 + 25.0 * (ms - float(s)) / (ms - opt)
            score_hit = max(score_hit, 0.0)
        # score_survive
        p = float(step_count) if step_count is not None else 0.0
        score_survive = min(p / ms, 1.0) * 100.0
        return 0.7 * score_hit + 0.3 * score_survive

    if rtype == "taxi_milestone":
        # ① 未正确 pickup → 0
        if not info.get("taxi_correct_pickup"):
            return 0.0
        # ② 已 pickup 但未完成（或游戏结束非 victory）→ 30
        ending = info.get("ending", "")
        if ending != "victory":
            return 30.0
        # ③ 完成游戏：60 + 40*(max_steps-s)/(max_steps-x)
        # x = 最优步数 = 20 - max_score + 1（max_score 由外部 cfg-max-score 传入）
        if step_count is None or not max_steps or max_score is None:
            return 60.0
        try:
            x = 20.0 - float(max_score) + 1.0
            ms_f = float(max_steps)
            s_f = float(step_count)
            if ms_f <= x:
                return 100.0
            score = 60.0 + 40.0 * (ms_f - s_f) / (ms_f - x)
            return float(min(max(score, 60.0), 100.0))
        except (TypeError, ValueError, ZeroDivisionError):
            return 60.0

    return None


def get_score_game_ids() -> dict[str, list[str]]:
    """前端 / API 用：返回有 score 定义的两类游戏 id。"""
    return {
        "reward_as_score_games": sorted(REWARD_AS_SCORE_GAMES),
        "custom_score_games": sorted(SCORE_RULES.keys()),
    }
