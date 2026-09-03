"""
Batch Evaluation Tool — 从 ai_configs/ 批量读取配置并运行 AI 评测

用法:
    python vcl.py batch [options]

    --runs N          每个 config 跑 N 局 (default: 1)
    --workers N       最大并发数 (default: 3)
    --game GAME_ID_1,GAME_ID_2    按游戏过滤 (如 "ALE/Seaquest-v5")
    --config NAME_1_1,NAME_1_2/NAME_2_1,NAME_2_2     按 config 名过滤 (子串匹配)
    --output DIR      输出目录 (default: batch_results/)
    --set KEY=VALUE   临时覆盖 game_settings（可重复，如 --set cfg-max-steps=1）
    --resume RUN_ID   续传之前的批量运行 (如 "20260406_143000")
"""
import argparse
import asyncio
import base64
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 项目根目录（锚回仓库根 + 注入 sys.path）
from tools._paths import ROOT  # noqa: E402
from tools.runtime_init import ensure_pygame_surfarray  # noqa: E402

from src.core import GameSession
from src.ai import AIPlayer
from src.utils import DEFAULT_CONFIG, frame_to_base64
from src.video.frame_processor import (
    load_frames_from_source, get_op_dir, add_caption, add_ending_label, save_video,
)
from src.video.trajectory_visualizer import generate_trajectory_image

AI_CONFIGS_DIR = ROOT / "ai_configs"
ACTION_SEQ_DIR = ROOT / "action_sequences"


# ---------------------------------------------------------------------------
# 1. game_settings 转换
# ---------------------------------------------------------------------------

# game_id → tag 映射（复刻 webui.py GAMES 中的 tag）
CUSTOM_MINIGRID_IDS = {"CustomLavaCrossing-v0", "CustomMultiRoom-v0", "CustomUnlockPickup-v0"}

# 与前端 static/js/core/state.js GAME_DEFAULTS 对齐：skip-initial-action 没有 UI 元素，
# 不会写入 config 的 game_settings，这里 fallback 保证 batch_eval 与 webui 行为一致。
SKIP_INITIAL_ACTION_DEFAULTS = {
    "ALE/Tennis-v5": 1,  # 开局自动 Swing，让 AI 从已开球的状态开始
}


def _get_game_tag(game_id: str) -> str:
    """根据 game_id 判断 tag"""
    if game_id.startswith("ALE/"):
        return "ale"
    if game_id == "CustomBreakout":
        return "custom_breakout"
    if game_id.startswith("MiniGrid") or game_id in CUSTOM_MINIGRID_IDS:
        return "minigrid"
    # classic_control / toy_text 不需要特殊处理
    return "other"


def convert_game_settings(game_id: str, game_settings: dict) -> dict:
    """将前端 cfg-xxx 格式的 game_settings 转换为 GameSession config dict。

    严格复刻 static/js/game/control.js buildConfig() 的逻辑。
    """
    cfg = {}
    tag = _get_game_tag(game_id)

    # 通用参数
    seed_str = game_settings.get("cfg-seed", "")
    if seed_str != "":
        cfg["seed"] = int(seed_str)
    cfg["repeat"] = int(game_settings.get("cfg-repeat", "1")) or 1
    cfg["noop_fill"] = game_settings.get("cfg-noop-fill", False)
    cfg["pre_score"]=float(game_settings.get("cfg-pre-score","0"))
    if cfg["noop_fill"]:
        cfg["action_repeat"] = int(game_settings.get("cfg-action-repeat", "1")) or 1

    max_steps_str = game_settings.get("cfg-max-steps", "")
    if max_steps_str != "":
        cfg["max_steps"] = int(max_steps_str)

    max_score_str = game_settings.get("cfg-max-score", "")
    if max_score_str != "":
        cfg["max_score"] = float(max_score_str)

    # ALE 专属
    if tag == "ale":
        cfg["frameskip"] = int(game_settings.get("cfg-frameskip", "4")) or 4
        cfg["auto_respawn"] = game_settings.get("cfg-auto-respawn", True)
        lives_str = game_settings.get("cfg-initial-lives", "")
        if lives_str != "":
            cfg["initial_lives"] = int(lives_str)
        cfg["end_on_life_loss"] = game_settings.get("cfg-end-on-life-loss", False)
        skip_str = game_settings.get("cfg-skip-initial-steps", "")
        if skip_str != "" and int(skip_str) > 0:
            cfg["skip_initial_steps"] = int(skip_str)
        skip_action_str = game_settings.get("cfg-skip-initial-action", "")
        if skip_action_str != "" and int(skip_action_str) > 0:
            cfg["skip_initial_action"] = int(skip_action_str)
        elif game_id in SKIP_INITIAL_ACTION_DEFAULTS:
            cfg["skip_initial_action"] = SKIP_INITIAL_ACTION_DEFAULTS[game_id]
        cfg["action_set"] = game_settings.get("cfg-action-set", "simple")
        modeval=game_settings.get("cfg-mode","")
        if modeval!="":
            cfg["mode"]=int(modeval)
        diffval=game_settings.get("cfg-difficulty","")
        if diffval!="":
            cfg["difficulty"]=int(diffval)


        mode_val = game_settings.get("cfg-mode") or game_settings.get("cfg-game-mode", "")
        if mode_val != "":
            cfg["mode"] = int(mode_val)
        diff_val = game_settings.get("cfg-difficulty") or game_settings.get("cfg-game-difficulty", "")
        if diff_val != "":
            cfg["difficulty"] = int(diff_val)

        # initial_ram: 支持字符串格式 "114=5\n62=0" 或直接传 dict {114: 5}
        ram_raw = game_settings.get("cfg-initial-ram", "")
        if isinstance(ram_raw, dict) and ram_raw:
            cfg["initial_ram"] = {int(k): int(v) for k, v in ram_raw.items()}
        elif isinstance(ram_raw, str) and ram_raw.strip():
            ram_obj = {}
            for line in ram_raw.strip().split('\n'):
                line = line.strip()
                if '=' in line:
                    try:
                        addr, val = line.split('=', 1)
                        ram_obj[int(addr.strip())] = int(val.strip())
                    except ValueError:
                        pass
            if ram_obj:
                cfg["initial_ram"] = ram_obj

    # CustomBreakout 专属
    elif tag == "custom_breakout":
        cfg["paddle_width"] = int(game_settings.get("cfg-paddle-width", "20")) or 20
        cfg["ball_speed"] = float(game_settings.get("cfg-ball-speed", "2.0")) or 2.0
        cfg["brick_rows"] = int(game_settings.get("cfg-brick-rows", "6")) or 6
        cfg["brick_cols"] = int(game_settings.get("cfg-brick-cols", "10")) or 10
        cfg["brick_area_top_offset"] = int(game_settings.get("cfg-brick-offset", "10")) or 10
        cfg["launch_angle_mode"] = game_settings.get("cfg-launch-angle-mode", "random")
        cfg["frameskip"] = int(game_settings.get("cfg-cb-frameskip", "4")) or 4
        cfg["initial_lives"] = int(game_settings.get("cfg-cb-lives", "5")) or 5
        cfg["end_on_life_loss"] = game_settings.get("cfg-cb-end-on-life-loss", False)
        cb_skip_str = game_settings.get("cfg-cb-skip-initial-steps", "")
        if cb_skip_str != "" and int(cb_skip_str) > 0:
            cfg["skip_initial_steps"] = int(cb_skip_str)

    # MiniGrid 参数
    if tag == "minigrid":
        cfg["partial_obs"] = game_settings.get("cfg-partial-obs", True)

    # Custom MiniGrid 参数
    if game_id == "CustomLavaCrossing-v0":
        cfg["size"] = int(game_settings.get("cfg-lc-size", "11")) or 11
        cfg["num_crossings"] = int(game_settings.get("cfg-lc-crossings", "5")) or 5
    if game_id == "CustomMultiRoom-v0":
        cfg["grid_size"] = int(game_settings.get("cfg-mr-grid-size", "25")) or 25
        cfg["num_rooms"] = int(game_settings.get("cfg-mr-rooms", "6")) or 6
        cfg["max_room_size"] = int(game_settings.get("cfg-mr-room-size", "10")) or 10
    if game_id == "CustomUnlockPickup-v0":
        cfg["room_size"] = int(game_settings.get("cfg-up-room-size", "6")) or 6
        cfg["num_rows"] = int(game_settings.get("cfg-up-rows", "1")) or 1
        cfg["num_cols"] = int(game_settings.get("cfg-up-cols", "2")) or 2

    # FrozenLake
    if "FrozenLake" in game_id:
        cfg["is_slippery"] = game_settings.get("cfg-slippery", False)

    if "highway" in game_id:
        cfg["lanes_count"]=int(game_settings.get("cfg-lanes-count","4"))
        cfg["vehicles_density"]=float(game_settings.get("cfg-vehicles-density","1.4"))

    if "procgen" in game_id:
        procgen_skip=game_settings.get("cfg-procgen-skip-initial-steps",'')
        if procgen_skip!='' and int(procgen_skip)>0:
            cfg["skip_initial_steps"]=int(procgen_skip)

    return cfg


# ---------------------------------------------------------------------------
# 2. 扫描 ai_configs
# ---------------------------------------------------------------------------

def scan_configs(game_filter: str = None, config_filter: str = None) -> List[dict]:
    """扫描 ai_configs/ 目录，返回配置列表。"""
    if game_filter:
        game_list=game_filter.split(',')
    if config_filter:
        config_list=config_filter.split('/')
        for i,sublist in enumerate(config_list):
            config_list[i]=sublist.split(',')
    results = []
    if not AI_CONFIGS_DIR.exists():
        return results

    for game_dir in sorted(AI_CONFIGS_DIR.iterdir()):
        if not game_dir.is_dir() or game_dir.name.startswith("."):
            continue
        for cfg_dir in sorted(game_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            cfg_file = cfg_dir / "config.json"
            if not cfg_file.exists():
                continue
            try:
                cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            game_id = cfg_data.get("game_id", game_dir.name.replace("_", "/", 1))
            config_name = cfg_dir.name

            # 过滤
            if game_filter and not (game_id in game_list):
                continue
            if config_filter and len(config_list)>=game_list.index(game_id)+1 and not (config_name in config_list[game_list.index(game_id)]):
                continue

            results.append({
                "game_id": game_id,
                "config_name": config_name,
                "config_dir": cfg_dir,
                "config_data": cfg_data,
            })

    return results


# ---------------------------------------------------------------------------
# 3. Context 加载
# ---------------------------------------------------------------------------

def load_context_for_ai(ai: AIPlayer, config_data: dict, config_dir: Path, context: bool=True):
    """将 context 帧/视频加载到 AIPlayer。

    复刻前端 aiStart() 的 context 加载逻辑。
    """
    ai.clear_context()

    context_items = config_data.get("context_items", [])
    if not context_items or not context:
        return

    frame_source = ai.frame_source

    # 先加载 config 目录下保存的所有帧（用于 upload 类型的 context items）
    frames_dir = config_dir / "frames"
    all_saved_frames = []
    if frames_dir.exists():
        for f in sorted(frames_dir.glob("*.jpg")):
            all_saved_frames.append(base64.b64encode(f.read_bytes()).decode("ascii"))

    offset = 0
    for item in context_items:
        frame_count = item.get("frame_count", 0)

        if item.get("type") == "folder" and item.get("folder"):
            # 从 human_operation/ai_operation 加载
            folder_path = item["folder"]  # 如 "human/Seaquest-v5_crash"
            parts = folder_path.split("/", 1)
            if len(parts) == 2:
                source, name = parts
            else:
                source, name = "human", parts[0]

            op_dir = get_op_dir(source)
            frames_b64 = load_frames_from_source(op_dir, name, frame_source)

            # 加载对应视频文件（避免运行时重编码）
            video_name_map = {
                "original": "video_original.mp4",
                "subbed": "video.mp4",
                "subbed_nl": "video_nl.mp4",
            }
            video_file = op_dir / name / video_name_map.get(frame_source, "video.mp4")
            video_b64 = None
            if video_file.exists():
                video_b64 = base64.b64encode(video_file.read_bytes()).decode()

            if frames_b64:
                ai.add_context_group(frames_b64, video_b64=video_b64)
                print(f"  [Context] Loaded folder '{folder_path}': {len(frames_b64)} frames"
                      f"{', video' if video_b64 else ''}")
            else:
                # Fallback: folder 不存在时，用 ai_config 自己保存的 frames/ 按 context item 边界切分
                item_frames = all_saved_frames[offset:offset + frame_count]
                if item_frames:
                    ai.add_context_group(item_frames)
                    print(f"  [Context] Folder '{folder_path}' not found, fallback to ai_config saved segment: "
                          f"{len(item_frames)} frames")

        elif item.get("type") == "upload":
            # 从 ai_config 保存的帧中按边界切分
            item_frames = all_saved_frames[offset:offset + frame_count]
            if item_frames:
                ai.add_context_group(item_frames)
                print(f"  [Context] Loaded upload '{item.get('label', '?')}': {len(item_frames)} frames")

        offset += frame_count


# ---------------------------------------------------------------------------
# 4. Action Sequence 回放
# ---------------------------------------------------------------------------

async def replay_action_sequence(session: GameSession, action_sequence: dict):
    """复刻 websocket_handler 的 replay_sequence 逻辑。"""
    game_safe = action_sequence.get("game_safe", "")
    seq_name = action_sequence.get("name", "")
    if not game_safe or not seq_name:
        return

    seq_file = ACTION_SEQ_DIR / game_safe / seq_name / "sequence.json"
    if not seq_file.exists():
        print(f"  [Warning] Action sequence not found: {seq_file}")
        return

    seq_data = json.loads(seq_file.read_text(encoding="utf-8"))
    actions = seq_data.get("actions", [])
    if not actions:
        return

    print(f"  [Replay] Replaying {len(actions)} actions from sequence '{seq_name}'...")

    # 临时禁用 max_steps 和 end_on_life_loss
    saved_max_steps = session.config.get("_max_steps")
    session.config["_max_steps"] = None
    saved_end_on_life_loss = False
    if session.env and hasattr(session.env, "end_on_life_loss"):
        saved_end_on_life_loss = session.env.end_on_life_loss
        session.env.end_on_life_loss = False

    # 逐个执行动作
    for action in actions:
        if session.is_game_over:
            break
        await session.step(int(action))

    if "freeway" in session.game_name.lower() or "cliffwalking" in session.game_name.lower():
        session.env.env.state_reset()

    # 恢复 max_steps 和 end_on_life_loss
    session.config["_max_steps"] = saved_max_steps
    if session.env and hasattr(session.env, "end_on_life_loss"):
        session.env.end_on_life_loss = saved_end_on_life_loss

    # 记录序列边界，重置计数（序列步数不算 AI 操作）
    session._seq_action_count = len(session.action_history)
    session.step_count = 0
    session.episode_reward = 0.0
    session.event_counts = {}
    session.is_game_over = False
    # env_wrapper 层清零（_ever_scored / _pong_balls_caught / _taxi_correct_pickup /
    # _tennis_last_ball_y / _reward_shaper 等）；必须先于 score_state.reset()，
    # 因为 MiniGrid score_state 需要读 _reward_shaper.shortest_path（reset 后重算）
    if session.env is not None and not "custombreakout" in session.game_name.lower():
        session.env.reset_tracking_state()
    # ScoreState 重置：清零 battlezone_first_hit_step / frostbite_ice_jumps /
    # raw_episode_reward 等
    session.score_state.reset(session.env)
    for k in ("ending", "pass", "truncated"):
        session.last_info.pop(k, None)
    # 清除序列步骤的撤回历史
    session._snapshots = session._snapshots[-1:]

    print(f"  [Replay] Done. Game over: {session.is_game_over}")


# ---------------------------------------------------------------------------
# 5. Batch Run Loop（无 WebSocket）
# ---------------------------------------------------------------------------

async def batch_run_loop(ai: AIPlayer, session: GameSession) -> dict:
    """复刻 AIPlayer.run_loop() 但无 WebSocket 依赖。

    返回 {"status", "actions", "rewards", "rounds", "token_usage", "error"}
    """
    action_info = session.action_info
    cumulative_reward = 0.0
    step = 0
    messages_history = []

    actions_log = []
    rewards_log = []
    rounds_log = []  # conversation 记录
    token_usage_total = {"total_input": 0, "total_output": 0, "total_thoughts": 0, "per_round": []}

    def _record_usage(usage: dict, round_num: int):
        entry = {"round": round_num}
        for k in ("input", "output", "thoughts", "total",
                   "input_text_tokens", "input_image_tokens", "input_video_tokens",
                   "input_image_count", "input_video_count", "input_video_sha256",
                   "temporary_files_uploaded", "temporary_files_deleted",
                   "temporary_file_cleanup_errors"):
            if usage.get(k) is not None:
                entry[k] = usage[k]
        token_usage_total["per_round"].append(entry)
        if usage.get("input"):
            token_usage_total["total_input"] += usage["input"]
        if usage.get("output"):
            token_usage_total["total_output"] += usage["output"]
        if usage.get("thoughts"):
            token_usage_total["total_thoughts"] += usage["thoughts"]

    # ==================================================================
    # ROUND 1: context 帧 + 初始帧
    # ==================================================================
    context_prompt = ai._create_initial_prompt_section("context")
    instruction_prompt = ai._create_initial_prompt_section("instruction")
    initial_prompt = f"{context_prompt}\n\n{instruction_prompt}"
    current_b64 = session.get_frame_base64(len(session.frames) - 1)

    content, context_frames_b64, context_videos_b64 = ai._build_initial_message_content(
        current_b64, context_prompt, instruction_prompt
    )

    # 保存与模型实际请求完全一致的 context 媒体，供最终 chat.html 内嵌展示。
    ai._batch_context_frames_b64 = context_frames_b64
    ai._batch_context_videos_b64 = context_videos_b64

    api_messages = [{"role": "user", "content": content}]

    try:
        response_text, usage = await ai._call_model(api_messages)
    except Exception as e:
        return {"status": "error", "actions": [], "rewards": [], "rounds": [],
                "token_usage": token_usage_total, "error": f"Round 1 API error: {e}"}

    _record_usage(usage, 1)
    if usage.get("input") is None or usage.get("input") == 0:
        print(f"  [Warning] Round 1 input tokens is {usage.get('input')} — check API response")

    messages_history.append({"role": "user", "content": content})
    messages_history.append(ai._assistant_history_message(response_text))

    action = ai._parse_action(response_text)
    retry_count = 0
    max_retries = 3

    round_record = {
        "round": 1, "prompt": initial_prompt, "response": response_text,
        "message_content": content,
        "parsed_action": action,
        "action_meaning": action_info.get(action, str(action)) if action is not None else None,
        "is_valid": action is not None, "usage": usage, "retries": 0,
    }

    while action is None and retry_count < max_retries:
        retry_count += 1
        error_msg = (
            f"Your action is INVALID.\n\n"
            f"Your response MUST start with the action on the FIRST LINE:\n"
            f"[Action: <action_name>]\n\n"
            f"The square brackets [ ] are REQUIRED.\n"
            f"Use the EXACT action names I provided.\n"
            f"Put [Action: ...] FIRST, then reasoning after."
        )
        messages_history.append({"role": "user", "content": error_msg})

        try:
            response_text, usage = await ai._call_model(messages_history)
        except Exception as e:
            return {"status": "error", "actions": [], "rewards": [], "rounds": rounds_log,
                    "token_usage": token_usage_total, "error": f"Round 1 retry API error: {e}"}

        _record_usage(usage, 1)
        messages_history.append(ai._assistant_history_message(response_text))
        action = ai._parse_action(response_text)
        round_record["response"] = response_text
        round_record["parsed_action"] = action
        round_record["action_meaning"] = action_info.get(action, str(action)) if action is not None else None
        round_record["is_valid"] = action is not None
        round_record["retries"] = retry_count

    rounds_log.append(round_record)

    if action is None:
        return {"status": "error_parse", "actions": [], "rewards": [], "rounds": rounds_log,
                "token_usage": token_usage_total, "error": "Round 1: Failed to parse valid action after retries"}

    # ==================================================================
    # ROUND 2+
    # ==================================================================
    round_num = 2

    while not session.is_game_over:
        # 执行上一轮的动作
        new_b64 = await session.step(action)
        step += 1
        reward = session.last_reward
        cumulative_reward += reward

        actions_log.append(action)
        rewards_log.append(reward)

        if session.is_game_over:
            break

        # 构建 followup prompt
        followup_kwargs = dict(
            step=step,
            last_action=action,
            action_meaning=action_info.get(action, str(action)),
            reward=reward,
            cumulative_reward=cumulative_reward,
            is_terminated=session.last_info.get("terminated", False),
            is_truncated=session.last_info.get("truncated", False),
            info=session.last_info,
        )
        followup_context_prompt = ai._create_followup_prompt_section("context", **followup_kwargs)
        followup_instruction_prompt = ai._create_followup_prompt_section("instruction", **followup_kwargs)
        followup_prompt = f"{followup_context_prompt}\n\n{followup_instruction_prompt}"

        followup_content = ai._build_followup_message_content(
            new_b64, followup_context_prompt, followup_instruction_prompt
        )

        # frame_window 裁剪
        if ai.frame_window > 0:
            recent_count = (ai.frame_window - 1) * 2
            if recent_count == 0:
                history_to_send = messages_history[:2]
            elif len(messages_history) > 2 + recent_count:
                history_to_send = messages_history[:2] + messages_history[-recent_count:]
            else:
                history_to_send = list(messages_history)
        else:
            history_to_send = list(messages_history)

        api_messages = history_to_send + [{"role": "user", "content": followup_content}]

        try:
            response_text, usage = await ai._call_model(api_messages)
        except Exception as e:
            return {"status": "error", "actions": actions_log, "rewards": rewards_log,
                    "rounds": rounds_log, "token_usage": token_usage_total,
                    "error": f"Round {round_num} API error: {e}"}

        _record_usage(usage, round_num)

        messages_history.append({"role": "user", "content": followup_content})
        messages_history.append(ai._assistant_history_message(response_text))

        next_action = ai._parse_action(response_text)
        retry_count = 0

        round_record = {
            "round": round_num, "prompt": followup_prompt, "response": response_text,
            "message_content": followup_content,
            "parsed_action": next_action,
            "action_meaning": action_info.get(next_action, str(next_action)) if next_action is not None else None,
            "is_valid": next_action is not None, "usage": usage, "retries": 0,
        }

        while next_action is None and retry_count < max_retries:
            retry_count += 1
            error_msg = (
                f"Your action is INVALID and not in the action space.\n\n"
                f"You MUST respond using the exact format:\n"
                f"[Action: <action_name>]\n\n"
                f"Where <action_name> is one of the action names I provided.\n"
                f"The square brackets [ ] are REQUIRED.\n"
                f"Please try again with a valid action."
            )
            messages_history.append({"role": "user", "content": error_msg})

            try:
                response_text, usage = await ai._call_model(messages_history)
            except Exception as e:
                return {"status": "error", "actions": actions_log, "rewards": rewards_log,
                        "rounds": rounds_log, "token_usage": token_usage_total,
                        "error": f"Round {round_num} retry API error: {e}"}

            _record_usage(usage, round_num)
            messages_history.append(ai._assistant_history_message(response_text))
            next_action = ai._parse_action(response_text)
            round_record["response"] = response_text
            round_record["parsed_action"] = next_action
            round_record["action_meaning"] = action_info.get(next_action, str(next_action)) if next_action is not None else None
            round_record["is_valid"] = next_action is not None
            round_record["retries"] = retry_count

        rounds_log.append(round_record)

        if next_action is None:
            return {"status": "error_parse", "actions": actions_log, "rewards": rewards_log,
                    "rounds": rounds_log, "token_usage": token_usage_total,
                    "error": f"Round {round_num}: Failed to parse valid action after retries"}

        action = next_action
        round_num += 1

    # 游戏结束 — 最后一步的动作已记录
    game_over_reason = "terminated" if session.last_info.get("terminated") else "truncated"

    return {
        "status": "completed",
        "actions": actions_log,
        "rewards": rewards_log,
        "rounds": rounds_log,
        "token_usage": token_usage_total,
        "total_steps": step,
        "total_reward": cumulative_reward,
        "total_rounds": round_num - 1,
        "game_over_reason": game_over_reason,
        "error": None,
    }


# ---------------------------------------------------------------------------
# 6. 单次评测
# ---------------------------------------------------------------------------

async def run_single_eval(
    entry: dict,
    run_index: int,
    global_ai_config: dict,
    output_dir: Path,
    game_setting_overrides: dict[str, str] | None = None,
    *,
    detailed_rules: bool = False,
    context_transform=None,
    result_overrides: dict | None = None,
) -> dict:
    """运行单次评测。返回 result dict。"""
    game_setting_overrides = game_setting_overrides or {}
    game_id = entry["game_id"]
    config_name = entry["config_name"]
    config_dir = entry["config_dir"]
    config_data = entry["config_data"]
    game_safe = game_id.replace("/", "_").replace(":","_")

    run_output_dir = output_dir / game_safe / config_name / f"run_{run_index}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Eval] {game_id} / {config_name} / run_{run_index}")
    print(f"{'='*60}")

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()

    # --- 1. 创建 GameSession ---
    game_settings = dict(config_data.get("game_settings", {}))
    game_settings.update(game_setting_overrides)
    session_config = convert_game_settings(game_id, game_settings)
    session_config["render_mode"] = "rgb_array"

    session = GameSession(
        session_id=f"batch_{game_safe}_{run_index}_{int(time.time())}",
        game_name=game_id,
        config=session_config,
    )

    try:
        initial_b64 = await session.initialize()
    except Exception as e:
        result = _make_error_result(
            entry, run_index, global_ai_config, started_at, t0, str(e),
            result_overrides=result_overrides,
        )
        _atomic_write_json(run_output_dir / "result.json", result)
        return result

    # --- 2. 创建 AIPlayer 并配置 ---
    ai = AIPlayer(session)
    ai_kwargs = dict(
        api_key=global_ai_config.get("api_key", ""),
        base_url=global_ai_config.get("base_url", ""),
        model=global_ai_config.get("model", ""),
        temperature=config_data.get("temperature", global_ai_config.get("temperature", 0.7)),
        max_tokens=min(
            config_data.get("max_tokens", 50000),
            global_ai_config.get("max_tokens", 50000),
        ),
        game_rules=config_data.get("game_rules", ""),
        action_mode=config_data.get("action_mode", "natural_language"),
        frame_source=global_ai_config.get("frame_source", config_data.get("frame_source", "subbed_nl")),
        video_mode=global_ai_config.get("video_mode", config_data.get("video_mode", True)),
        api_mode=global_ai_config.get("api_mode", "gemini_native"),
        provider=global_ai_config.get("provider"),
        reasoning_effort=global_ai_config.get("reasoning_effort"),
        frame_window=config_data.get("frame_window", 0),
        hide_reward=config_data.get("hide_reward", True),
        detailed_game_rules=detailed_rules or global_ai_config.get("detailed_game_rules", False),
    )
    if config_data.get("video_description"):
        ai_kwargs["video_description"] = config_data["video_description"]
    ai.configure(**ai_kwargs)

    # --- 3. 加载 context ---
    context=global_ai_config.get("context",True)
    load_context_for_ai(ai, config_data, config_dir, context)
    if context_transform is not None:
        context_transform(ai, entry)

    # --- 4. 回放 action sequence ---
    action_seq = config_data.get("action_sequence")
    if action_seq and action_seq.get("name"):
        await replay_action_sequence(session, action_seq)
        if session.is_game_over:
            result = _make_error_result(
                entry, run_index, global_ai_config, started_at, t0,
                "Game over after action sequence replay",
                result_overrides=result_overrides,
            )
            _atomic_write_json(run_output_dir / "result.json", result)
            return result

    # --- 5. 运行评测主循环 ---
    loop_result = await batch_run_loop(ai, session)

    # --- 6. 整理并保存结果 ---
    finished_at = datetime.now().isoformat(timespec="seconds")
    duration = time.monotonic() - t0

    total_reward = loop_result.get("total_reward", sum(loop_result.get("rewards", [])))
    total_steps = loop_result.get("total_steps", len(loop_result.get("actions", [])))

    result = {
        "config_path": str(config_dir.relative_to(ROOT)),
        "game_id": game_id,
        "config_name": config_name,
        "run_index": run_index,
        "status": loop_result["status"],
        "total_steps": total_steps,
        "total_reward": total_reward,
        "max_steps": session.config.get("_max_steps"),
        "is_game_over": bool(session.is_game_over),
        "game_over_reason": loop_result.get("game_over_reason"),
        "ending": session.last_info.get("ending"),
        "total_rounds": loop_result.get("total_rounds", len(loop_result.get("rounds", []))),
        "actions": loop_result.get("actions", []),
        "rewards": loop_result.get("rewards", []),
        "token_usage": loop_result.get("token_usage", {}),
        "model": global_ai_config.get("model", ""),
        "api_config": {
            "base_url": global_ai_config.get("base_url", ""),
            "api_mode": ai.api_mode,
            "temperature": config_data.get("temperature", 0.7),
            "max_tokens": config_data.get("max_tokens", 50000),
            "action_mode": config_data.get("action_mode"),
            "frame_source": ai.frame_source,
            "video_mode": ai.video_mode,
            "frame_window": config_data.get("frame_window"),
            "hide_reward": config_data.get("hide_reward"),
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration, 1),
        "error": loop_result.get("error"),
        "context": context,
        "game_setting_overrides": game_setting_overrides,
    }
    if result_overrides:
        result.update(result_overrides)

    ending_type = None
    if loop_result["status"] == "completed":
        ending_type = session.last_info.get("ending", None)

    _save_full_result(
        run_output_dir, result,
        loop_result.get("rounds", []),
        session, ending_type, ai,
    )

    status_icon = "OK" if result["status"] == "completed" else "ERR"
    print(f"  [{status_icon}] Steps={result['total_steps']}, Reward={result['total_reward']}, "
          f"Rounds={result['total_rounds']}, Duration={result['duration_seconds']}s")

    # 清理 env
    session.close()

    return result


def _make_error_result(
    entry, run_index, global_ai_config, started_at, t0, error_msg,
    result_overrides: dict | None = None,
):
    result = {
        "config_path": str(entry["config_dir"].relative_to(ROOT)),
        "game_id": entry["game_id"],
        "config_name": entry["config_name"],
        "run_index": run_index,
        "status": "error",
        "total_steps": 0, "total_reward": 0.0,
        "is_game_over": False, "game_over_reason": None,
        "total_rounds": 0, "actions": [], "rewards": [],
        "token_usage": {}, "model": global_ai_config.get("model", ""),
        "api_config": {},
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(time.monotonic() - t0, 1),
        "error": error_msg,
    }
    if result_overrides:
        result.update(result_overrides)
    return result


def _collect_chat_context_media(ai: AIPlayer):
    """返回 batch 首轮实际发送给模型的 context 媒体。"""
    if ai is None:
        return [], []

    if ai.video_mode:
        context_videos_b64 = list(getattr(ai, "_batch_context_videos_b64", []))
        if not context_videos_b64 and ai.context_groups:
            context_videos_b64 = [
                group["video_b64"]
                for group in ai.context_groups
                if group.get("video_b64")
            ]
        return context_videos_b64, []

    context_frames_b64 = list(getattr(ai, "_batch_context_frames_b64", []))
    if not context_frames_b64:
        context_frames_b64 = list(ai.context_frames)
    return [], context_frames_b64


def _save_full_result(run_dir: Path, result: dict, rounds: list,
                      session: GameSession, ending_type: str,
                      ai: AIPlayer = None):
    """完整保存评测结果，复刻前端 _do_save 的全部输出。

    生成: frames/, subbed/, subbed_nl/, caption.txt, caption_nl.txt,
          actions.json, video.mp4, video_nl.mp4, video_original.mp4,
          trajectory.png, chat.html, result.json, conversation.json
    """
    import io
    from PIL import Image

    # result.json + conversation.json
    _atomic_write_json(run_dir / "result.json", result)
    if rounds:
        _atomic_write_json(run_dir / "conversation.json", {"rounds": rounds})

    # 提取 AI 部分的帧/动作/奖励（去掉 action_sequence 回放的前缀）
    seq_count = getattr(session, "_seq_action_count", 0)
    frames_jpeg = session.frames[seq_count:]  # JPEG bytes list
    action_hist = session.action_history[seq_count:]
    reward_hist = session.reward_history[seq_count:]
    action_info = session.action_info
    game_name = session.game_name

    if not frames_jpeg:
        return

    # --- 帧目录 ---
    frames_dir = run_dir / "frames"
    subbed_dir = run_dir / "subbed"
    subbed_nl_dir = run_dir / "subbed_nl"
    frames_dir.mkdir(parents=True, exist_ok=True)
    subbed_dir.mkdir(parents=True, exist_ok=True)
    subbed_nl_dir.mkdir(parents=True, exist_ok=True)

    def _make_caption(idx):
        if idx == 0:
            return "Start State"
        act_idx = idx - 1
        if act_idx < len(action_hist):
            act = action_hist[act_idx]
            return f"Step {idx}: Action {act}"
        return f"Step {idx}"

    def _make_caption_nl(idx):
        if idx == 0:
            return "Start State"
        act_idx = idx - 1
        if act_idx < len(action_hist):
            act_id = action_hist[act_idx]
            act_name = action_info.get(act_id, f"action{act_id}")
            return f"Step {idx}: Action: {act_name}"
        return f"Step {idx}"

    pil_frames = []
    pil_subbed = []
    pil_subbed_nl = []

    for i, jpeg_bytes in enumerate(frames_jpeg):
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        img.save(frames_dir / f"step_{i:06d}.png")
        pil_frames.append(img)

        subbed_img = add_caption(img, _make_caption(i))
        subbed_img.save(subbed_dir / f"step_{i:06d}.png")
        pil_subbed.append(subbed_img)

        subbed_nl_img = add_caption(img, _make_caption_nl(i))
        subbed_nl_img.save(subbed_nl_dir / f"step_{i:06d}.png")
        pil_subbed_nl.append(subbed_nl_img)

    # 追加结束类型标注帧
    if ending_type and pil_frames:
        ending_idx = len(frames_jpeg)
        ending_frame = add_ending_label(pil_frames[-1].copy(), ending_type)
        ending_frame.save(frames_dir / f"step_{ending_idx:06d}.png")
        pil_frames.append(ending_frame)

        ending_subbed = add_ending_label(pil_subbed[-1].copy(), ending_type)
        ending_subbed.save(subbed_dir / f"step_{ending_idx:06d}.png")
        pil_subbed.append(ending_subbed)

        ending_subbed_nl = add_ending_label(pil_subbed_nl[-1].copy(), ending_type)
        ending_subbed_nl.save(subbed_nl_dir / f"step_{ending_idx:06d}.png")
        pil_subbed_nl.append(ending_subbed_nl)

    # --- 字幕文本 ---
    captions = [_make_caption(i) for i in range(len(frames_jpeg))]
    (run_dir / "caption.txt").write_text("\n".join(captions), encoding="utf-8")
    captions_nl = [_make_caption_nl(i) for i in range(len(frames_jpeg))]
    (run_dir / "caption_nl.txt").write_text("\n".join(captions_nl), encoding="utf-8")

    # --- actions.json ---
    seed = session.config.get("_seed")
    actions_data = {
        "game_name": game_name,
        "config": {k: v for k, v in session.config.items() if not k.startswith("_")},
        "seed": seed,
        "total_steps": len(action_hist),
        "total_frames": len(frames_jpeg),
        "episode_reward": session.episode_reward,
        "actions": list(action_hist),
        "rewards": list(reward_hist),
        "video_fps": 1.0,
        "ending": ending_type,
    }
    _atomic_write_json(run_dir / "actions.json", actions_data)

    # --- 视频 ---
    save_video(pil_subbed, run_dir / "video.mp4", 1.0)
    save_video(pil_subbed_nl, run_dir / "video_nl.mp4", 1.0)
    save_video(pil_frames, run_dir / "video_original.mp4", 1.0)

    # --- 轨迹图 ---
    try:
        trajectory_img = generate_trajectory_image(session,seq_count)
        if trajectory_img is not None:
            if ending_type:
                trajectory_img = add_ending_label(trajectory_img, ending_type)
            trajectory_img.save(run_dir / "trajectory.png")
    except Exception as e:
        print(f"  [Warning] Trajectory image failed: {e}")

    # --- chat.html ---
    # 收集 context 媒体数据供 chat.html 使用
    context_videos_b64, context_frames_b64 = _collect_chat_context_media(ai)

    # 游戏帧 b64（去掉 action_sequence 回放前缀的部分）
    game_frames_b64 = []
    for jpeg_bytes in frames_jpeg:
        game_frames_b64.append(base64.b64encode(jpeg_bytes).decode("ascii"))

    _save_chat_html(
        run_dir, rounds, game_name, action_info,
        context_videos_b64=context_videos_b64,
        context_frames_b64=context_frames_b64,
        game_frames_b64=game_frames_b64,
        video_mode=ai.video_mode if ai else False,
    )


def _atomic_write_json(path: Path, data: dict):
    """先写 .tmp 再 rename，保证原子性。"""
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _render_actual_message_content(content, html_mod):
    """按 API 实际发送的 content 顺序渲染文本、视频和图片。"""
    if not isinstance(content, list):
        return f'<div class="chat-msg-text">{html_mod.escape(str(content or ""))}</div>'

    blocks = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            blocks.append(
                f'<div class="chat-msg-text">{html_mod.escape(part.get("text", ""))}</div>'
            )
            continue
        if part.get("type") != "image_url":
            continue

        url = part.get("image_url", {}).get("url", "")
        safe_url = html_mod.escape(url, quote=True)
        if url.startswith("data:video/"):
            blocks.append(
                '<div class="chat-media-block">'
                f'<video controls style="max-width:100%;border-radius:4px;" src="{safe_url}"></video>'
                '</div>'
            )
        elif url.startswith("data:image/"):
            blocks.append(
                '<div class="chat-media-block">'
                f'<img class="chat-thumb" src="{safe_url}" style="border:2px solid #64b5f6;">'
                '</div>'
            )
    return "".join(blocks)


def _save_chat_html(run_dir: Path, rounds: list, game_name: str, action_info: dict,
                    *, context_videos_b64: list = None, context_frames_b64: list = None,
                    game_frames_b64: list = None, video_mode: bool = False):
    """从 conversation rounds 生成 chat.html，复刻前端 appendChatPrompt / updateChatResponse 样式。

    嵌入 base64 图片/视频，和前端 save 出来的 chat.html 一致。
    """
    if not rounds:
        return

    import html as html_mod

    context_videos_b64 = context_videos_b64 or []
    context_frames_b64 = context_frames_b64 or []
    game_frames_b64 = game_frames_b64 or []

    msg_blocks = []
    for r in rounds:
        round_num = r.get("round", "?")
        step = round_num - 1  # step 从 0 开始，round 从 1 开始
        prompt = html_mod.escape(r.get("prompt", ""))
        response = html_mod.escape(r.get("response", ""))
        action = r.get("parsed_action")
        meaning = r.get("action_meaning", "")
        is_valid = r.get("is_valid", False)
        retries = r.get("retries", 0)
        usage = r.get("usage", {})
        is_init = (round_num == 1)

        # ===================== 用户消息（左边）=====================
        media_html = '<div class="chat-media-blocks">'

        if is_init:
            # Context 媒体
            if video_mode and context_videos_b64:
                for vi, vid_b64 in enumerate(context_videos_b64):
                    label = f"Demo video {'#' + str(vi + 1) + ' ' if len(context_videos_b64) > 1 else ''}(context)"
                    media_html += f'''
                      <div class="chat-media-block">
                        <div class="chat-media-label">&#x1f4f9; {label}</div>
                        <video controls style="max-width:100%;border-radius:4px;">
                          <source src="data:video/mp4;base64,{vid_b64}" type="video/mp4">
                        </video>
                      </div>'''
            elif context_frames_b64:
                media_html += '<div class="chat-media-block"><div class="chat-media-label">&#x1f4f8; Demo frames (context)</div><div class="chat-frames">'
                for fi, f_b64 in enumerate(context_frames_b64):
                    media_html += f'<img class="chat-thumb chat-thumb-ctx" src="data:image/jpeg;base64,{f_b64}" title="Context frame {fi + 1}">'
                media_html += '</div></div>'

        # 当前游戏帧
        # Round 1 → game_frames_b64[0] (initial frame)
        # Round N → game_frames_b64[N-1] (step N-1 之后的帧)
        frame_idx = step  # round_num - 1
        if frame_idx < len(game_frames_b64):
            frame_label = '&#x1f3af; Initial state (what AI plays)' if is_init else '&#x1f4cd; Current state'
            media_html += f'''
              <div class="chat-media-block">
                <div class="chat-media-label">{frame_label}</div>
                <img class="chat-thumb" src="data:image/jpeg;base64,{game_frames_b64[frame_idx]}"
                  title="Step {step} frame" style="border:2px solid #64b5f6;">
              </div>'''

        media_html += '</div>'

        retry_label = ''
        if retries > 0:
            retry_label = f'<span style="color:#ef6c00;font-size:11px;">[Retry {retries}] </span>'

        user_html = (
            f'<div class="chat-msg chat-msg-user{" chat-msg-retry" if retries > 0 else ""}">'
            f'<div class="chat-msg-role">User &mdash; Step {step + 1}</div>'
            f'{media_html}'
            f'<div class="chat-msg-text">{retry_label}{prompt}</div>'
            f'</div>'
        )

        # 新 batch 记录直接以 API message content 为准，避免重新套用模板或推导媒体。
        if "message_content" in r:
            actual_content_html = _render_actual_message_content(r["message_content"], html_mod)
            user_html = (
                f'<div class="chat-msg chat-msg-user{" chat-msg-retry" if retries > 0 else ""}">'
                f'<div class="chat-msg-role">User &mdash; Step {step + 1}</div>'
                f'{actual_content_html}'
                f'</div>'
            )

        # ===================== AI 回复（右边）=====================
        ai_cls = "chat-msg chat-msg-ai"
        if not is_valid:
            ai_cls += " chat-msg-invalid"

        # Thinking 折叠区
        thinking_html = ''
        thinking_text = usage.get("thinking")
        if thinking_text:
            thinking_html = (
                f'<details class="chat-thinking">'
                f'<summary>&#x1f4ad; Thinking ({len(thinking_text):,} chars)</summary>'
                f'<div class="chat-thinking-body">{html_mod.escape(thinking_text)}</div>'
                f'</details>'
            )

        # Parsed action 框
        if is_valid:
            action_bg = "#f5f5f5"
            action_color = "#666"
            action_border = "#e0e0e0"
            action_text = f"{action}"
            if meaning:
                action_text += f" - {meaning}"
        else:
            action_bg = "#1a0000"
            action_color = "#ef5350"
            action_border = "#c62828"
            action_text = "(invalid)"

        action_box = (
            f'<div style="margin-top:8px;padding:6px 10px;background:{action_bg};'
            f'border-radius:4px;font-size:0.9em;color:{action_color};border:1px solid {action_border};">'
            f'<strong>Parsed action:</strong> {action_text}</div>'
        )

        # Token usage
        usage_html = ''
        if usage.get("input") is not None or usage.get("output") is not None:
            parts = []
            if usage.get("input") is not None:
                parts.append(f"Input: {usage['input']:,}")
            if usage.get("output") is not None:
                parts.append(f"Output: {usage['output']:,}")
            if usage.get("thoughts") is not None:
                parts.append(f"Thoughts: {usage['thoughts']:,}")
            if usage.get("total") is not None:
                parts.append(f"Total: {usage['total']:,}")
            line = " &middot; ".join(parts) + " tokens"

            media_parts = []
            if usage.get("input_image_tokens") is not None:
                media_parts.append(f"img: {usage['input_image_tokens']:,} tok")
            elif usage.get("input_image_count") is not None:
                media_parts.append(f"{usage['input_image_count']} img")
            if usage.get("input_video_tokens") is not None:
                media_parts.append(f"video: {usage['input_video_tokens']:,} tok")
            elif usage.get("input_video_count") is not None:
                media_parts.append(f"{usage['input_video_count']} video")
            if media_parts:
                line += f'<br><span style="color:#8bc34a">{" &middot; ".join(media_parts)}</span>'

            usage_html = f'<div style="margin-top:4px;font-size:0.8em;color:#999;">{line}</div>'

        ai_html = (
            f'<div class="{ai_cls}">'
            f'<div class="chat-msg-role">AI &rarr; Step {step + 1}</div>'
            f'{thinking_html}'
            f'<div class="chat-msg-text">{response}</div>'
            f'{action_box}{usage_html}'
            f'</div>'
        )

        msg_blocks.append(f'<div class="chat-turn">{user_html}{ai_html}</div>')

    chat_html_content = "\n".join(msg_blocks)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Chat Log — {game_name} {timestamp}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Courier New',monospace;background:#111122;color:#e0e0e0;padding:20px}}
h1{{color:#64b5f6;font-size:16px;margin-bottom:16px}}
#chat-messages{{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-content:start}}
.chat-turn{{display:contents}}
.chat-msg{{border-radius:8px;padding:12px 14px;font-size:12px;line-height:1.6;word-break:break-word;white-space:pre-wrap}}
.chat-msg-role{{font-size:11px;font-weight:bold;margin-bottom:8px;opacity:.8;white-space:nowrap}}
.chat-msg-user{{background:#1a2a4a;border:1px solid #2a4a7a}}
.chat-msg-user .chat-msg-role{{color:#64b5f6}}
.chat-msg-ai{{background:#1a2a1a;border:1px solid #2a4a2a}}
.chat-msg-ai .chat-msg-role{{color:#81c784}}
.chat-msg-retry{{background:#1a1000;border:1px solid #e65100}}
.chat-msg-retry .chat-msg-role{{color:#ef6c00}}
.chat-msg-invalid{{background:#1a0000;border:2px solid #c62828}}
.chat-msg-invalid .chat-msg-role{{color:#ef5350}}
.chat-msg-text{{color:#ccc}}
.chat-media-blocks{{margin:8px 0}}
.chat-media-block{{margin-bottom:8px}}
.chat-media-label{{font-size:10px;color:#888;margin-bottom:4px}}
.chat-frames{{display:flex;flex-wrap:wrap;gap:4px}}
.chat-thumb{{max-width:200px;max-height:150px;border-radius:4px;cursor:pointer;object-fit:contain}}
.chat-thumb-ctx{{max-width:120px;max-height:90px;opacity:.85}}
.chat-thinking{{margin:6px 0;font-size:11px;color:#999}}
.chat-thinking summary{{cursor:pointer;color:#aaa}}
.chat-thinking-body{{margin-top:4px;padding:8px;background:rgba(255,255,255,.03);border-radius:4px;white-space:pre-wrap;max-height:400px;overflow-y:auto}}
video{{max-width:100%;max-height:300px;border-radius:4px}}
</style>
</head>
<body>
<h1>Chat Log — {game_name} &nbsp; {timestamp}</h1>
<div id="chat-messages">{chat_html_content}</div>
</body>
</html>"""
    (run_dir / "chat.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Main — 并发 + 断点续传 + summary
# ---------------------------------------------------------------------------

def _is_completed(output_dir: Path, game_safe: str, config_name: str, run_index: int) -> bool:
    """检查某次评测是否已完成（断点续传用）。"""
    result_file = output_dir / game_safe / config_name / f"run_{run_index}" / "result.json"
    if not result_file.exists():
        return False
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
        return data.get("status") == "completed"
    except Exception:
        return False


def parse_game_setting_overrides(items: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set key is empty: {item}")
        overrides[key] = value
    return overrides


async def main():
    parser = argparse.ArgumentParser(description="Batch AI Evaluation Tool")
    parser.add_argument("--runs", type=int, default=1, help="每个 config 跑几局 (default: 1)")
    parser.add_argument("--workers", type=int, default=3, help="最大并发数 (default: 3)")
    parser.add_argument("--game", type=str, default=None, help="按游戏 ID 过滤")
    parser.add_argument("--config", type=str, default=None, help="按 config 名过滤 (子串匹配)")
    parser.add_argument("--output", type=str, default="batch_results", help="输出目录")
    parser.add_argument("--resume", type=str, default=None, help="续传之前的 run_id")
    parser.add_argument("--detailed-rules", action="store_true",
                        help="用 prompt/templates/detailed_game_rules.json 替代 game_rules.json")
    parser.add_argument(
        "--set",
        dest="game_setting_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override game_settings for this run; repeatable, e.g. --set cfg-max-steps=1",
    )
    args = parser.parse_args()
    try:
        game_setting_overrides = parse_game_setting_overrides(args.game_setting_overrides)
    except ValueError as e:
        parser.error(str(e))

    # 扫描 configs
    configs = scan_configs(game_filter=args.game, config_filter=args.config)
    if not configs:
        print("[Info] 没有找到匹配的 ai_config。")
        sys.exit(0)

    print(f"[Info] Found {len(configs)} configs, {args.runs} run(s) each, "
          f"{len(configs) * args.runs} total tasks, workers={args.workers}")

    # 确定输出目录
    if args.resume:
        run_id = args.resume
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (Path(args.output) / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 全局 AI 配置：resume 时读存档的 config.json（保证用原批次的模型），
    # 新批次时读当前根 config.json 并存档到 run_id 目录
    archived_config = output_dir / "config.json"
    if args.resume and archived_config.exists():
        try:
            with open(archived_config, "r", encoding="utf-8") as f:
                global_ai_config = json.load(f).get("ai", {})
            print(f"[Resume] Loaded archived config from {archived_config}")
        except Exception as e:
            print(f"[Error] 读取存档 config 失败 ({archived_config}): {e}")
            sys.exit(1)
    else:
        global_ai_config = DEFAULT_CONFIG.get("ai", {})
        src_config = ROOT / "config.json"
        if src_config.exists():
            shutil.copy2(src_config, archived_config)

    if not global_ai_config.get("api_key"):
        print("[Error] ai.api_key 未配置，请先设置。")
        sys.exit(1)

    if game_setting_overrides:
        print(f"[Info] Game setting overrides: {game_setting_overrides}")
    print(f"[Info] Output: {output_dir}")
    print(f"[Info] Model: {global_ai_config.get('model', '?')}")
    print(f"[Info] API mode: {global_ai_config.get('api_mode', 'gemini_native')}")

    # 构建任务列表（跳过已完成的）
    tasks_to_run = []
    skipped = 0
    for entry in configs:
        game_safe = entry["game_id"].replace("/", "_").replace(":","_")
        for run_i in range(args.runs):
            if _is_completed(output_dir, game_safe, entry["config_name"], run_i):
                skipped += 1
                continue
            tasks_to_run.append((entry, run_i))

    if skipped:
        print(f"[Resume] Skipping {skipped} already completed task(s)")
    if not tasks_to_run:
        print("[Info] All tasks already completed!")
        _write_summary(output_dir, configs, args.runs)
        sys.exit(0)

    # pygame.surfarray is lazily loaded by pygame. Force it to finish loading
    # before concurrent GameSession initialization can access it from threads.
    ensure_pygame_surfarray()

    print(f"[Info] Running {len(tasks_to_run)} task(s)...")

    # 并发执行
    semaphore = asyncio.Semaphore(args.workers)
    all_results = []

    async def run_with_semaphore(entry, run_i):
        async with semaphore:
            try:
                return await run_single_eval(entry, run_i, global_ai_config, output_dir, game_setting_overrides,
                                              detailed_rules=args.detailed_rules)
            except Exception as e:
                print(f"  [FATAL] {entry['game_id']}/{entry['config_name']}/run_{run_i}: {e}")
                return {"status": "error", "error": str(e),
                        "game_id": entry["game_id"], "config_name": entry["config_name"],
                        "run_index": run_i}

    tasks = [run_with_semaphore(entry, run_i) for entry, run_i in tasks_to_run]
    all_results = await asyncio.gather(*tasks)

    # 写 summary
    _write_summary(output_dir, configs, args.runs)

    # 打印汇总
    completed = sum(1 for r in all_results if r.get("status") == "completed")
    errors = sum(1 for r in all_results if r.get("status") != "completed")
    print(f"\n{'='*60}")
    print(f"[Summary] Completed: {completed}, Errors: {errors}, Skipped: {skipped}")
    print(f"[Summary] Results: {output_dir}")
    print(f"{'='*60}")


def _write_summary(output_dir: Path, configs: list, runs: int, summary_overrides: dict | None = None):
    """汇总所有 result.json → summary.json"""
    all_results = []
    for entry in configs:
        game_safe = entry["game_id"].replace("/", "_").replace(":", "_")
        for run_i in range(runs):
            result_file = output_dir / game_safe / entry["config_name"] / f"run_{run_i}" / "result.json"
            if result_file.exists():
                try:
                    all_results.append(json.loads(result_file.read_text(encoding="utf-8")))
                except Exception:
                    pass

    summary = {
        "run_id": output_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_configs": len(configs),
        "runs_per_config": runs,
        "total_tasks": len(configs) * runs,
        "completed": sum(1 for r in all_results if r.get("status") == "completed"),
        "errors": sum(1 for r in all_results if r.get("status") != "completed"),
        "results": [
            {
                "game_id": r.get("game_id"),
                "config_name": r.get("config_name"),
                "run_index": r.get("run_index"),
                "status": r.get("status"),
                "total_steps": r.get("total_steps"),
                "total_reward": r.get("total_reward"),
                "total_rounds": r.get("total_rounds"),
                "ending": r.get("ending"),
                "duration_seconds": r.get("duration_seconds"),
                "error": r.get("error"),
            }
            for r in all_results
        ],
    }
    if summary_overrides:
        summary.update(summary_overrides)
    _atomic_write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    asyncio.run(main())
