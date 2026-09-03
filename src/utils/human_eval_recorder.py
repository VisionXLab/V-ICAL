"""Web 人类评测的轻量轨迹保存。"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path


def get_human_evaluation_lock(session) -> asyncio.Lock:
    lock = getattr(session, "_human_eval_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        session._human_eval_lock = lock
    return lock


def reset_human_evaluation_tracking(session) -> None:
    session._human_eval_started_at = datetime.now().isoformat(timespec="seconds")
    session._human_eval_started_monotonic = time.monotonic()
    session._human_eval_saved = False
    session._human_eval_path = None


def _atomic_write_json(path: Path, data: dict) -> None:
    """先写同目录临时文件，再原子替换目标 JSON。"""
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _allocate_run_dir(output_dir: Path, session) -> tuple[Path, int]:
    game_name = session.game_name
    game_safe = game_name.replace("/", "_").replace(":", "_")
    config_name = getattr(session, "evaluation_config_name", None)
    if not config_name:
        config_name = "crossval" if getattr(session, "interaction_mode", "human") == "crossval" else "human"
    config_safe = str(config_name).replace("/", "_").replace("\\", "_").replace(":", "_")
    config_dir = output_dir / game_safe / config_safe
    config_dir.mkdir(parents=True, exist_ok=True)

    # Cross-validation 对同一个“游戏 + AI 配置”只保留一份人工结果。
    # 重测或刷新后再次完成时原子覆盖 run_0，避免产生假的重复样本。
    if getattr(session, "interaction_mode", "human") == "crossval":
        run_dir = config_dir / "run_0"
        run_dir.mkdir(exist_ok=True)
        return run_dir, 0

    run_index = 0
    while True:
        run_dir = config_dir / f"run_{run_index}"
        try:
            run_dir.mkdir()
            return run_dir, run_index
        except FileExistsError:
            run_index += 1


def save_human_evaluation(session, output_dir: Path) -> Path:
    """按 batch 目录层级保存一局人类评测，仅生成两个 JSON 文件。"""
    run_dir, run_index = _allocate_run_dir(output_dir, session)
    seq_count = getattr(session, "_seq_action_count", 0)
    actions = list(session.action_history[seq_count:])
    rewards = list(session.reward_history[seq_count:])
    finished_at = datetime.now().isoformat(timespec="seconds")
    started_at = getattr(session, "_human_eval_started_at", finished_at)
    started_monotonic = getattr(session, "_human_eval_started_monotonic", None)
    duration = (
        round(time.monotonic() - started_monotonic, 1)
        if started_monotonic is not None
        else None
    )

    ending = session.last_info.get("ending")
    passed = session.last_info.get("pass")
    if ending == "manual_end":
        game_over_reason = "manual_end"
    elif session.last_info.get("terminated"):
        game_over_reason = "terminated"
    elif session.last_info.get("truncated"):
        game_over_reason = "truncated"
    else:
        game_over_reason = None

    actions_data = {
        "game_name": session.game_name,
        "config": {
            key: value
            for key, value in session.config.items()
            if not key.startswith("_")
        },
        "seed": session.config.get("_seed"),
        "total_steps": len(actions),
        "total_frames": len(actions) + 1,
        "episode_reward": sum(rewards),
        "actions": actions,
        "rewards": rewards,
        "ending": ending,
        "pass": passed,
    }
    result_data = {
        "game_id": session.game_name,
        "config_name": getattr(session, "evaluation_config_name", None) or "human",
        "run_index": run_index,
        "status": "completed",
        "total_steps": len(actions),
        "total_reward": sum(rewards),
        "max_steps": session.config.get("_max_steps"),
        "is_game_over": bool(session.is_game_over),
        "game_over_reason": game_over_reason,
        "ending": ending,
        "pass": passed,
        "score": session.last_info.get("score"),
        "actions": actions,
        "rewards": rewards,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration,
        "error": None,
    }

    _atomic_write_json(run_dir / "actions.json", actions_data)
    _atomic_write_json(run_dir / "result.json", result_data)
    return run_dir


async def save_human_evaluation_once_locked(session, output_dir: Path) -> tuple[Path, bool]:
    """调用方持有 session lock 时执行幂等保存。"""
    if getattr(session, "_human_eval_saved", False):
        return Path(session._human_eval_path), True
    path = await asyncio.to_thread(save_human_evaluation, session, output_dir)
    session._human_eval_saved = True
    session._human_eval_path = str(path)
    return path, False
