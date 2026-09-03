"""WebSocket 连接处理

处理实时游戏交互的 WebSocket 通信
"""
import asyncio
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from src.ai import AIPlayer
from src.utils.human_eval_recorder import (
    get_human_evaluation_lock,
    reset_human_evaluation_tracking,
    save_human_evaluation_once_locked,
)


def register_websocket(app, sessions: dict, ai_players: dict,
                       human_eval_root: Path = None):
    """
    注册 WebSocket 端点

    Args:
        app: FastAPI 应用实例
        sessions: 会话存储字典
        ai_players: AI 玩家存储字典
    """

    if human_eval_root is None:
        human_eval_root = Path(__file__).resolve().parent.parent.parent / "human_eval_results"
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(ws: WebSocket, session_id: str):
        await ws.accept()

        session = sessions.get(session_id)
        if not session:
            await ws.send_json({"type": "error", "message": "Session not found"})
            await ws.close()
            return

        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")

                if msg_type == "ping":
                    await ws.send_json({"type": "pong"})

                elif msg_type == "action":
                    action = int(data["action"])
                    human_mode = getattr(session, "interaction_mode", "human") in ("human", "crossval")
                    lock = get_human_evaluation_lock(session) if human_mode else None
                    if lock:
                        await lock.acquire()
                    try:
                        new_b64 = await session.step(action)
                        state = session.get_state_info()
                        if human_mode and session.is_game_over:
                            try:
                                run_dir, _ = await save_human_evaluation_once_locked(
                                    session, human_eval_root
                                )
                                state["human_eval_path"] = str(run_dir)
                                print(f"[HumanEval] Saved: {run_dir}")
                            except Exception as exc:
                                state["human_eval_save_error"] = str(exc)
                                print(f"[HumanEval] Save failed: {exc}")
                    finally:
                        if lock:
                            lock.release()
                    await ws.send_json({"type": "frame", "frame": new_b64, "state": state})

                elif msg_type == "reset":
                    # 停止正在运行的 AI
                    ai = ai_players.get(session_id)
                    if ai:
                        ai.stop()
                    human_mode = getattr(session, "interaction_mode", "human") in ("human", "crossval")
                    if human_mode:
                        async with get_human_evaluation_lock(session):
                            new_b64 = await session.reset()
                            reset_human_evaluation_tracking(session)
                    else:
                        new_b64 = await session.reset()
                    state = session.get_state_info()
                    await ws.send_json({"type": "frame", "frame": new_b64, "state": state})

                elif msg_type == "undo":
                    b64 = await session.undo()
                    if b64:
                        state = session.get_state_info()
                        await ws.send_json({"type": "frame", "frame": b64, "state": state})
                    else:
                        await ws.send_json({"type": "error", "message": "Nothing to undo"})

                elif msg_type == "replay_sequence":
                    actions = data.get("actions", [])
                    delay_ms = data.get("delay_ms", 80)

                    # 停止运行中的 AI
                    ai = ai_players.get(session_id)
                    if ai:
                        ai.stop()

                    # Reset 游戏
                    async with get_human_evaluation_lock(session):
                        new_b64 = await session.reset()
                        reset_human_evaluation_tracking(session)
                    state = session.get_state_info()
                    await ws.send_json({"type": "frame", "frame": new_b64, "state": state})

                    # 通知回放开始
                    await ws.send_json({"type": "replay_start", "total_actions": len(actions)})

                    # 回放期间临时禁用 max_steps 和 end_on_life_loss（序列是设定初始状态，不受限制）
                    saved_max_steps = session.config.get("_max_steps")
                    session.config["_max_steps"] = None
                    saved_end_on_life_loss = getattr(session.env, 'end_on_life_loss', False) if session.env else False
                    if session.env and hasattr(session.env, 'end_on_life_loss'):
                        session.env.end_on_life_loss = False

                    # 逐个执行动作
                    for action in actions:
                        if session.is_game_over:
                            break
                        new_b64 = await session.step(int(action))
                        state = session.get_state_info()
                        await ws.send_json({"type": "frame", "frame": new_b64, "state": state})
                        await asyncio.sleep(delay_ms / 1000.0)

                    # 恢复 max_steps 和 end_on_life_loss，记录序列边界，重置计数（序列步数不算）
                    if "freeway" in session.game_name.lower() or "cliffwalking" in session.game_name.lower():
                        session.env.env.state_reset()

                    session.config["_max_steps"] = saved_max_steps
                    if session.env and hasattr(session.env, 'end_on_life_loss'):
                        session.env.end_on_life_loss = saved_end_on_life_loss
                    session._seq_action_count = len(session.action_history)
                    session.step_count = 0
                    session.episode_reward = 0.0
                    session.event_counts = {}
                    session.is_game_over = False
                    # Treat the replayed sequence as part of the initial state,
                    # matching the batch pipeline's preload boundary.
                    if session.env is not None and "custombreakout" not in session.game_name.lower():
                        session.env.reset_tracking_state()
                    session.score_state.reset(session.env)
                    for key in ("ending", "pass", "truncated"):
                        session.last_info.pop(key, None)
                    # 清除序列步骤的撤回历史，只保留当前状态作为新起点
                    # （序列作为"初始状态"不可撤销）
                    session._snapshots = session._snapshots[-1:]

                    # 通知回放结束
                    await ws.send_json({
                        "type": "replay_done",
                        "total_steps": 0,
                        "is_game_over": session.is_game_over,
                    })

                elif msg_type == "get_frame":
                    idx = int(data["index"])
                    b64 = session.get_frame_base64(idx)
                    if b64:
                        await ws.send_json({
                            "type": "frame",
                            "frame": b64,
                            "state": session.get_state_info(),
                            "index": idx,
                            "total": len(session.frames),
                        })

                # ---- AI 控制 ----
                elif msg_type == "ai_start":
                    ai = ai_players.get(session_id)
                    if not ai:
                        ai = AIPlayer(session)
                        ai_players[session_id] = ai
                    if not ai.is_running:
                        skip_reset = data.get("skip_reset", False)
                        if not skip_reset:
                            new_b64 = await session.reset()
                            await ws.send_json({
                                "type": "frame",
                                "frame": new_b64,
                                "state": session.get_state_info(),
                            })
                        ai.start(ws)

                elif msg_type == "ai_retry":
                    ai = ai_players.get(session_id)
                    if ai and hasattr(ai, '_retry_event'):
                        ai._retry_choice = "retry"
                        ai._retry_event.set()

                elif msg_type == "ai_stop":
                    ai = ai_players.get(session_id)
                    if ai:
                        if hasattr(ai, '_retry_event'):
                            ai._retry_choice = "stop"
                            ai._retry_event.set()
                        ai.stop()

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
