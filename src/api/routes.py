"""REST API 路由定义

包含所有 HTTP 端点的处理逻辑
"""
import asyncio
import base64
import io
import json
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image

from src.core import GameSession, get_prompt_manager
from src.ai import AIPlayer
from src.utils import DEFAULT_CONFIG
from src.utils.human_eval_recorder import (
    get_human_evaluation_lock,
    reset_human_evaluation_tracking,
    save_human_evaluation_once_locked,
)
from src.utils.score_config import REWARD_AS_SCORE_GAMES, get_score_game_ids
from src.video import get_op_dir, add_caption, add_ending_label, save_video, load_frames_from_source, generate_trajectory_image
def register_routes(app, sessions: dict, ai_players: dict, games_dict: dict,
                    static_dir: Path, human_op_dir: Path, ai_op_dir: Path,
                    crossval_op_dir: Path = None, human_eval_root: Path = None):
    """
    注册所有 REST API 路由

    Args:
        app: FastAPI 应用实例
        sessions: 会话存储字典
        ai_players: AI 玩家存储字典
        games_dict: 游戏定义字典
        static_dir: 静态文件目录
        human_op_dir: 人类操作存储目录
        ai_op_dir: AI 操作存储目录
        crossval_op_dir: 交叉验证（人类代入 AI 配置）操作存储目录
    """
    if crossval_op_dir is None:
        crossval_op_dir = Path(__file__).resolve().parent.parent.parent / "crossval_operation"
        crossval_op_dir.mkdir(exist_ok=True)
    if human_eval_root is None:
        human_eval_root = Path(__file__).resolve().parent.parent.parent / "human_eval_results"

    # ---- Game Tags 文件读写 ----
    game_tags_path = Path(__file__).resolve().parent.parent.parent / "prompt" / "game_tags.json"

    def _load_game_tags() -> dict:
        if game_tags_path.exists():
            with open(game_tags_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_game_tags(tags: dict):
        with open(game_tags_path, "w", encoding="utf-8") as f:
            json.dump(tags, f, indent=2, ensure_ascii=False)
            f.write("\n")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = static_dir / "index.html"
        if not html_path.exists():
            return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/games")
    async def get_games():
        all_tags = _load_game_tags()
        result = []
        for key, (game_id, desc, tag) in games_dict.items():
            entry = {
                "key": key,
                "game_id": game_id,
                "description": desc,
                "tag": tag,
                "rules": get_prompt_manager().get_game_rules(game_id),
                "game_tags": all_tags.get(game_id),  # None if not tagged
            }
            result.append(entry)
        return result

    @app.get("/api/config/ai-defaults")
    async def get_ai_defaults():
        """返回 config.json 中的 AI 默认配置（不含 api_key 明文，仅返回是否已配置）"""
        ai = DEFAULT_CONFIG.get("ai", {})
        return {
            "has_api_key": bool(ai.get("api_key")),
            "base_url": ai.get("base_url", "https://api.openai.com/v1"),
            "model": ai.get("model", "gpt-4o"),
            "temperature": ai.get("temperature", 0.7),
            "max_tokens": ai.get("max_tokens", 50000),
        }

    @app.get("/api/config/score-games")
    async def get_score_games():
        """两类有 score 定义的游戏列表（详见 src/utils/score_config.py）：
        - reward_as_score_games：episode_reward 即 score（不归一）
        - custom_score_games   ：归一化到 [0, 100]，需配合 info / step_count 计算
        前端 SCORE_GAMES 仅展示 reward 直传那批；自定义规则 score 由后端 eval 计算后落盘。
        """
        return get_score_game_ids()

    @app.post("/api/game_tags")
    async def update_game_tags(body: dict):
        """更新游戏分类标签，立即写入 prompt/game_tags.json"""
        game_id = body.get("game_id")
        tags = body.get("tags")
        if not game_id:
            raise HTTPException(status_code=400, detail="game_id is required")

        all_tags = _load_game_tags()

        if tags:
            cleaned = {k: v for k, v in tags.items() if v is not None}
            if cleaned:
                all_tags[game_id] = cleaned
            else:
                all_tags.pop(game_id, None)
        else:
            all_tags.pop(game_id, None)

        _save_game_tags(all_tags)
        return {"ok": True, "game_id": game_id, "tags": all_tags.get(game_id)}

    @app.post("/api/session/create")
    async def create_session(body: dict):
        game_name = body.get("game_name", "Taxi-v3")
        config = body.get("config", {})
        interaction_mode = body.get("mode", "human")

        session_id = str(uuid.uuid4())[:8]
        session = GameSession(session_id, game_name, config)
        session.interaction_mode = interaction_mode
        session.evaluation_config_name = body.get("evaluation_config_name")
        reset_human_evaluation_tracking(session)

        try:
            initial_frame = await session.initialize()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        sessions[session_id] = session
        return {
            "session_id": session_id,
            "frame": initial_frame,
            "action_info": session.action_info,
            "state": session.get_state_info(),
        }

    @app.post("/api/session/{session_id}/reset")
    async def reset_session(session_id: str):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        async with get_human_evaluation_lock(session):
            frame = await session.reset()
            reset_human_evaluation_tracking(session)
        return {"frame": frame, "state": session.get_state_info()}

    @app.get("/api/session/{session_id}/frame/{index}")
    async def get_frame(session_id: str, index: int):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        b64 = session.get_frame_base64(index)
        if b64 is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        return {"frame": b64, "index": index, "total": len(session.frames)}

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        session = sessions.pop(session_id, None)
        if session:
            # 停止 AI
            ai = ai_players.pop(session_id, None)
            if ai:
                ai.stop()
            session.close()
        return {"ok": True}

    @app.post("/api/session/{session_id}/save-evaluation")
    async def save_evaluation(session_id: str):
        """保存 human/crossval 的轻量评测结果（actions.json + result.json）。"""
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if getattr(session, "interaction_mode", "human") not in ("human", "crossval"):
            raise HTTPException(status_code=400, detail="Only human evaluations can be saved")
        async with get_human_evaluation_lock(session):
            if not session.is_game_over:
                session.is_game_over = True
                session.last_info = dict(session.last_info or {})
                session.last_info.update({
                    "ending": "manual_end",
                    "pass": False,
                    "terminated": False,
                    "truncated": False,
                })
            try:
                path, already_saved = await save_human_evaluation_once_locked(
                    session, human_eval_root
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to save evaluation: {exc}")
        return {"ok": True, "path": str(path), "already_saved": already_saved}

    @app.post("/api/session/{session_id}/ai/upload-context")
    async def upload_context(session_id: str, files: List[UploadFile] = File(...)):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        ai = ai_players.setdefault(session_id, AIPlayer(session))
        group_frames = []

        for f in files:
            data = await f.read()
            if f.filename and f.filename.endswith(".zip"):
                # zip 包含多张图片
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for name in sorted(zf.namelist()):
                        if name.lower().endswith((".png", ".jpg", ".jpeg")):
                            img_data = zf.read(name)
                            img = Image.open(io.BytesIO(img_data)).convert("RGB")
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=85)
                            group_frames.append(base64.b64encode(buf.getvalue()).decode())
            else:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                group_frames.append(base64.b64encode(buf.getvalue()).decode())

        ai.add_context_group(group_frames)
        return {"uploaded": len(group_frames), "total_context_frames": len(ai.context_frames)}

    @app.post("/api/session/{session_id}/ai/upload-context-video")
    async def upload_context_video(
        session_id: str,
        video: UploadFile = File(...),
        fps: float = Form(1.0)
    ):
        """从上传的视频文件中提取帧，并添加到 AI context"""
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        ai = ai_players.setdefault(session_id, AIPlayer(session))

        # 保存视频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(await video.read())

        try:
            # 使用 video_encoder 提取帧
            from ai_backends.video_encoder import video_file_to_frames
            frame_bytes_list = video_file_to_frames(tmp_path, fps=fps)

            # 将帧转换为 base64 并作为一组添加到 AI context
            frames_b64 = []
            for jpeg_bytes in frame_bytes_list:
                frames_b64.append(base64.b64encode(jpeg_bytes).decode())
            ai.add_context_group(frames_b64)

            return {
                "frames_extracted": len(frame_bytes_list),
                "total_context_frames": len(ai.context_frames),
                "frames": frames_b64  # 返回给前端用于缓存
            }
        finally:
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()

    @app.post("/api/session/{session_id}/ai/clear-context")
    async def clear_context(session_id: str):
        """清空 AIPlayer 的所有 context"""
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        ai = ai_players.get(session_id)
        if ai:
            ai.clear_context()
        return {"ok": True}

    @app.post("/api/session/{session_id}/ai/upload-context-frames")
    async def upload_context_frames(session_id: str, body: dict):
        """直接添加 base64 帧数组为一个 context group（不清空已有 context）"""
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        ai = ai_players.setdefault(session_id, AIPlayer(session))
        frames = body.get("frames", [])
        ai.add_context_group(frames)
        return {"added": len(frames), "total_context_frames": len(ai.context_frames)}

    @app.post("/api/session/{session_id}/ai/configure")
    async def configure_ai(session_id: str, body: dict):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        ai = ai_players.setdefault(session_id, AIPlayer(session))
        ai.configure(
            api_key=body.get("api_key", ""),
            base_url=body.get("base_url", ""),
            model=body.get("model", ""),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            video_description=body.get("video_description"),
            game_rules=body.get("game_rules"),
            action_mode=body.get("action_mode"),
            frame_source=body.get("frame_source"),
            video_mode=body.get("video_mode"),
            frame_window=body.get("frame_window"),
            hide_reward=body.get("hide_reward"),
            provider=body.get("provider"),
        )
        return {"ok": True}

    # --- AI Config Save/Load ---
    AI_CONFIGS_DIR = Path(__file__).parent.parent.parent / "ai_configs"

    @app.get("/api/ai-configs")
    async def list_ai_configs():
        """列出所有已保存的 AI 配置，按游戏分组"""
        result = {}
        if not AI_CONFIGS_DIR.exists():
            return result
        for game_dir in sorted(AI_CONFIGS_DIR.iterdir()):
            if not game_dir.is_dir() or game_dir.name.startswith('.'):
                continue
            configs = []
            for cfg_dir in sorted(game_dir.iterdir()):
                if not cfg_dir.is_dir():
                    continue
                cfg_file = cfg_dir / "config.json"
                if not cfg_file.exists():
                    continue
                try:
                    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                frames_dir = cfg_dir / "frames"
                frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
                has_video = (cfg_dir / "video.mp4").exists()
                configs.append({
                    "name": cfg_dir.name,
                    "game_id": cfg.get("game_id", ""),
                    "frame_count": frame_count,
                    "has_video": has_video,
                    "created_at": cfg.get("created_at", ""),
                })
            if configs:
                game_id = configs[0]["game_id"] if configs else game_dir.name.replace("_", "/", 1)
                result[game_id] = configs
        return result

    @app.post("/api/ai-configs/save")
    async def save_ai_config(body: dict):
        """保存 AI 配置（超参数 + context 帧 + 生成视频）"""
        game_id = body.get("game_id")
        name = body.get("name")
        config = body.get("config", {})
        frames = body.get("frames", [])

        if not game_id or not name:
            raise HTTPException(status_code=400, detail="game_id and name are required")

        game_safe = game_id.replace("/", "_").replace(":","_")
        name_safe = name.replace("/", "_").replace("\\", "_").replace("..", "_").replace(":","_")
        cfg_dir = AI_CONFIGS_DIR / game_safe / name_safe

        if cfg_dir.exists():
            shutil.rmtree(cfg_dir)

        cfg_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "game_id": game_id,
            "name": name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **config,
            "context_frame_count": len(frames),
        }
        (cfg_dir / "config.json").write_text(
            json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if frames:
            frames_dir = cfg_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            for i, b64 in enumerate(frames):
                img_bytes = base64.b64decode(b64)
                (frames_dir / f"{i:04d}.jpg").write_bytes(img_bytes)

            try:
                from ai_backends.video_encoder import frames_to_mp4_base64
                video_b64 = frames_to_mp4_base64(frames, fps=1.0)
                video_bytes = base64.b64decode(video_b64)
                (cfg_dir / "video.mp4").write_bytes(video_bytes)
            except Exception as e:
                print(f"[AIConfig] Warning: failed to generate video: {e}")

        return {"ok": True, "path": str(cfg_dir.relative_to(AI_CONFIGS_DIR))}

    @app.get("/api/ai-configs/{game_safe}/{config_name}/config")
    async def get_ai_config(game_safe: str, config_name: str):
        """返回指定配置的 config.json 内容"""
        cfg_path = AI_CONFIGS_DIR / game_safe / config_name / "config.json"
        if not cfg_path.exists():
            raise HTTPException(status_code=404, detail="Config not found")
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    @app.get("/api/ai-configs/{game_safe}/{config_name}/frames")
    async def get_ai_config_frames(game_safe: str, config_name: str):
        """返回指定配置的所有帧（base64 列表）"""
        frames_dir = AI_CONFIGS_DIR / game_safe / config_name / "frames"
        if not frames_dir.exists():
            return {"frames": []}
        frames = []
        for f in sorted(frames_dir.glob("*.jpg")):
            frames.append(base64.b64encode(f.read_bytes()).decode("ascii"))
        return {"frames": frames}

    @app.get("/api/ai-configs/{game_safe}/{config_name}/video")
    async def get_ai_config_video(game_safe: str, config_name: str):
        """返回指定配置的视频文件"""
        video_path = AI_CONFIGS_DIR / game_safe / config_name / "video.mp4"
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(video_path, media_type="video/mp4")

    @app.get("/api/ai-configs/{game_safe}/{config_name}/segments")
    async def get_ai_config_segments(game_safe: str, config_name: str):
        """返回指定配置的 context segments 元信息列表（与 AI 实际看到的多段视频对齐）

        读取 config.json 的 context_items，每段返回 {index, label, frame_count}。
        没有 context_items 但有 frames 时回退为单段。
        """
        cfg_dir = AI_CONFIGS_DIR / game_safe / config_name
        cfg_path = cfg_dir / "config.json"
        if not cfg_path.exists():
            raise HTTPException(status_code=404, detail="Config not found")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        items = cfg.get("context_items") or []
        frames_dir = cfg_dir / "frames"
        total_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
        segments = []
        if items:
            for i, it in enumerate(items):
                segments.append({
                    "index": i,
                    "label": it.get("label", f"Segment {i+1}"),
                    "frame_count": it.get("frame_count", 0),
                })
        elif total_frames > 0:
            segments.append({"index": 0, "label": "Demo", "frame_count": total_frames})
        return {"segments": segments, "total_frames": total_frames}

    @app.get("/api/ai-configs/{game_safe}/{config_name}/segment/{seg_index}")
    async def get_ai_config_segment_video(game_safe: str, config_name: str, seg_index: int):
        """按需切段编码：从 frames 目录里取第 seg_index 段对应的帧，编码成 mp4 返回。

        与 AI 运行时 ai_player.add_context_group 的边界完全一致。
        """
        cfg_dir = AI_CONFIGS_DIR / game_safe / config_name
        cfg_path = cfg_dir / "config.json"
        frames_dir = cfg_dir / "frames"
        if not cfg_path.exists() or not frames_dir.exists():
            raise HTTPException(status_code=404, detail="Config or frames not found")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        items = cfg.get("context_items") or []

        all_frame_files = sorted(frames_dir.glob("*.jpg"))
        if items:
            offset = 0
            target_files = None
            for i, it in enumerate(items):
                fc = it.get("frame_count", 0)
                if i == seg_index:
                    target_files = all_frame_files[offset:offset + fc]
                    break
                offset += fc
            if target_files is None:
                raise HTTPException(status_code=404, detail=f"Segment {seg_index} not found")
        else:
            if seg_index != 0:
                raise HTTPException(status_code=404, detail="No segments")
            target_files = all_frame_files

        if not target_files:
            raise HTTPException(status_code=404, detail="Segment has no frames")

        frames_b64 = [base64.b64encode(p.read_bytes()).decode("ascii") for p in target_files]

        from ai_backends.video_encoder import frames_to_mp4_base64
        try:
            video_b64 = frames_to_mp4_base64(frames_b64, fps=1.0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Encode failed: {e}")
        from fastapi.responses import Response
        return Response(content=base64.b64decode(video_b64), media_type="video/mp4")

    # --- Action Sequence Save/Load ---
    ACTION_SEQ_DIR = Path(__file__).parent.parent.parent / "action_sequences"

    @app.get("/api/action-sequences")
    async def list_action_sequences():
        """列出所有已保存的动作序列，按游戏分组"""
        result = {}
        if not ACTION_SEQ_DIR.exists():
            return result
        for game_dir in sorted(ACTION_SEQ_DIR.iterdir()):
            if not game_dir.is_dir() or game_dir.name.startswith('.'):
                continue
            sequences = []
            for seq_dir in sorted(game_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                seq_file = seq_dir / "sequence.json"
                if not seq_file.exists():
                    continue
                try:
                    seq = json.loads(seq_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                sequences.append({
                    "name": seq_dir.name,
                    "game_id": seq.get("game_id", ""),
                    "action_count": len(seq.get("actions", [])),
                    "created_at": seq.get("created_at", ""),
                    "description": seq.get("description", ""),
                })
            if sequences:
                game_id = sequences[0]["game_id"] if sequences else game_dir.name.replace("_", "/", 1)
                result[game_id] = sequences
        return result

    @app.post("/api/action-sequences/save")
    async def save_action_sequence(body: dict):
        """保存动作序列"""
        game_id = body.get("game_id")
        name = body.get("name")
        actions = body.get("actions", [])
        action_info = body.get("action_info", {})

        if not game_id or not name:
            raise HTTPException(status_code=400, detail="game_id and name are required")
        if not actions:
            raise HTTPException(status_code=400, detail="actions list is empty")

        game_safe = game_id.replace("/", "_")
        name_safe = name.replace("/", "_").replace("\\", "_").replace("..", "_")
        seq_dir = ACTION_SEQ_DIR / game_safe / name_safe

        if seq_dir.exists():
            shutil.rmtree(seq_dir)
        seq_dir.mkdir(parents=True, exist_ok=True)

        # 自动生成描述（前 10 个动作名）
        desc_parts = []
        for a in actions[:10]:
            act_name = action_info.get(str(a), str(a))
            desc_parts.append(act_name)
        description = f"{len(actions)} actions: {', '.join(desc_parts)}"
        if len(actions) > 10:
            description += ", ..."

        seq_data = {
            "game_id": game_id,
            "name": name,
            "actions": actions,
            "action_info": action_info,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "description": description,
        }
        (seq_dir / "sequence.json").write_text(
            json.dumps(seq_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {"ok": True, "path": str(seq_dir.relative_to(ACTION_SEQ_DIR))}

    @app.delete("/api/action-sequences/{game_safe}/{seq_name}")
    async def delete_action_sequence(game_safe: str, seq_name: str):
        """删除动作序列"""
        seq_dir = ACTION_SEQ_DIR / game_safe / seq_name
        if not seq_dir.exists():
            raise HTTPException(status_code=404, detail="Sequence not found")
        shutil.rmtree(seq_dir)
        return {"ok": True}

    @app.get("/api/action-sequences/{game_safe}/{seq_name}/sequence")
    async def get_action_sequence(game_safe: str, seq_name: str):
        """获取完整序列数据"""
        seq_path = ACTION_SEQ_DIR / game_safe / seq_name / "sequence.json"
        if not seq_path.exists():
            raise HTTPException(status_code=404, detail="Sequence not found")
        return json.loads(seq_path.read_text(encoding="utf-8"))

    def _do_save(folder, frames_jpeg, action_hist, reward_hist,
                 action_info, game_name, config, seed, episode_reward,
                 ending_type, pass_type, chat_html, timestamp, session_for_trajectory=None,seq_count=0,
                 extra_meta=None):
        """保存帧、字幕、视频、动作序列到指定目录（save_session 的核心逻辑）"""
        frames_dir = folder / "frames"
        subbed_dir = folder / "subbed"
        subbed_nl_dir = folder / "subbed_nl"
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

        # 字幕文本
        captions = [_make_caption(i) for i in range(len(frames_jpeg))]
        (folder / "caption.txt").write_text("\n".join(captions), encoding="utf-8")
        captions_nl = [_make_caption_nl(i) for i in range(len(frames_jpeg))]
        (folder / "caption_nl.txt").write_text("\n".join(captions_nl), encoding="utf-8")

        # 动作序列 JSON
        actions_data = {
            "game_name": game_name,
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "seed": seed,
            "total_steps": len(action_hist),
            "total_frames": len(frames_jpeg),
            "episode_reward": episode_reward,
            "actions": list(action_hist),
            "rewards": list(reward_hist),
            "video_fps": 1.0,
            "ending": ending_type,
            "pass": pass_type
        }
        if extra_meta:
            actions_data.update(extra_meta)
        (folder / "actions.json").write_text(
            json.dumps(actions_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # 生成视频
        video_name = save_video(pil_subbed, folder / "video.mp4", 1.0)
        video_nl_name = save_video(pil_subbed_nl, folder / "video_nl.mp4", 1.0)
        video_original_name = save_video(pil_frames, folder / "video_original.mp4", 1.0)

        # 轨迹摘要图
        if session_for_trajectory:
            trajectory_img = generate_trajectory_image(session_for_trajectory,seq_count)
            if trajectory_img is not None:
                if ending_type:
                    trajectory_img = add_ending_label(trajectory_img, ending_type)
                trajectory_img.save(folder / "trajectory.png")

        # 聊天记录 HTML
        if chat_html:
            chat_page = f"""<!DOCTYPE html>
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
.chat-media-blocks{{display:flex;flex-direction:column;gap:10px;margin-bottom:10px}}
.chat-media-block{{display:flex;flex-direction:column;gap:4px}}
.chat-media-label{{font-size:10px;color:#888}}
.chat-frames{{display:flex;flex-wrap:wrap;gap:6px}}
.chat-thumb{{width:80px;height:80px;object-fit:cover;border-radius:4px;border:1px solid #444}}
.chat-thumb-ctx{{border-color:#555;opacity:.55}}
video{{max-width:100%;border-radius:4px}}
</style>
</head>
<body>
<h1>Chat Log — {game_name} &nbsp; {timestamp}</h1>
<div id="chat-messages">{chat_html}</div>
</body>
</html>"""
            (folder / "chat.html").write_text(chat_page, encoding="utf-8")

        return {
            "frames_saved": len(frames_jpeg),
            "actions_saved": len(action_hist),
            "video": video_name,
            "video_nl": video_nl_name,
            "video_original": video_original_name,
        }

    @app.post("/api/session/{session_id}/save")
    async def save_session(session_id: str, body: dict = None):
        """保存当前操作序列和帧（human_operation/ / ai_operation/ / crossval_operation/，取决于 mode 参数）"""
        if body is None:
            body = {}
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        mode = body.get("mode", "human")  # "human" | "ai" | "crossval"
        if mode == "crossval":
            raise HTTPException(
                status_code=410,
                detail="Cross-Val sessions are saved automatically as actions.json + result.json",
            )
        ending_type = body.get("ending_type") or session.last_info.get("ending")  # demo_end | victory | game_over | time_out | draw
        pass_type=body.get("pass_type", session.last_info.get("pass"))
        op_dir = get_op_dir(mode)
        # 交叉验证模式：把 source_ai_config 写入 metadata 以便后续追溯
        extra_meta = None
        if mode == "crossval":
            src = body.get("source_ai_config") or {}
            extra_meta = {"mode": "crossval", "source_ai_config": src}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_short = session.game_name.replace("ALE/", "").replace("/", "_").replace(":","_")
        custom_name = body.get("custom_name", "").strip()
        folder_name = custom_name if custom_name else f"{game_short}_{timestamp}"
        folder_name=folder_name.replace(":","_")
        chat_html = body.get("chat_html", "")
        seed = session.config.get("_seed")

        seq_count = getattr(session, '_seq_action_count', 0)

        def _run_save():
            if seq_count > 0:
                # 有预加载序列：保存两份
                # 主版本：仅 AI 操作（不含序列前缀）
                ai_frames = session.frames[seq_count:]  # frame[seq_count] = 序列结束后的起始状态
                ai_actions = session.action_history[seq_count:]
                ai_rewards = session.reward_history[seq_count:]
                folder = op_dir / folder_name
                result = _do_save(
                    folder, ai_frames, ai_actions, ai_rewards,
                    session.action_info, session.game_name, session.config,
                    seed, session.episode_reward, ending_type, pass_type,chat_html,
                    timestamp, session_for_trajectory=session,seq_count=seq_count,
                    extra_meta=extra_meta,
                )
                result["path"] = str(folder)

                # 副本：含序列的完整版本
                folder_full = op_dir / (folder_name + "_full")
                full_reward = sum(session.reward_history)
                _do_save(
                    folder_full, session.frames, session.action_history,
                    session.reward_history, session.action_info, session.game_name,
                    session.config, seed, full_reward, ending_type, pass_type, chat_html,
                    timestamp, session_for_trajectory=session,
                    extra_meta=extra_meta,
                )
                result["full_path"] = str(folder_full)
            else:
                # 无序列：正常保存
                folder = op_dir / folder_name
                result = _do_save(
                    folder, session.frames, session.action_history,
                    session.reward_history, session.action_info, session.game_name,
                    session.config, seed, session.episode_reward, ending_type, pass_type,
                    chat_html, timestamp, session_for_trajectory=session,
                    extra_meta=extra_meta,
                )
                result["path"] = str(folder)
            return result

        result = await asyncio.to_thread(_run_save)
        return result

    @app.post("/api/session/{session_id}/export-optimal")
    async def export_optimal_solution(session_id: str):
        """计算 BFS 最优解并导出视频到 human_operation/"""
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        game_name = session.game_name
        seed = session.config.get("_seed")
        config = {k: v for k, v in session.config.items() if not k.startswith("_")}

        def _compute_and_save():
            from src.env.bfs_solver import compute_and_execute_optimal

            result = compute_and_execute_optimal(game_name, config, seed)
            if result is None:
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            game_short = game_name.replace("ALE/", "").replace("/", "_")
            folder_name = f"{game_short}_optimal_{timestamp}"
            folder = human_op_dir / folder_name

            save_result = _do_save(
                folder, result["frames_jpeg"], result["action_history"],
                result["reward_history"], result["action_info"], game_name,
                config, seed, result["episode_reward"], result["ending_type"],
                result.get("pass_type"), chat_html="", timestamp=timestamp,
            )
            save_result["path"] = str(folder)
            save_result["folder_name"] = folder_name
            save_result["total_steps"] = result["total_steps"]
            save_result["episode_reward"] = result["episode_reward"]
            save_result["ending_type"] = result["ending_type"]
            save_result["pass_type"] = result.get("pass_type")
            return save_result

        result = await asyncio.to_thread(_compute_and_save)
        if result is None:
            raise HTTPException(status_code=422, detail="BFS 无法找到最优解（可能环境无解）")
        return result

    @app.get("/api/human-operations")
    async def list_human_operations():
        """扫描 human_operation/ + ai_operation/ + crossval_operation/ 下的已保存操作，供 context 选择"""
        results = []
        for source, op_dir in [("human", human_op_dir), ("ai", ai_op_dir), ("crossval", crossval_op_dir)]:
            if not op_dir.exists():
                continue
            for d in sorted(op_dir.iterdir()):
                if not d.is_dir():
                    continue
                frames_dir = d / "frames"
                actions_file = d / "actions.json"
                frame_count = len(list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
                meta = {}
                if actions_file.exists():
                    try:
                        with open(actions_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                results.append({
                    "name": d.name,
                    "source": source,
                    "frame_count": frame_count,
                    "game_name": meta.get("game_name", ""),
                    "total_steps": meta.get("total_steps", 0),
                    "episode_reward": meta.get("episode_reward", 0),
                    "source_ai_config": meta.get("source_ai_config"),
                })
        return results

    @app.get("/api/human-operations/{folder_path:path}/frames")
    async def get_operation_frames(folder_path: str):
        """获取某个操作文件夹的帧（用于预览）。folder_path 格式: source/name 或 name"""
        source, name = (folder_path.split("/", 1) + [None])[:2] if "/" in folder_path else ("human", folder_path)
        frames_dir = get_op_dir(source) / name / "frames"
        if not frames_dir.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        frames_b64 = []
        for img_path in sorted(frames_dir.glob("*.png")):
            img = Image.open(img_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode())

        return {"frames": frames_b64, "count": len(frames_b64)}

    @app.get("/api/operations/{folder_path:path}/frames")
    async def get_operation_frames_with_source(folder_path: str, source: str = "original"):
        """
        获取某个操作文件夹的帧（用于预览）

        参数：
        - folder_path: 文件夹路径（如 "human/Taxi-v3_xxx" 或 "Taxi-v3_xxx"）
        - source: 帧源（original/subbed/subbed_nl，默认 original）

        返回：
        - frames: base64 帧列表
        - count: 帧数量
        """
        src, name = (folder_path.split("/", 1) + [None])[:2] if "/" in folder_path else ("human", folder_path)
        op_dir = get_op_dir(src)

        frames_b64 = load_frames_from_source(op_dir, name, source)

        return {"frames": frames_b64, "count": len(frames_b64)}

    @app.post("/api/session/{session_id}/operations/{folder_path:path}/load-video")
    async def load_operation_video(session_id: str, folder_path: str, body: dict):
        """
        从 saved operation 加载或生成视频

        参数：
        - session_id: 会话 ID
        - folder_path: 文件夹路径（如 "human/Taxi-v3_bd9f96ab" 或 "Taxi-v3_bd9f96ab"）
        - body.frame_source: 帧源（original/subbed/subbed_nl）
        - body.action_mode: 动作模式（numeric/natural_language）- 用于选择视频文件
        - body.fps: 帧率（用于重新生成）
        - body.force_regenerate: 是否强制从图片重新生成

        返回：
        - video_path: 视频文件路径（相对于 op_dir）
        - video_url: 视频文件 URL
        - frames_b64: base64 帧数据（用于 AI）
        """
        session = sessions.get(session_id)
        # session 不存在时不报错，只跳过 AIPlayer 加载

        source, name = (folder_path.split("/", 1) + [None])[:2] if "/" in folder_path else ("human", folder_path)
        op_dir = get_op_dir(source)
        folder = op_dir / name

        if not folder.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        frame_source = body.get("frame_source", "subbed_nl")
        action_mode = body.get("action_mode", "natural_language")
        fps = float(body.get("fps", 1.0))
        force_regenerate = body.get("force_regenerate", False)

        # 根据 action_mode 选择视频文件
        video_filename = "video_nl.mp4" if action_mode == "natural_language" else "video.mp4"
        video_path = folder / video_filename

        # 如果视频文件存在且不强制重新生成，直接使用
        if video_path.exists() and not force_regenerate:
            # 从对应的图片文件夹加载帧（用于 AI）
            frames_b64 = load_frames_from_source(op_dir, name, frame_source)

            # 加载到 AIPlayer（如果有活跃会话）
            if session:
                ai = ai_players.setdefault(session_id, AIPlayer(session))
                ai.clear_context()
                ai.add_context_group(frames_b64)

            return {
                "video_path": str(video_path.relative_to(op_dir.parent)),
                "video_url": f"/api/video/{source}/{name}/{video_filename}",
                "frames_b64": frames_b64,
                "count": len(frames_b64),
                "fps": fps,
                "generated": False
            }

        # 否则从图片文件夹重新生成视频
        from ai_backends.video_encoder import frames_to_mp4_base64

        # 加载帧
        frames_b64 = load_frames_from_source(op_dir, name, frame_source)

        if not frames_b64:
            raise HTTPException(status_code=404, detail="No frames found")

        # 生成视频（保存到临时文件或覆盖原文件）
        video_b64 = frames_to_mp4_base64(frames_b64, fps=fps)
        video_bytes = base64.b64decode(video_b64)

        # 保存到临时文件
        tmp_filename = f"tmp_video_{frame_source}_{fps}.mp4"
        tmp_video_path = folder / tmp_filename
        tmp_video_path.write_bytes(video_bytes)

        # 加载到 AIPlayer（如果有活跃会话）
        if session:
            ai = ai_players.setdefault(session_id, AIPlayer(session))
            ai.clear_context()
            ai.add_context_group(frames_b64)

        return {
            "video_path": str(tmp_video_path.relative_to(op_dir.parent)),
            "video_url": f"/api/video/{source}/{name}/{tmp_filename}",
            "frames_b64": frames_b64,
            "count": len(frames_b64),
            "fps": fps,
            "generated": True
        }

    @app.post("/api/session/{session_id}/ai/load-context-folder")
    async def load_context_from_folder(session_id: str, body: dict):
        """将已预览的 context 帧加载到 AIPlayer"""
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        folder_name = body.get("folder")
        if not folder_name:
            raise HTTPException(status_code=400, detail="Missing folder name")

        source, name = (folder_name.split("/", 1) + [None])[:2] if "/" in folder_name else ("human", folder_name)
        op_dir = get_op_dir(source)

        ai = ai_players.setdefault(session_id, AIPlayer(session))
        append = body.get("append", False)
        if not append:
            ai.clear_context()

        # 使用 AI 当前配置的 frame_source 加载 context 帧（字幕仅作用于 demo，实时游戏帧始终原始）
        frames_b64 = load_frames_from_source(op_dir, name, ai.frame_source)

        # 直接加载对应的视频文件（避免运行时重编码）
        video_name_map = {"original": "video_original.mp4", "subbed": "video.mp4", "subbed_nl": "video_nl.mp4"}
        video_file = op_dir / name / video_name_map.get(ai.frame_source, "video.mp4")
        video_b64 = None
        if video_file.exists():
            import base64
            video_b64 = base64.b64encode(video_file.read_bytes()).decode()
            print(f"[LoadContext] Loaded video: {video_file.name} ({len(video_b64)} bytes b64)")

        ai.add_context_group(frames_b64, video_b64=video_b64)

        return {"loaded": len(frames_b64), "total_context_frames": len(ai.context_frames)}

    # 测试端点
    @app.get("/api/test-video")
    async def test_video():
        return {"status": "ok", "message": "Video endpoint is working"}

    # 视频文件服务端点（替代 StaticFiles）
    @app.get("/api/video/{source}/{folder_name}/{filename}")
    async def get_video_file(source: str, folder_name: str, filename: str):
        """
        直接返回视频文件
        source: 'human' 或 'ai'
        folder_name: 文件夹名称，如 'Taxi-v3_20260212_201252'
        filename: 视频文件名，如 'video_nl.mp4'
        """
        print(f"[DEBUG] Video request: source={source}, folder={folder_name}, file={filename}")

        if source == "human":
            base_dir = Path(__file__).parent.parent.parent / "human_operation"
        elif source == "ai":
            base_dir = Path(__file__).parent.parent.parent / "ai_operation"
        else:
            raise HTTPException(status_code=400, detail="Invalid source")

        video_path = base_dir / folder_name / filename
        print(f"[DEBUG] Video path: {video_path}, exists: {video_path.exists()}")

        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")

        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=filename
        )
