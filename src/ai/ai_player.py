"""AI 玩家管理"""
import asyncio
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from src.core import GameSession, get_prompt_manager
from src.utils import DEFAULT_CONFIG


class ModelDryRun(Exception):
    pass


class EmptySSECompletionError(RuntimeError):
    """OpenAI-compatible relay returned only an SSE end marker."""


def _content_part_to_text(value) -> str:
    """Extract text from either Chat Completions string or typed content parts."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    parts = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _parse_openai_sse_string(raw: str) -> tuple[str, Optional[dict], Optional[str]]:
    """Parse an SSE response leaked through an OpenAI-compatible SDK.

    Some relays occasionally return the raw SSE body even for a non-streaming
    request. Preserve ordinary strings, but merge JSON ``data:`` chunks and
    discard the terminal ``[DONE]`` marker.
    """
    stripped = raw.strip()
    if stripped == "[DONE]":
        raise EmptySSECompletionError(
            "OpenAI-compatible API returned only the SSE [DONE] marker"
        )

    payloads = []
    saw_sse_data = False
    for line in stripped.splitlines():
        candidate = line.lstrip()
        if candidate.startswith("data:"):
            saw_sse_data = True
            payloads.append(candidate[len("data:"):].strip())
        elif (
            not candidate
            or candidate.startswith(":")
            or candidate.startswith(("event:", "id:", "retry:"))
        ):
            continue
        else:
            # This is an ordinary model string which merely happens to contain
            # something resembling SSE on another line.
            return raw, None, None

    if not saw_sse_data:
        return raw, None, None

    json_payloads = [
        payload for payload in payloads
        if payload and payload.upper() != "[DONE]"
    ]
    if not json_payloads:
        raise EmptySSECompletionError(
            "OpenAI-compatible API returned only the SSE data: [DONE] marker"
        )

    chunks = []
    for payload in json_payloads:
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            # Do not reinterpret a non-JSON model response as a stream.
            return raw, None, None
        if isinstance(chunk, dict):
            chunks.append(chunk)

    content_parts = []
    thinking_parts = []
    merged_usage = {}
    for chunk in chunks:
        error = chunk.get("error")
        if error:
            raise RuntimeError(f"OpenAI-compatible SSE error: {error}")

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            merged_usage.update(
                {key: value for key, value in usage.items() if value is not None}
            )

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            source = choice.get("delta")
            if not isinstance(source, dict):
                source = choice.get("message")
            if not isinstance(source, dict):
                source = choice

            content = _content_part_to_text(source.get("content"))
            if content:
                content_parts.append(content)

            thinking = (
                source.get("reasoning_content")
                or source.get("reasoning")
                or source.get("thinking")
            )
            thinking_text = _content_part_to_text(thinking)
            if thinking_text:
                thinking_parts.append(thinking_text)

    return (
        "".join(content_parts),
        merged_usage or None,
        "".join(thinking_parts) or None,
    )


DEFAULT_VIDEO_DESCRIPTION = "These frames show example gameplay to help you understand the game."


def _dump_model_messages(messages: list, label: str):
    if not os.environ.get("VCL_DUMP_MODEL_MESSAGES") and not os.environ.get("VCL_DRY_RUN_MODEL"):
        return

    out_dir = Path(os.environ.get("VCL_MODEL_DUMP_DIR", "tmp/model_message_dumps"))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"{timestamp}_{label}.json"
    html_path = out_dir / f"{timestamp}_{label}.html"

    dumped = []
    html_blocks = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Model message dump</title>",
        "<style>body{font-family:Arial,sans-serif;background:#111;color:#ddd;padding:20px} .msg{border:1px solid #444;border-radius:8px;padding:16px;margin:16px 0}.part{border-left:3px solid #64b5f6;padding:10px;margin:10px 0;background:#1b1b1b}pre{white-space:pre-wrap;line-height:1.45}video,img{max-width:640px;max-height:420px;display:block;border:1px solid #555;border-radius:4px}.meta{color:#aaa;font-size:12px;margin-bottom:6px}</style>",
        "</head><body><h1>Model message dump</h1>",
    ]
    for mi, message in enumerate(messages):
        content = message.get("content")
        entry = {"index": mi, "role": message.get("role"), "content": []}
        html_blocks.append(f"<section class='msg'><h2>Message {mi}: role={message.get('role')}</h2>")
        if isinstance(content, str):
            entry["content"].append({"index": 0, "type": "text", "text": content})
            html_blocks.append(f"<div class='part'><div class='meta'>part 0 · text</div><pre>{html.escape(content)}</pre></div>")
        elif isinstance(content, list):
            for ci, item in enumerate(content):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    entry["content"].append({"index": ci, "type": "text", "text": text})
                    html_blocks.append(f"<div class='part'><div class='meta'>part {ci} · text</div><pre>{html.escape(text)}</pre></div>")
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    media_type = "video" if url.startswith("data:video/") else "image" if url.startswith("data:image/") else "image_url"
                    entry["content"].append({
                        "index": ci,
                        "type": media_type,
                        "prefix": url[:40],
                        "base64_chars": len(url.split(",", 1)[1]) if "," in url else 0,
                    })
                    html_blocks.append(f"<div class='part'><div class='meta'>part {ci} · {media_type} · {len(url)} chars</div>")
                    if media_type == "video":
                        html_blocks.append(f"<video controls src='{url}'></video>")
                    elif media_type == "image":
                        html_blocks.append(f"<img src='{url}'>")
                    else:
                        html_blocks.append(f"<pre>{html.escape(url[:200])}</pre>")
                    html_blocks.append("</div>")
                else:
                    entry["content"].append({"index": ci, "type": item.get("type", "unknown"), "keys": list(item.keys())})
                    html_blocks.append(f"<div class='part'><div class='meta'>part {ci} · unknown</div><pre>{html.escape(str(item))}</pre></div>")
        else:
            entry["content"].append({"index": 0, "type": type(content).__name__, "value": str(content)})
            html_blocks.append(f"<div class='part'><div class='meta'>part 0 · {type(content).__name__}</div><pre>{html.escape(str(content))}</pre></div>")
        html_blocks.append("</section>")
        dumped.append(entry)

    path.write_text(json.dumps(dumped, indent=2, ensure_ascii=False), encoding="utf-8")
    html_blocks.append("</body></html>")
    html_path.write_text("\n".join(html_blocks), encoding="utf-8")
    print(f"[ModelDump] Wrote {path}")
    print(f"[ModelDump] Wrote {html_path}")


class AIPlayer:
    """管理 AI 模型交互 — 逻辑与 model_gen/multi_turn_game.py 完全一致"""

    def __init__(self, session: GameSession):
        self.session = session
        # API 配置：优先从 config.json 读取默认值
        ai_defaults = DEFAULT_CONFIG.get("ai", {})
        self.api_key: str = ai_defaults.get("api_key", "")
        self.base_url: str = ai_defaults.get("base_url", "https://api.openai.com/v1")
        self.model: str = ai_defaults.get("model", "gpt-4o")
        self.temperature: float = ai_defaults.get("temperature", 0.7)
        self.max_tokens: int = ai_defaults.get("max_tokens", 50000)
        # 仅由根 config.json 控制；空字符串表示不向兼容接口发送该字段。
        self.reasoning_effort: str = str(
            ai_defaults.get("reasoning_effort", "") or ""
        ).strip()
        # OpenRouter provider routing 配置；例如 only=["alibaba"] 可锁定渠道。
        raw_provider = ai_defaults.get("provider")
        self.provider: dict | None = (
            dict(raw_provider) if isinstance(raw_provider, dict) else None
        )
        # context 组 (每组来自一次 load/upload 操作)
        # 每个 group: {"frames": List[str], "video_b64": str|None}
        self.context_groups: List[dict] = []
        # 视频描述 & 游戏规则 (对应原 model_gen 的 video_description / game_rules)
        self.video_description: str = DEFAULT_VIDEO_DESCRIPTION
        self.game_rules: str = ""
        # 动作模式: "natural_language"（默认）或 "numeric"
        self.action_mode: str = DEFAULT_CONFIG.get("default_action_mode", "natural_language")
        # 帧源选择: "original"（默认）, "subbed", "subbed_nl"
        # 注意：仅影响 context/demo 帧的字幕；实时游戏帧始终是原始帧（不加字幕）
        self.frame_source: str = ai_defaults.get("frame_source", "original")
        # 视频模式: 是否将帧序列编码为视频发送给 AI
        self.video_mode: bool = ai_defaults.get("video_mode", True)
        # API 模式: "gemini_native" | "openrouter" | "kimi" | "openai" | "openai_responses"
        #   - gemini_native : 走 Gemini generateContent (inline_data)
        #   - openrouter    : OpenAI 兼容 + OpenRouter 的 video_url 扩展
        #   - kimi          : Moonshot Chat Completions + 文件上传视频
        #   - openai        : 纯 OpenAI 兼容 (image_url only, 无视频)
        #   - openai_responses: OpenAI Responses API (input_image, 非流式)
        self.api_mode: str = ai_defaults.get("api_mode", "gemini_native")
        self._last_assistant_message: dict | None = None
        # 帧窗口: 0 = 完整历史, N > 0 = 滑动窗口（最近 N 轮对话）
        self.frame_window: int = 0
        # 隐藏 reward 信息（不让 AI 看到奖励数值）
        self.hide_reward: bool = True
        # 控制
        self.is_running: bool = False
        self._task: Optional[asyncio.Task] = None
        # 用户重试机制（通过 websocket_handler 中转消息）
        self._retry_event: asyncio.Event = asyncio.Event()
        self._retry_choice: Optional[str] = None  # "retry" or "stop"
        self.hide_step=False
        self.detailed_game_rules=ai_defaults.get("detailed_game_rules",False)

    def configure(self, api_key: str = "", base_url: str = "", model: str = "",
                  temperature: float = None, max_tokens: int = None,
                  video_description: str = None, game_rules: str = None,
                  action_mode: str = None, frame_source: str = None,
                  video_mode: bool = None, api_mode: str = None,
                  frame_window: int = None, hide_reward: bool = None,
                  provider: dict | None = None, hide_step=None,
                  detailed_game_rules=None, reasoning_effort: str = None,
                  **_kwargs):
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url
        if model:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if video_description is not None:
            self.video_description = video_description or DEFAULT_VIDEO_DESCRIPTION
        if game_rules is not None:
            self.game_rules = game_rules
        if action_mode in ("numeric", "natural_language"):
            self.action_mode = action_mode
        if frame_source in ("original", "subbed", "subbed_nl"):
            self.frame_source = frame_source
        if video_mode is not None:
            self.video_mode = video_mode
        if api_mode in ("gemini_native", "openrouter", "kimi", "openai", "openai_responses"):
            self.api_mode = api_mode
        if frame_window is not None:
            self.frame_window = frame_window
        if hide_reward is not None:
            self.hide_reward = hide_reward
        if provider is not None:
            self.provider = dict(provider) if isinstance(provider, dict) else None
        if reasoning_effort is not None:
            self.reasoning_effort = str(reasoning_effort or "").strip()
        if hide_step is not None:
            self.hide_step=hide_step
        if detailed_game_rules is not None:
            self.detailed_game_rules=detailed_game_rules

    @property
    def context_frames(self) -> List[str]:
        """所有 context 帧（扁平化，向后兼容）"""
        return [f for g in self.context_groups for f in g["frames"]]

    def add_context_group(self, frames: List[str], video_b64: str = None):
        """添加一组 context（来自一次 load/upload 操作）"""
        if frames:
            self.context_groups.append({"frames": frames, "video_b64": video_b64})

    def clear_context(self):
        """清空所有 context 组"""
        self.context_groups.clear()

    # ------------------------------------------------------------------
    # Prompt 构建 — 支持 numeric / natural_language 两种模式
    # ------------------------------------------------------------------

    def _create_initial_prompt(self) -> str:
        """通过 PromptManager 构建 initial prompt，支持两种模式"""
        return self._create_initial_prompt_section("full")

    def _create_initial_prompt_section(self, section: str) -> str:
        if self.detailed_game_rules:
            rules = get_prompt_manager().get_game_rules(self.session.game_name,self.detailed_game_rules)
        else:
            rules = self.game_rules or get_prompt_manager().get_game_rules(self.session.game_name,self.detailed_game_rules)
        prompt = get_prompt_manager().build_initial_prompt(
            mode=self.action_mode,
            game_name=self.session.game_name,
            action_info=self.session.action_info,
            game_rules=rules,
            video_description=self.video_description,
            context_frames=self.context_frames,
            max_steps=self.session.config.get("_max_steps"),
            section=section,
        )
        print(f"[AIPlayer] _create_initial_prompt section={section} mode={self.action_mode}")
        print(f"[AIPlayer] Prompt preview (first 500 chars):\n{prompt[:500]}")
        return prompt

    def _create_followup_prompt(
        self,
        step: int,
        last_action: int,
        action_meaning: str,
        reward: float,
        cumulative_reward: float,
        is_terminated: bool,
        is_truncated: bool,
        info: dict,
    ) -> str:
        return self._create_followup_prompt_section(
            "full", step, last_action, action_meaning, reward,
            cumulative_reward, is_terminated, is_truncated, info
        )

    def _create_followup_prompt_section(
        self,
        section: str,
        step: int,
        last_action: int,
        action_meaning: str,
        reward: float,
        cumulative_reward: float,
        is_terminated: bool,
        is_truncated: bool,
        info: dict,
    ) -> str:
        prompt = get_prompt_manager().build_followup_prompt(
            mode=self.action_mode,
            step=step,
            last_action=last_action,
            action_meaning=action_meaning,
            reward=reward,
            cumulative_reward=cumulative_reward,
            is_terminated=is_terminated,
            is_truncated=is_truncated,
            info=info,
            hide_reward=self.hide_reward,
            hide_step=self.hide_step,
            section=section,
        )
        print(f"[AIPlayer] _create_followup_prompt section={section} mode={self.action_mode} round_step={step+1}")
        print(f"[AIPlayer] Prompt preview (first 300 chars):\n{prompt[:300]}")
        return prompt

    # ------------------------------------------------------------------
    # API 调用
    # ------------------------------------------------------------------

    def _build_message_content(self, frames_b64: List[str], text: str) -> list:
        """
        构建消息内容（支持图像序列模式和视频模式）

        Args:
            frames_b64: base64 编码的帧列表
            text: 文本内容

        Returns:
            OpenAI-compatible message content
        """
        if self.video_mode and len(frames_b64) > 1:
            # 视频模式：多帧编码为视频（单帧直接走图片模式）
            from ai_backends.video_encoder import frames_to_mp4_base64
            video_b64 = frames_to_mp4_base64(frames_b64, fps=1.0)
            return [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                },
                {"type": "text", "text": text},
            ]
        else:
            # 图像序列模式：每个帧单独发送
            content = []
            for frame_b64 in frames_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                })
            content.append({"type": "text", "text": text})
            return content

    def _build_initial_message_content(self, current_b64: str, context_prompt: str, instruction_prompt: str) -> tuple[list, list, list]:
        content = []
        context_videos_b64 = []
        context_frames_b64 = []
        structure = []

        content.append({"type": "text", "text": context_prompt})
        structure.append("text(game+rules+actions)")

        if self.context_groups:
            content.append({
                "type": "text",
                "text": (
                    f"The next {len(self.context_groups)} video(s) show EXAMPLE GAMEPLAY footage from previous\n"
                    f"runs of this game ({self.session.game_name}). These are NOT necessarily good or\n"
                    "optimal play. They are provided ONLY to help you understand the\n"
                    "game's dynamics: what objects do, what causes death or rewards, and\n"
                    "how the environment reacts to actions. Do NOT treat these videos as\n"
                    "the current game state, and do NOT choose actions for them."
                ),
            })
            structure.append("text(demo label)")

            if self.video_mode:
                from ai_backends.video_encoder import frames_to_mp4_base64
                for i, group in enumerate(self.context_groups, start=1):
                    content.append({"type": "text", "text": f"Video {i}:"})
                    structure.append(f"text(video {i} label)")
                    if group.get("video_b64"):
                        vid_b64 = group["video_b64"]
                    else:
                        vid_b64 = frames_to_mp4_base64(group["frames"], fps=1.0)
                    context_videos_b64.append(vid_b64)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:video/mp4;base64,{vid_b64}"},
                    })
                    structure.append(f"video {i}")
            else:
                for group_index, group in enumerate(self.context_groups, start=1):
                    frames = group["frames"]
                    content.append({
                        "type": "text",
                        "text": (
                            f"The following {len(frames)} image(s) are the ordered frames of Video {group_index}. "
                            "Read them left-to-right, top-to-bottom as a single continuous clip."
                        ),
                    })
                    structure.append(f"text(video {group_index} frames label)")
                    context_frames_b64.extend(frames)
                    for ctx_frame in frames:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{ctx_frame}"},
                        })
                        structure.append(f"image(video {group_index} frame)")

            state_label = (
                "The demonstrations end here. The next image is the CURRENT INITIAL\n"
                "STATE of the game -- this is the state you must act on. Choose your\n"
                "next action ONLY for this current state."
            )
        else:
            state_label = (
                "The next image is the CURRENT INITIAL STATE of the game -- this is\n"
                "the state you must act on."
            )

        content.append({"type": "text", "text": state_label})
        structure.append("text(state label)")
        content.append({"type": "text", "text": "Current initial state image:"})
        structure.append("text(current image label)")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{current_b64}"},
        })
        structure.append("image")
        content.append({"type": "text", "text": instruction_prompt})
        structure.append("text(task+response-format)")
        print(f"content structure: [{', '.join(structure)}]")
        return content, context_frames_b64, context_videos_b64

    def _build_followup_message_content(self, current_b64: str, context_prompt: str, instruction_prompt: str) -> list:
        content = [
            {"type": "text", "text": context_prompt},
            {
                "type": "text",
                "text": "The next image is the CURRENT STATE after executing your previous action.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{current_b64}"},
            },
            {"type": "text", "text": instruction_prompt},
        ]
        print("content structure: [text(step result), text(current state label), image, text(task+response-format)]")
        return content

    async def _call_model(self, messages: list) -> tuple[str, dict]:
        """调用 API，失败时自动重试（指数退避 2s→4s→8s）。
        返回 (content, usage_dict)。"""
        _dump_model_messages(messages, "call_model")
        if os.environ.get("VCL_DRY_RUN_MODEL_FAKE"):
            fake_action = os.environ.get("VCL_DRY_RUN_FAKE_ACTION", "NOOP")
            return f"[Action: {fake_action}]\n(dry-run fake response)", {"input": 0, "output": 0, "total": 0}
        if os.environ.get("VCL_DRY_RUN_MODEL"):
            raise ModelDryRun("VCL_DRY_RUN_MODEL is set; model call skipped after dumping messages")

        from ai_backends.google_genai import EmptyPartsError
        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    wait = 2 ** attempt  # 2s, 4s, 8s
                    print(f"[AIPlayer] Retry {attempt}/{max_retries}, waiting {wait}s...")
                    await asyncio.sleep(wait)

                return await self._call_model_once(messages)

            except Exception as e:
                print(f"[AIPlayer] API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    # 最后一次仍是"空 parts"（thinking token 吃光预算）→ 视为"AI 未按格式输出"，
                    # 返回空文本让上游走原有的格式失败链路（追加 INVALID 提示 → 3 次格式重试 → 判死）。
                    # 其他异常（网络/HTTP/解析错误）仍按原逻辑抛出，交给上游的 _wait_for_user_retry 处理。
                    if isinstance(e, EmptyPartsError):
                        print(f"[AIPlayer] All {max_retries + 1} attempts returned empty parts. "
                              f"Treating as format failure (empty response).")
                        return "", getattr(e, "usage", {}) or {}
                    raise  # 最后一次重试也失败，抛出异常

    async def _call_model_once(self, messages: list) -> tuple[str, dict]:
        """单次调用后端 API，按 self.api_mode 分支。
        返回 (content, usage_dict)，usage_dict = {input, output, total, ...}。"""
        self._last_assistant_message = None
        # Gemini native: generateContent + inline_data
        if self.api_mode == "gemini_native":
            from ai_backends.google_genai import call_gemini_native
            return await call_gemini_native(
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        # OpenRouter: /chat/completions + video_url 扩展
        if self.api_mode == "openrouter":
            from ai_backends.openrouter import call_openrouter
            return await call_openrouter(
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                provider=self.provider,
                reasoning_effort=self.reasoning_effort,
            )
        # Kimi: /chat/completions；内嵌视频先上传为 Moonshot 文件
        if self.api_mode == "kimi":
            from ai_backends.kimi import call_kimi
            text, usage = await call_kimi(
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
            self._last_assistant_message = usage.pop("_assistant_message", None)
            return text, usage
        # OpenAI Responses API: /responses + input_text/input_image
        if self.api_mode == "openai_responses":
            from ai_backends.openai_responses import call_openai_responses
            return await call_openai_responses(
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
        # 默认: 纯 OpenAI 兼容 (image_url only, 不支持视频)

        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed")

        # 统计当前发送的媒体数量（OpenAI 不返回分项，只能从消息里数）
        img_count = vid_count = 0
        for m in messages:
            c = m.get("content", "")
            if not isinstance(c, list):
                continue
            for item in c:
                if item.get("type") != "image_url":
                    continue
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:video/"):
                    vid_count += 1
                elif url.startswith("data:image/"):
                    img_count += 1

        usage_dict = {
            "input": None, "output": None, "total": None,
            "input_image_count": img_count or None,
            "input_video_count": vid_count or None,
        }

        client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url,
            timeout=openai.Timeout(connect=10, read=120, write=30, pool=10),
        )
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        resp = await client.chat.completions.create(**request_kwargs)
        # 兼容不同 API 返回格式
        if isinstance(resp, str):
            content, sse_usage, thinking_text = _parse_openai_sse_string(resp)
            if thinking_text:
                print(f"[API Thinking] Found SSE reasoning_content, length={len(thinking_text)}")
                usage_dict["thinking"] = thinking_text
            if sse_usage:
                usage_dict["input"] = sse_usage.get(
                    "prompt_tokens", sse_usage.get("input_tokens")
                )
                usage_dict["output"] = sse_usage.get(
                    "completion_tokens", sse_usage.get("output_tokens")
                )
                usage_dict["total"] = sse_usage.get("total_tokens")
                if (
                    usage_dict["total"] is None
                    and usage_dict["input"] is not None
                    and usage_dict["output"] is not None
                ):
                    usage_dict["total"] = usage_dict["input"] + usage_dict["output"]
                print(f"[API Usage] Input: {usage_dict['input']}, Output: {usage_dict['output']}, Total: {usage_dict['total']}")
                from ai_backends.google_genai import _log_api_response
                _log_api_response("openai_compat_sse", sse_usage, f"model={self.model}")
        elif hasattr(resp, "choices"):
            msg = resp.choices[0].message
            content = msg.content
            # Debug: 看 relay 有没有透传 reasoning/thinking 字段
            thinking_text = (
                getattr(msg, "reasoning_content", None)
                or (msg.model_extra or {}).get("reasoning_content")
                or (msg.model_extra or {}).get("thinking")
            )
            if thinking_text:
                print(f"[API Thinking] Found reasoning_content, length={len(thinking_text)}")
            else:
                print(f"[API Thinking] No reasoning_content. model_extra keys: {list((msg.model_extra or {}).keys())}")
            usage_dict["thinking"] = thinking_text
            if hasattr(resp, "usage") and resp.usage:
                usage_dict["input"] = resp.usage.prompt_tokens
                usage_dict["output"] = resp.usage.completion_tokens
                usage_dict["total"] = resp.usage.total_tokens
                print(f"[API Usage] Input: {usage_dict['input']}, Output: {usage_dict['output']}, Total: {usage_dict['total']}")
                # 记录原始 usage 到日志
                from ai_backends.google_genai import _log_api_response
                raw = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else {"prompt_tokens": usage_dict["input"], "completion_tokens": usage_dict["output"], "total_tokens": usage_dict["total"]}
                _log_api_response("openai_compat", raw, f"model={self.model}")
        elif isinstance(resp, dict):
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", str(resp))
            if "usage" in resp:
                u = resp["usage"]
                usage_dict["input"] = u.get("prompt_tokens")
                usage_dict["output"] = u.get("completion_tokens")
                usage_dict["total"] = u.get("total_tokens")
                print(f"[API Usage] Input: {usage_dict['input']}, Output: {usage_dict['output']}, Total: {usage_dict['total']}")
                from ai_backends.google_genai import _log_api_response
                _log_api_response("openai_compat_dict", u, f"model={self.model}")
        else:
            content = str(resp)

        # 记录响应长度
        print(f"[Response] Length: {len(content)} chars, Content preview: {content[:200]}")

        # 检测 API 返回了 HTML（通常说明 base_url 配错了）
        stripped = (content or "").strip()
        if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html"):
            raise RuntimeError(
                f"API returned HTML instead of JSON. "
                f"Your base_url is likely incorrect: {self.base_url}\n"
                f"It should point to the API endpoint, e.g. https://xxx.com/v1"
            )
        return content, usage_dict

    def _assistant_history_message(self, response_text: str) -> dict:
        """构造历史消息；Kimi K3 需要原样回传 reasoning_content。"""
        message = self._last_assistant_message
        self._last_assistant_message = None
        if self.api_mode == "kimi" and isinstance(message, dict):
            return message
        return {"role": "assistant", "content": response_text}

    # ------------------------------------------------------------------
    # 动作解析 — 支持 numeric / natural_language 两种模式
    # ------------------------------------------------------------------

    def _parse_action(self, raw: str) -> Optional[int]:
        """
        解析 AI 响应中的动作。
        - natural_language 模式：ActionMapper → "Action: X" 格式 → 失败
        - numeric 模式：优先匹配 "Action: X"，然后任意数字
        """
        action_space_size = self.session.env.action_space_size

        # 自然语言模式：先用 ActionMapper 解析
        if self.action_mode == "natural_language":
            from action_mapper import get_action_mapper
            mapper = get_action_mapper()
            action = mapper.parse_action_from_natural_language(
                response=raw,
                game_name=self.session.game_name,
                action_space_size=action_space_size,
            )
            if action is not None:
                # 验证 action 在当前 action_info 中（simple 模式可能隐藏了部分动作）
                if action not in self.session.action_info:
                    print(f"[ActionMapper] Parsed action {action} not in current action_info, rejecting")
                    return None
                return action

            # 严格模式：natural_language 不接受数字格式
            # 如果 AI 输出了 "Action: 1" 这种数字格式，也视为无效
            print(f"[ERROR] Failed to parse natural language action. Response:\n{raw[:300]}")
            return None

        # 数字模式（numeric）
        # 1) 优先匹配 "[Action: X]" 格式
        match = re.search(r"\[\s*action\s*:\s*(\d+)\s*\]", raw, re.IGNORECASE)
        if match:
            action = int(match.group(1))
            if 0 <= action < action_space_size:
                if action not in self.session.action_info:
                    print(f"[Numeric] Parsed action {action} not in current action_info, rejecting")
                    return None
                return action

        # 2) 任意数字 fallback（仅用于 numeric 模式）
        numbers = re.findall(r"\b(\d+)\b", raw)
        for num_str in numbers:
            action = int(num_str)
            if 0 <= action < action_space_size:
                if action not in self.session.action_info:
                    continue
                return action

        return None

    async def _wait_for_user_retry(self, ws: WebSocket, error_msg: str, step: int, round_num: int) -> bool:
        """发送可恢复错误，等待用户选择重试或停止。返回 True 表示重试。"""
        self._retry_event.clear()
        self._retry_choice = None
        try:
            await ws.send_json({
                "type": "ai_error_recoverable",
                "message": f"Round {round_num} API error: {error_msg}",
                "step": step,
                "round": round_num,
            })
        except Exception:
            print(f"[AIPlayer] WebSocket broken, cannot send retry dialog. Stopping.")
            return False
        try:
            await asyncio.wait_for(self._retry_event.wait(), timeout=300)
            return self._retry_choice == "retry"
        except asyncio.TimeoutError:
            try:
                await ws.send_json({"type": "ai_error", "message": "Retry timeout (5min), stopping."})
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # 主循环 — 与 model_gen.play_game_with_vision_model 一致
    # ------------------------------------------------------------------

    async def run_loop(self, ws: WebSocket):
        """AI 游戏循环 (multi-turn dialogue, 完全复刻 model_gen 逻辑)"""
        self.is_running = True

        action_info = self.session.action_info
        action_space_size = self.session.env.action_space_size
        cumulative_reward = 0.0
        step = 0
        messages_history: List[dict] = []  # 完整对话历史（无 system message）

        # ==================================================================
        # ROUND 1: context 帧 + 初始帧，一条 message 全部打包
        # ==================================================================
        context_prompt = self._create_initial_prompt_section("context")
        instruction_prompt = self._create_initial_prompt_section("instruction")
        initial_prompt = f"{context_prompt}\n\n{instruction_prompt}"
        current_b64 = self.session.get_frame_base64(len(self.session.frames) - 1)

        first_round_content, context_frames_b64, context_videos_b64 = self._build_initial_message_content(
            current_b64, context_prompt, instruction_prompt
        )

        # 发送 ai_prompt → 前端立刻显示左边（用户消息）
        await ws.send_json({
            "type": "ai_prompt",
            "step": 0,
            "prompt": initial_prompt,
            "frame_b64": current_b64,
            "context_frames": context_frames_b64,
            "context_videos_b64": context_videos_b64,
            "video_mode": self.video_mode,
        })

        api_messages = list(messages_history) + [{"role": "user", "content": first_round_content}]
        print(f"[Message Stats] Total: {len(api_messages)}, User: {sum(1 for m in api_messages if m['role']=='user')}, Assistant: {sum(1 for m in api_messages if m['role']=='assistant')}")

        # Round 1 API 调用 + 可恢复重试
        response_text = None
        usage = {}
        while True:
            try:
                response_text, usage = await self._call_model(api_messages)
                break
            except Exception as e:
                if await self._wait_for_user_retry(ws, str(e), step=0, round_num=1):
                    print(f"[AIPlayer] User chose to retry round 1")
                    continue
                else:
                    self.is_running = False
                    return

        messages_history.append({"role": "user", "content": first_round_content})
        messages_history.append(self._assistant_history_message(response_text))

        action = self._parse_action(response_text)
        retry_count = 0
        max_retries = 3

        # 立刻推送 AI 回复（不管成功失败，右边立刻显示）
        await ws.send_json({
            "type": "ai_response",
            "step": 0,
            "raw_response": response_text,
            "is_valid": action is not None,
            "parsed_action": action,
            "action_meaning": action_info.get(action, str(action)) if action is not None else None,
            "usage": usage,
        })

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

            # 推送 retry prompt → 前端显示左边（错误提示）
            await ws.send_json({
                "type": "ai_prompt",
                "step": 0,
                "prompt": error_msg,
                "is_retry": True,
                "retry_count": retry_count,
                "max_retries": max_retries,
            })

            # Round 1 retry API 调用 + 可恢复重试
            r1_retry_ok = False
            while True:
                try:
                    response_text, usage = await self._call_model(messages_history)
                    r1_retry_ok = True
                    break
                except Exception as e:
                    if await self._wait_for_user_retry(ws, str(e), step=0, round_num=1):
                        print(f"[AIPlayer] User chose to retry round 1 action-retry {retry_count}")
                        continue
                    else:
                        self.is_running = False
                        return
            if not r1_retry_ok:
                return

            messages_history.append(self._assistant_history_message(response_text))
            action = self._parse_action(response_text)

            # 立刻推送 AI 回复
            await ws.send_json({
                "type": "ai_response",
                "step": 0,
                "raw_response": response_text,
                "is_valid": action is not None,
                "parsed_action": action,
                "action_meaning": action_info.get(action, str(action)) if action is not None else None,
                "usage": usage,
            })

        if action is None:
            await ws.send_json({
                "type": "ai_error",
                "message": f"Round 1: Failed to parse valid action after {max_retries} retries.",
            })
            self.is_running = False
            return

        # ==================================================================
        # ROUND 2+: 执行上一轮动作 → 获取新帧 → 构建 followup prompt → 调用 API
        # ==================================================================
        round_num = 2

        try:
            while self.is_running and not self.session.is_game_over:
                # 执行上一轮解析出的动作
                new_b64 = await self.session.step(action)
                step += 1
                state_info = self.session.get_state_info()
                reward = self.session.last_reward
                cumulative_reward += reward

                # 发送新帧给前端
                await ws.send_json({
                    "type": "frame",
                    "frame": new_b64,
                    "state": state_info,
                })

                # 检查游戏是否结束
                if self.session.is_game_over:
                    await ws.send_json({
                        "type": "ai_game_over",
                        "final_reward": cumulative_reward,
                        "total_steps": step,
                    })
                    break

                followup_kwargs = dict(
                    step=step,
                    last_action=action,
                    action_meaning=action_info.get(action, str(action)),
                    reward=reward,
                    cumulative_reward=cumulative_reward,
                    is_terminated=self.session.last_info.get("terminated", False),
                    is_truncated=self.session.last_info.get("truncated", False),
                    info=self.session.last_info,
                )
                followup_context_prompt = self._create_followup_prompt_section("context", **followup_kwargs)
                followup_instruction_prompt = self._create_followup_prompt_section("instruction", **followup_kwargs)
                followup_prompt = f"{followup_context_prompt}\n\n{followup_instruction_prompt}"

                followup_content = self._build_followup_message_content(
                    new_b64, followup_context_prompt, followup_instruction_prompt
                )

                # 注意：followup 不生成视频，只发送当前单帧图片
                # 视频模式仅用于初始轮（多帧上下文）

                # 先发 ai_prompt → 前端立刻显示左边（用户消息）
                await ws.send_json({
                    "type": "ai_prompt",
                    "step": step,
                    "prompt": followup_prompt,
                    "frame_b64": new_b64,
                })

                # 调用 API（根据 frame_window 裁剪历史）
                # frame_window=0: 完整历史; N>0: 保留首轮(2条) + 最近 N-1 轮(每轮2条)
                if self.frame_window > 0:
                    recent_count = (self.frame_window - 1) * 2
                    if recent_count == 0:
                        # frame_window=1: 仅保留首轮
                        history_to_send = messages_history[:2]
                    elif len(messages_history) > 2 + recent_count:
                        history_to_send = messages_history[:2] + messages_history[-recent_count:]
                    else:
                        history_to_send = list(messages_history)
                else:
                    history_to_send = list(messages_history)
                api_messages = history_to_send + [{"role": "user", "content": followup_content}]

                # 记录消息历史统计
                total_msg_count = len(api_messages)
                user_msg_count = sum(1 for m in api_messages if m["role"] == "user")
                assistant_msg_count = sum(1 for m in api_messages if m["role"] == "assistant")
                print(f"[Message Stats Step {step}] Total: {total_msg_count}, User: {user_msg_count}, Assistant: {assistant_msg_count}")

                # Round 2+ API 调用 + 可恢复重试
                response_text = None
                usage = {}
                while True:
                    try:
                        response_text, usage = await self._call_model(api_messages)
                        break
                    except Exception as e:
                        if await self._wait_for_user_retry(ws, str(e), step=step, round_num=round_num):
                            print(f"[AIPlayer] User chose to retry round {round_num}")
                            continue
                        else:
                            self.is_running = False
                            break
                if not self.is_running:
                    break

                # 写入历史
                messages_history.append({"role": "user", "content": followup_content})
                messages_history.append(self._assistant_history_message(response_text))

                # 解析下一个动作
                next_action = self._parse_action(response_text)
                retry_count = 0
                max_retries = 3

                # 立刻推送 AI 回复（不管成功失败，右边立刻显示）
                await ws.send_json({
                    "type": "ai_response",
                    "step": step,
                    "raw_response": response_text,
                    "is_valid": next_action is not None,
                    "parsed_action": next_action,
                    "action_meaning": action_info.get(next_action, str(next_action)) if next_action is not None else None,
                    "usage": usage,
                })

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

                    # 推送 retry prompt → 前端显示左边（错误提示）
                    await ws.send_json({
                        "type": "ai_prompt",
                        "step": step,
                        "prompt": error_msg,
                        "is_retry": True,
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                    })

                    # 发送错误提示给 AI
                    messages_history.append({"role": "user", "content": error_msg})

                    # Round 2+ retry API 调用 + 可恢复重试
                    r2_retry_ok = False
                    while True:
                        try:
                            response_text, usage = await self._call_model(messages_history)
                            r2_retry_ok = True
                            break
                        except Exception as e:
                            if await self._wait_for_user_retry(ws, str(e), step=step, round_num=round_num):
                                print(f"[AIPlayer] User chose to retry round {round_num} action-retry {retry_count}")
                                continue
                            else:
                                self.is_running = False
                                break
                    if not r2_retry_ok:
                        break

                    messages_history.append(self._assistant_history_message(response_text))
                    next_action = self._parse_action(response_text)

                    # 立刻推送 AI 回复
                    await ws.send_json({
                        "type": "ai_response",
                        "step": step,
                        "raw_response": response_text,
                        "is_valid": next_action is not None,
                        "parsed_action": next_action,
                        "action_meaning": action_info.get(next_action, str(next_action)) if next_action is not None else None,
                        "usage": usage,
                    })

                if next_action is None:
                    await ws.send_json({
                        "type": "ai_error",
                        "message": f"Round {round_num}: Failed to parse valid action after {max_retries} retries. Stopping game.",
                    })
                    break

                action = next_action
                round_num += 1

                # 小延迟防止刷屏
                await asyncio.sleep(0.1)

        except WebSocketDisconnect:
            pass
        finally:
            self.is_running = False

    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()

    def start(self, ws: WebSocket):
        self._task = asyncio.create_task(self.run_loop(ws))
