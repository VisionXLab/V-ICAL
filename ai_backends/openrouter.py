"""
OpenRouter backend.

与普通 OpenAI 兼容接口 (`/chat/completions`) 相同，但把 messages 里
`image_url` 承载的 `data:video/*;base64,...` 改写成 OpenRouter 私有扩展的
`video_url` content type（OpenAI 标准不支持 video_url，仅 OpenRouter 有）。

Docs: https://openrouter.ai/docs/guides/overview/multimodal/videos
"""
import copy
from typing import Any
import httpx

from .google_genai import _log_api_response


def _rewrite_content_for_openrouter(messages: list) -> list:
    """Deep-copy messages 并把 image_url(data:video/*) → video_url。图片/文字不变。"""
    new_messages = []
    for msg in messages:
        raw = msg.get("content")
        if not isinstance(raw, list):
            new_messages.append(msg)
            continue
        new_content = []
        for item in raw:
            if item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:video/") or url.startswith("video/"):
                    new_content.append({"type": "video_url", "video_url": {"url": url}})
                    continue
            new_content.append(copy.deepcopy(item))
        new_messages.append({**msg, "content": new_content})
    return new_messages


def _count_media(messages: list) -> tuple[int, int, int]:
    """返回 (image_count, video_count, base64_video_count)."""
    img_count = vid_count = b64_vid_count = 0
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for item in c:
            t = item.get("type")
            if t == "image_url":
                img_count += 1
            elif t == "video_url":
                vid_count += 1
                url = item.get("video_url", {}).get("url", "")
                if url.startswith("data:"):
                    b64_vid_count += 1
    return img_count, vid_count, b64_vid_count


async def call_openrouter(
    messages: list,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 50000,
    provider: dict | None = None,
    reasoning_effort: str = "",
) -> tuple[str, dict]:
    """通过 OpenRouter /chat/completions 发送请求，带 video_url 扩展。
    provider: OpenRouter routing 配置（如 {"only": ["google-vertex"]}）。
        传 None 时，若 messages 里包含 base64 视频则自动限制到 google-vertex
        （Google AI Studio provider 只接 YouTube 链接，base64 必须走 Vertex）。
    reasoning_effort: 空字符串时不发送；非空值原样传给兼容接口。
    返回 (response_text, usage_dict)。"""
    payload_messages = _rewrite_content_for_openrouter(messages)
    img_count, vid_count, b64_vid_count = _count_media(payload_messages)

    if provider is None and b64_vid_count > 0:
        if 'seed' in model.lower():
            provider={"only":["seed"]}
        else:
            provider = {"only": ["google-vertex"]}
        print(f"[OpenRouter] auto-routing to google-vertex (base64 video x{b64_vid_count})")

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": payload_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider:
        payload["provider"] = provider
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"[OpenRouter] POST {url}  (images={img_count}, videos={vid_count})")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API HTTP {resp.status_code}:\n{resp.text[:500]}")

    data = resp.json()

    usage_raw = data.get("usage", {}) or {}
    _log_api_response("openrouter", usage_raw, f"model={model}")

    try:
        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        thinking_text = (
            msg.get("reasoning_content")
            or msg.get("reasoning")
            or msg.get("thinking")
        )
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response: {e}\n{str(data)[:500]}")

    usage_dict = {
        "input": usage_raw.get("prompt_tokens"),
        "output": usage_raw.get("completion_tokens"),
        "total": usage_raw.get("total_tokens"),
        "thoughts": None,
        "thinking": thinking_text,
        "input_image_count": img_count or None,
        "input_video_count": vid_count or None,
    }

    print(f"[OpenRouter Usage] Input: {usage_dict['input']}, "
          f"Output: {usage_dict['output']}, Total: {usage_dict['total']}")
    print(f"[OpenRouter Response] Length: {len(text)} chars, Preview: {text[:200]}")
    return text, usage_dict
