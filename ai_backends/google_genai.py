"""
Gemini native generateContent backend.
video_mode 下绕过 OpenAI-compatible relay 的 image_url 限制，
直接向 Gemini generateContent 端点发送 inlineData 格式的视频/图片。
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import httpx

# API 响应日志目录
_API_LOG_DIR = Path(__file__).parent.parent / "api_logs"


class EmptyPartsError(RuntimeError):
    """Gemini thinking model 把输出预算全烧在思考上，candidates[0].content 只有 role 无 parts。
    上层收到时应视为"AI 未按格式输出"，走原有的格式失败链路（而非网络重试弹窗）。
    usage 字段保留本次请求的 token 统计供上游展示。"""
    def __init__(self, msg: str, usage: dict | None = None):
        super().__init__(msg)
        self.usage = usage or {}


def _log_api_response(backend: str, raw_usage: dict, round_info: str = ""):
    """将 API 原始 usage 响应追加写入日志文件（按天分文件）。"""
    try:
        _API_LOG_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        log_file = _API_LOG_DIR / f"{today}.jsonl"
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "backend": backend,
            "raw_usage": raw_usage,
        }
        if round_info:
            entry["info"] = round_info
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[APILog] Failed to write log: {e}")


def _parse_modality_tokens(usage_metadata: dict) -> dict:
    """从 usageMetadata.promptTokensDetails 解析各 modality 的 token 数。
    返回 {"text": N, "image": N, "video": N}，缺失字段为 None。"""
    details = usage_metadata.get("promptTokensDetails") or []
    result = {}
    for entry in details:
        modality = entry.get("modality", "").upper()
        count = entry.get("tokenCount")
        if modality == "TEXT":
            result["text"] = count
        elif modality == "IMAGE":
            result["image"] = count
        elif modality == "VIDEO":
            result["video"] = count
    return result


def _convert_content_item(item: dict) -> dict:
    """OpenAI content item → Gemini part"""
    if item.get("type") == "text":
        return {"text": item["text"]}
    if item.get("type") == "image_url":
        url: str = item["image_url"]["url"]
        match = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
        if match:
            return {"inline_data": {"mime_type": match.group(1), "data": match.group(2)}}
        return {"text": f"[unsupported image_url: {url[:80]}...]"}
    return {"text": str(item)}


def _convert_message(msg: dict) -> dict:
    """OpenAI message → Gemini content entry"""
    role = "model" if msg["role"] == "assistant" else "user"
    raw = msg["content"]
    if isinstance(raw, str):
        parts = [{"text": raw}]
    elif isinstance(raw, list):
        parts = [_convert_content_item(item) for item in raw]
    else:
        parts = [{"text": str(raw)}]
    return {"role": role, "parts": parts}


def openai_messages_to_gemini(messages: list) -> list:
    """OpenAI messages list → Gemini contents list"""
    return [_convert_message(m) for m in messages]


def build_gemini_url(base_url: str, model: str) -> str:
    """
    构建 Gemini generateContent URL。
    "https://yunwu.ai/v1" + "gemini-3-pro" → "https://yunwu.ai/v1beta/models/gemini-3-pro:generateContent"
    """
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        stripped = stripped[:-3]
    return f"{stripped}/v1beta/models/{model}:generateContent"


async def call_gemini_native(
    messages: list,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 50000,
) -> tuple[str, dict]:
    """将 OpenAI 格式 messages 转换并发送到 Gemini generateContent 端点。
    返回 (response_text, usage_dict)。"""
    url = build_gemini_url(base_url, model)
    contents = openai_messages_to_gemini(messages)

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "mediaResolution": "MEDIA_RESOLUTION_HIGH",
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"[GeminiNative] POST {url}")
    print(f"[GeminiNative] contents count: {len(contents)}")
    # 调试：打印每条 content 的结构
    for i, c in enumerate(contents):
        role = c.get("role", "?")
        parts_summary = []
        for p in c.get("parts", []):
            if "text" in p:
                parts_summary.append(f"text({len(p['text'])} chars)")
            elif "inline_data" in p:
                mime = p["inline_data"].get("mime_type", "?")
                data_len = len(p["inline_data"].get("data", ""))
                parts_summary.append(f"inline_data({mime}, {data_len} bytes b64)")
            else:
                parts_summary.append(f"unknown({list(p.keys())})")
        print(f"[GeminiNative]   [{i}] role={role} parts=[{', '.join(parts_summary)}]")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API HTTP {resp.status_code}:\n{resp.text[:500]}")

    data = resp.json()

    raw_usage = data.get("usageMetadata", {})
    _log_api_response("gemini_native", raw_usage, f"model={model}")
    thoughts = raw_usage.get("thoughtsTokenCount")
    modality = _parse_modality_tokens(raw_usage)
    usage_dict = {
        "input": raw_usage.get("promptTokenCount"),
        "output": raw_usage.get("candidatesTokenCount"),
        "thoughts": thoughts,
        "total": raw_usage.get("totalTokenCount"),
        "input_text_tokens": modality.get("text"),
        "input_image_tokens": modality.get("image"),
        "input_video_tokens": modality.get("video"),
    }
    if raw_usage:
        print(f"[GeminiNative Usage] Input: {usage_dict['input']}, "
              f"Output: {usage_dict['output']}, "
              f"Thoughts: {thoughts}, "
              f"Total: {usage_dict['total']} | "
              f"Text: {modality.get('text')} tok, "
              f"Image: {modality.get('image')} tok, "
              f"Video: {modality.get('video')} tok")

    try:
        candidate = data["candidates"][0]
        content = candidate.get("content") or {}
        parts = content.get("parts")
        if not parts:
            finish_reason = candidate.get("finishReason", "?")
            raise EmptyPartsError(
                f"Empty parts (finishReason={finish_reason}, "
                f"candidatesTokenCount={raw_usage.get('candidatesTokenCount')}, "
                f"thoughtsTokenCount={thoughts})",
                usage=usage_dict,
            )
        thought_parts = [p for p in parts if p.get("thought", False)]
        response_parts = [p for p in parts if not p.get("thought", False)]
        thinking_text = "\n".join(p.get("text", "") for p in thought_parts) or None
        text = (response_parts[0] if response_parts else parts[0])["text"]
    except EmptyPartsError:
        raise
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected Gemini response: {e}\n{str(data)[:500]}")

    usage_dict["thinking"] = thinking_text
    print(f"[GeminiNative Response] Length: {len(text)} chars, Preview: {text[:200]}")
    return text, usage_dict
