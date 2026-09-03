"""OpenAI Responses API backend.

Converts the project's Chat Completions-style message history into the
Responses API input format.  This backend is deliberately non-streaming:
some OpenAI-compatible relays intermittently return a bare SSE ``[DONE]``
marker from ``/chat/completions`` even when streaming was not requested.
"""
from __future__ import annotations

import json
from typing import Any

import openai

from .google_genai import _log_api_response


class EmptyResponsesCompletionError(RuntimeError):
    """The Responses endpoint completed without any model text."""


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _convert_content_item(item: dict) -> dict:
    item_type = item.get("type")
    if item_type in ("text", "input_text"):
        return {"type": "input_text", "text": item.get("text", "")}

    if item_type in ("image_url", "input_image"):
        image_value = item.get("image_url")
        if isinstance(image_value, dict):
            image_url = image_value.get("url", "")
            detail = image_value.get("detail")
        else:
            image_url = image_value or ""
            detail = item.get("detail")

        if image_url.startswith("data:video/") or image_url.startswith("video/"):
            raise ValueError(
                "openai_responses does not support video inputs; "
                "set ai.video_mode to false in config.json"
            )
        if not image_url:
            raise ValueError("Responses API image input is missing image_url")

        converted = {"type": "input_image", "image_url": image_url}
        if detail:
            converted["detail"] = detail
        return converted

    raise ValueError(f"Unsupported Responses API content type: {item_type!r}")


def openai_messages_to_responses_input(messages: list) -> list:
    """Convert Chat Completions-style messages to Responses API input items."""
    converted_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            converted_content: str | list = content
        elif isinstance(content, list):
            converted_content = [_convert_content_item(item) for item in content]
        else:
            converted_content = str(content)
        converted_messages.append({"role": role, "content": converted_content})
    return converted_messages


def _count_input_images(response_input: list) -> int:
    count = 0
    for message in response_input:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count += sum(1 for item in content if item.get("type") == "input_image")
    return count


def _extract_output_text(response: Any) -> str:
    output_text = _item_value(response, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    text_parts = []
    for output_item in _item_value(response, "output", []) or []:
        if _item_value(output_item, "type") != "message":
            continue
        for content_item in _item_value(output_item, "content", []) or []:
            if _item_value(content_item, "type") != "output_text":
                continue
            text = _item_value(content_item, "text")
            if isinstance(text, str):
                text_parts.append(text)
    return "".join(text_parts)


def _parse_string_response(raw: str) -> dict:
    stripped = raw.strip()
    if stripped in ("[DONE]", "data: [DONE]"):
        raise EmptyResponsesCompletionError(
            "OpenAI-compatible Responses API returned only the SSE [DONE] marker"
        )
    if stripped.startswith(("data:", "event:")):
        raise EmptyResponsesCompletionError(
            "OpenAI-compatible Responses API returned an unexpected SSE body "
            "for stream=false"
        )
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Unexpected non-JSON Responses API response: {stripped[:200]}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Unexpected Responses API response type: {type(parsed).__name__}"
        )
    return parsed


async def call_openai_responses(
    messages: list,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 50000,
    reasoning_effort: str = "",
) -> tuple[str, dict]:
    """Call ``POST /responses`` in explicit non-streaming mode."""
    response_input = openai_messages_to_responses_input(messages)
    image_count = _count_input_images(response_input)

    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=openai.Timeout(connect=10, read=120, write=30, pool=10),
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "input": response_input,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "stream": False,
        "store": False,
    }
    if reasoning_effort:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    print(
        f"[OpenAI Responses] POST {base_url.rstrip('/')}/responses "
        f"(messages={len(response_input)}, images={image_count}, stream=false)"
    )
    response = await client.responses.create(**request_kwargs)
    if isinstance(response, str):
        response = _parse_string_response(response)

    error = _item_value(response, "error")
    if error:
        if hasattr(error, "model_dump"):
            error = error.model_dump()
        raise RuntimeError(f"OpenAI Responses API error: {error}")

    status = _item_value(response, "status")
    if status == "failed":
        raise RuntimeError("OpenAI Responses API returned status=failed")

    content = _extract_output_text(response)
    if not content:
        raise EmptyResponsesCompletionError(
            f"OpenAI Responses API returned no output_text (status={status})"
        )

    usage = _item_value(response, "usage")
    if hasattr(usage, "model_dump"):
        raw_usage = usage.model_dump()
    elif isinstance(usage, dict):
        raw_usage = usage
    else:
        raw_usage = {}

    usage_dict = {
        "input": _item_value(usage, "input_tokens"),
        "output": _item_value(usage, "output_tokens"),
        "total": _item_value(usage, "total_tokens"),
        "input_image_count": image_count or None,
    }
    if raw_usage:
        _log_api_response("openai_responses", raw_usage, f"model={model}")

    print(
        f"[OpenAI Responses Usage] Input: {usage_dict['input']}, "
        f"Output: {usage_dict['output']}, Total: {usage_dict['total']}"
    )
    print(
        f"[OpenAI Responses] Length: {len(content)} chars, "
        f"Preview: {content[:200]}"
    )
    return content, usage_dict
