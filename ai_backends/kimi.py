"""Kimi (Moonshot) Chat Completions backend with video-file upload support."""

import base64
import copy
import hashlib
from typing import Any

import httpx

from .google_genai import _log_api_response


def _decode_data_video(url: str) -> tuple[bytes, str] | None:
    if not url.startswith("data:video/") or "," not in url:
        return None
    header, encoded = url.split(",", 1)
    if ";base64" not in header:
        return None
    mime_type = header[5:].split(";", 1)[0]
    try:
        return base64.b64decode(encoded, validate=True), mime_type
    except ValueError as exc:
        raise RuntimeError("Kimi video input contains invalid base64 data") from exc


async def _upload_video(client, headers, base_url, video_bytes, mime_type, index):
    extension = mime_type.split("/", 1)[-1].replace("x-", "") or "mp4"
    response = await client.post(
        f"{base_url}/files",
        headers=headers,
        data={"purpose": "video"},
        files={"file": (f"vcl-context-{index}.{extension}", video_bytes, mime_type)},
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Kimi file upload HTTP {response.status_code}:\n{response.text[:500]}")
    try:
        return str(response.json()["id"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Kimi file upload response: {response.text[:500]}") from exc


async def _rewrite_video_content(client, headers, base_url, messages, uploaded_ids, video_hashes):
    """Upload embedded videos and return Kimi-compatible copied messages."""
    rewritten = []
    image_count = video_count = 0
    for message in messages:
        raw_content = message.get("content")
        if not isinstance(raw_content, list):
            rewritten.append(copy.deepcopy(message))
            continue
        content = []
        for part in raw_content:
            part_copy = copy.deepcopy(part)
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                decoded = _decode_data_video(url)
                if decoded is None:
                    image_count += 1
                else:
                    video_bytes, mime_type = decoded
                    video_hashes.append(hashlib.sha256(video_bytes).hexdigest())
                    file_id = await _upload_video(
                        client, headers, base_url, video_bytes, mime_type,
                        len(uploaded_ids) + 1,
                    )
                    uploaded_ids.append(file_id)
                    video_count += 1
                    part_copy = {"type": "video_url", "video_url": {"url": f"ms://{file_id}"}}
            content.append(part_copy)
        rewritten.append({**copy.deepcopy(message), "content": content})
    return rewritten, image_count, video_count


async def call_kimi(
    messages: list,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = 50000,
    reasoning_effort: str = "",
) -> tuple[str, dict]:
    """Call Kimi K3, translating embedded MP4 data URLs into ``ms://`` files."""
    api_base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    uploaded_ids: list[str] = []
    video_hashes: list[str] = []
    usage_result: dict | None = None
    if reasoning_effort and reasoning_effort not in {"low", "high", "max"}:
        raise ValueError("Kimi K3 reasoning_effort must be one of: low, high, max (or empty)")

    timeout = httpx.Timeout(connect=10, read=180, write=180, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            payload_messages, image_count, video_count = await _rewrite_video_content(
                client, headers, api_base, messages, uploaded_ids, video_hashes
            )
            payload: dict[str, Any] = {
                "model": model,
                "messages": payload_messages,
                "max_completion_tokens": max_tokens,
            }
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            print(f"[Kimi] POST {api_base}/chat/completions (images={image_count}, videos={video_count})")
            response = await client.post(
                f"{api_base}/chat/completions",
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Kimi API HTTP {response.status_code}:\n{response.text[:500]}")
            try:
                data = response.json()
                assistant_message = data["choices"][0]["message"]
                text = assistant_message.get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected Kimi response: {response.text[:500]}") from exc

            usage_raw = data.get("usage", {}) or {}
            _log_api_response("kimi", usage_raw, f"model={model}")
            usage = {
                "input": usage_raw.get("prompt_tokens"),
                "output": usage_raw.get("completion_tokens"),
                "total": usage_raw.get("total_tokens"),
                "thinking": assistant_message.get("reasoning_content"),
                "input_image_count": image_count or None,
                "input_video_count": video_count or None,
                "input_video_sha256": video_hashes,
                "_assistant_message": copy.deepcopy(assistant_message),
            }
            usage_result = usage
            print(f"[Kimi Usage] Input: {usage['input']}, Output: {usage['output']}, Total: {usage['total']}")
            return text, usage
        finally:
            deleted_count = 0
            cleanup_errors = []
            for file_id in uploaded_ids:
                try:
                    delete_response = await client.delete(f"{api_base}/files/{file_id}", headers=headers)
                    if delete_response.status_code not in (200, 204):
                        cleanup_errors.append(f"{file_id}: HTTP {delete_response.status_code}")
                        print(f"[Kimi] Warning: failed to delete temporary file {file_id} (HTTP {delete_response.status_code})")
                    else:
                        deleted_count += 1
                except Exception as exc:
                    cleanup_errors.append(f"{file_id}: {exc}")
                    print(f"[Kimi] Warning: failed to delete temporary file {file_id}: {exc}")
            if usage_result is not None:
                usage_result["temporary_files_uploaded"] = len(uploaded_ids)
                usage_result["temporary_files_deleted"] = deleted_count
                usage_result["temporary_file_cleanup_errors"] = cleanup_errors
