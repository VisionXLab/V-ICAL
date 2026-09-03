"""AI backends for different video processing modes"""

from .video_encoder import frames_to_mp4_base64, video_file_to_frames
from .google_genai import call_gemini_native, openai_messages_to_gemini, build_gemini_url
from .openrouter import call_openrouter
from .kimi import call_kimi
from .openai_responses import (
    call_openai_responses,
    openai_messages_to_responses_input,
)

__all__ = [
    "frames_to_mp4_base64",
    "video_file_to_frames",
    "call_gemini_native",
    "openai_messages_to_gemini",
    "build_gemini_url",
    "call_openrouter",
    "call_kimi",
    "call_openai_responses",
    "openai_messages_to_responses_input",
]
