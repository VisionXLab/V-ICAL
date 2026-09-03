"""API 层模块

包含路由定义和 WebSocket 处理
"""
from .routes import register_routes
from .websocket_handler import register_websocket
from .state import sessions, ai_players

__all__ = [
    "register_routes",
    "register_websocket",
    "sessions",
    "ai_players",
]
