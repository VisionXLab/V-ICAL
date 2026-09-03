"""全局状态管理

集中管理会话、AI玩家等全局状态变量
"""
from typing import Dict
from src.core import GameSession
from src.ai import AIPlayer

# 会话存储
sessions: Dict[str, GameSession] = {}

# AI玩家存储
ai_players: Dict[str, AIPlayer] = {}
