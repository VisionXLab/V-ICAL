"""
ActionMapper: 自然语言到动作编号的映射器

从 prompt/action_mappings/natural_language.json 加载配置，
支持精确匹配和模糊匹配（difflib.SequenceMatcher）。
"""

import json
import re
from pathlib import Path
from typing import Optional


class ActionMapper:
    """自然语言到动作编号的映射器"""

    def __init__(self, mapping_file: str = "prompt/action_mappings/natural_language.json"):
        self.mapping_file = Path(mapping_file)
        self._mappings: dict = {}        # game_name → raw config
        self._reverse: dict = {}         # game_name → {text → action_id}
        self._loaded = False
        self._load()

    # ------------------------------------------------------------------
    # 加载与初始化
    # ------------------------------------------------------------------

    def _load(self):
        if not self.mapping_file.exists():
            print(f"[ActionMapper] Warning: mapping file not found: {self.mapping_file}")
            return
        try:
            with open(self.mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._mappings = data.get("games", {})
            self._build_reverse()
            self._loaded = True
        except Exception as e:
            print(f"[ActionMapper] Error loading mappings: {e}")

    def _build_reverse(self):
        """构建反向查找表：game → {text → action_id}，直接使用原始文本，不做规范化"""
        for game, game_data in self._mappings.items():
            lookup = {}
            for action in game_data.get("actions", []):
                action_id = action["id"]
                for text in [action["canonical"]] + action.get("variants", []):
                    if text:
                        if text in lookup and lookup[text] != action_id:
                            print(f"[ActionMapper] WARNING: Collision in {game}: '{text}' maps to both {lookup[text]} and {action_id}")
                        lookup[text] = action_id
            self._reverse[game] = lookup

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def is_supported(self, game_name: str) -> bool:
        return game_name in self._reverse

    def get_canonical_name(self, game_name: str, action_id: int) -> Optional[str]:
        """返回动作的规范名称"""
        game_data = self._mappings.get(game_name, {})
        for action in game_data.get("actions", []):
            if action["id"] == action_id:
                return action["canonical"]
        return None

    def parse_action_from_natural_language(
        self,
        response: str,
        game_name: str,
        action_space_size: int,
    ) -> Optional[int]:
        """
        从 AI 响应中解析动作编号。

        严格只匹配 "[Action: <text>]" 方括号格式，不做全文子串扫描。
        如果 AI 未使用方括号格式，返回 None → 触发重试。
        """
        if not self.is_supported(game_name):
            return None

        # 提取 "[Action: <text>]" 中的目标文本
        # 严格要求方括号格式，避免从推理过程中误匹配
        match = re.search(r"\[\s*action\s*:\s*(.+?)\s*\]", response, re.IGNORECASE)
        if match:
            action_text = match.group(1).strip()
            print(f"[ActionMapper] Extracted action text: '{action_text}'")
            action_id = self._match(action_text, game_name)
            print(f"[ActionMapper] Match result: action_id={action_id}")
            if action_id is not None and 0 <= action_id < action_space_size:
                return action_id

        print(f"[ActionMapper] No [Action: ...] bracket format found in response")
        return None

    # ------------------------------------------------------------------
    # 内部匹配
    # ------------------------------------------------------------------

    def _match(self, text: str, game_name: str) -> Optional[int]:
        """精确匹配，匹配不到返回 None → 触发重试"""
        lookup = self._reverse.get(game_name, {})
        print(f"[ActionMapper] _match: text='{text}'")

        if text in lookup:
            result = lookup[text]
            print(f"[ActionMapper] Exact match: '{text}' → action_id={result}")
            return result

        print(f"[ActionMapper] No match for '{text}'")
        return None


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_instance: Optional[ActionMapper] = None


def get_action_mapper() -> ActionMapper:
    global _instance
    if _instance is None:
        _instance = ActionMapper()
    return _instance
