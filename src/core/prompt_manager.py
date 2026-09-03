"""Prompt 模板管理器"""
import json
from pathlib import Path
from typing import Optional


class PromptManager:
    """从 prompt/templates/ 加载 prompt 模板，支持 numeric / natural_language 两种模式"""

    _PROMPT_DIR = Path(__file__).parent.parent.parent / "prompt"

    def __init__(self):
        self._initial: dict = {}
        self._followup: dict = {}
        self._game_rules: dict = {}
        self._detailed_game_rules: dict = {}
        self._load()

    def _load(self):
        try:
            p = self._PROMPT_DIR / "templates" / "initial.json"
            if p.exists():
                self._initial = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[PromptManager] Warning: cannot load initial.json: {e}")
        try:
            p = self._PROMPT_DIR / "templates" / "followup.json"
            if p.exists():
                self._followup = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[PromptManager] Warning: cannot load followup.json: {e}")
        try:
            p = self._PROMPT_DIR / "templates" / "game_rules.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                self._game_rules = data.get("games", {})
        except Exception as e:
            print(f"[PromptManager] Warning: cannot load game_rules.json: {e}")
        try:
            p = self._PROMPT_DIR / "templates" / "detailed_game_rules.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                self._detailed_game_rules = data.get("games", {})
        except Exception as e:
            print(f"[PromptManager] Warning: cannot load detailed_game_rules.json: {e}")

    def get_game_rules(self, game_name: str, detailed_game_rules:bool = False) -> str:
        """从 game_rules.json 获取游戏规则（无 fallback）"""
        if detailed_game_rules:
            return self._detailed_game_rules.get(game_name, "")
        return self._game_rules.get(game_name, "")

    def build_initial_prompt(
        self,
        mode: str,
        game_name: str,
        action_info: dict,
        game_rules: str = "",
        video_description: str = "",
        context_frames: list = None,
        max_steps: int = None,
        section: str = "full",
    ) -> str:
        """构建 initial prompt；fallback 到原始硬编码逻辑"""
        mode_cfg = self._initial.get("modes", {}).get(mode)
        if not mode_cfg:
            fallback = self._build_initial_fallback(
                game_name, action_info, game_rules, video_description, context_frames or [])
            if section == "context":
                return fallback.split("\nYour task:", 1)[0].rstrip()
            if section == "instruction" and "\nYour task:" in fallback:
                return "Your task:" + fallback.split("\nYour task:", 1)[1]
            return fallback

        # 动作列表
        fmt = mode_cfg.get("action_list_format", "  {action_id}: {action_meaning}")
        if mode == "numeric":
            action_list = "\n".join(
                fmt.format(action_id=k, action_meaning=v) for k, v in action_info.items()
            )
        else:
            action_list = "\n".join(
                fmt.format(action_meaning=v) for v in action_info.values()
            )

        # game_rules / video_context 预格式化
        rules_str = ""
        if game_rules:
            rules_fmt = mode_cfg.get("game_rules_format", "\n\nGame Rules:\n{game_rules}")
            rules_str = rules_fmt.format(game_rules=game_rules)

        max_steps_str=""
        if max_steps:
            max_steps_fmt=mode_cfg.get("max_steps_format","\n\nMax Steps: {max_steps}")
            max_steps_str=max_steps_fmt.format(max_steps=max_steps)

        video_str = ""
        if context_frames and video_description:
            vid_fmt = mode_cfg.get("video_context_format", "\n\nExample gameplay videos:\n{video_description}")
            video_str = vid_fmt.format(video_description=video_description)

        if section == "context":
            template_lines = mode_cfg.get("context_template", mode_cfg["template"])
        elif section == "instruction":
            template_lines = mode_cfg.get("instruction_template", mode_cfg["template"])
        else:
            template_lines = mode_cfg["template"]
        template = "\n".join(template_lines)
        return template.format(
            game_name=game_name,
            game_rules=rules_str,
            video_context=video_str,
            action_list=action_list,
            max_steps=max_steps_str,
        )

    def build_followup_prompt(
        self,
        mode: str,
        step: int,
        last_action: int,
        action_meaning: str,
        reward: float,
        cumulative_reward: float,
        is_terminated: bool,
        is_truncated: bool,
        info: dict,
        hide_reward: bool = False,
        hide_step=False,
        section: str = "full",
    ) -> str:
        """构建 followup prompt；fallback 到原始硬编码逻辑"""
        mode_cfg = self._followup.get("modes", {}).get(mode)
        if not mode_cfg:
            return self._build_followup_fallback(
                step, last_action, action_meaning, reward,
                cumulative_reward, is_terminated, is_truncated, info)

        status_tpls = self._followup.get("status_templates", {})
        status_info = ""
        if is_terminated:
            status_info = status_tpls.get("terminated", "\n⚠️ Game TERMINATED (episode ended naturally).")
        elif is_truncated:
            status_info = status_tpls.get("truncated", "\n⚠️ Game TRUNCATED (time limit or other constraint).")

        life_info = ""
        if info.get("life_lost"):
            tpl = status_tpls.get("life_lost", "\n💀 Life lost! Lives: {lives_before} → {lives_after}{auto_respawn_info}")
            auto_respawn_info = " (auto-respawned)" if info.get("auto_respawned") else ""
            life_info = tpl.format(
                lives_before=info.get("lives_before", "?"),
                lives_after=info.get("lives_after", "?"),
                auto_respawn_info=auto_respawn_info,
            )

        if section == "context":
            lines = mode_cfg.get("context_template", mode_cfg["template"])
        elif section == "instruction":
            lines = mode_cfg.get("instruction_template", mode_cfg["template"])
        else:
            lines = mode_cfg["template"]
        if hide_reward:
            lines = [l for l in lines if "{reward" not in l and "{cumulative_reward" not in l]
        if hide_step:
            lines = [l for l in lines if "{step" not in l]
        template = "\n".join(lines)
        kwargs = dict(
            step=step,
            last_action_meaning=action_meaning,
            reward=reward,
            cumulative_reward=cumulative_reward,
            status_info=status_info,
            life_info=life_info,
        )
        if mode == "numeric":
            kwargs["last_action_id"] = last_action
        return template.format(**kwargs)

    # ------------------------------------------------------------------
    # Fallback — 与原始硬编码逻辑完全一致
    # ------------------------------------------------------------------

    @staticmethod
    def _build_initial_fallback(game_name, action_info, game_rules,
                                 video_description, context_frames) -> str:
        action_list = "\n".join(f"  {aid}: {m}" for aid, m in action_info.items())
        sections = [f"You are playing the game: {game_name}"]
        if game_rules:
            sections.append(f"\nGame Rules:\n{game_rules}")
        sections.append(f"\nAvailable actions:\n{action_list}")
        sections.append("\nYour task: Choose the best action for the CURRENT INITIAL STATE shown in the image above.")
        sections.append("""
IMPORTANT - Response format:
Your response MUST start with your chosen action on the FIRST LINE in this exact format:
[Action: <number>]

For example:
[Action: 1]

The square brackets [ ] are REQUIRED. The [Action: X] line MUST be the FIRST line of your response.
You may add brief reasoning AFTER the action line.

What action do you choose?""")
        return "\n".join(sections)

    @staticmethod
    def _build_followup_fallback(step, last_action, action_meaning, reward,
                                  cumulative_reward, is_terminated, is_truncated, info) -> str:
        status = ""
        if is_terminated:
            status = "\n⚠️ Game TERMINATED (episode ended naturally)."
        elif is_truncated:
            status = "\n⚠️ Game TRUNCATED (time limit or other constraint)."
        life_info = ""
        if info.get("life_lost"):
            lives_before = info.get("lives_before", "?")
            lives_after = info.get("lives_after", "?")
            life_info = f"\n💀 Life lost! Lives: {lives_before} → {lives_after}"
            if info.get("auto_respawned"):
                life_info += " (auto-respawned)"
        return f"""Step {step} - Result of your last action:

Previous action: {last_action} ({action_meaning})
Reward: {reward:.2f}
Cumulative reward: {cumulative_reward:.2f}{status}{life_info}

This is the CURRENT STATE after executing your action.

What action do you choose next?

IMPORTANT: Your response MUST start with [Action: <number>] on the FIRST LINE.
You may add brief reasoning after."""


# 全局单例
_prompt_manager: Optional["PromptManager"] = None


def get_prompt_manager() -> "PromptManager":
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
