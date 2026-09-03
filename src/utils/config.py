"""配置文件加载"""
import json
from pathlib import Path

# 配置文件路径（相对于项目根目录）
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"


def load_config() -> dict:
    """加载 config.json（不存在则返回空 dict）"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# 全局默认配置
DEFAULT_CONFIG = load_config()
